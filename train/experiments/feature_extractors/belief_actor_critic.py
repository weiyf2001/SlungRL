from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Normal


TensorDict = Dict[str, torch.Tensor]


@dataclass
class NetworkConfig:
    # Actor-side observation dims
    obs_dim: int
    err_dim: int
    prev_act_dim: int
    ref_dim: int

    # Critic privileged state dim
    priv_dim: int

    # Action
    act_dim: int = 4

    # Hidden sizes
    current_hidden_dim: int = 128
    preview_hidden_dim: int = 64
    fast_gru_hidden_dim: int = 128
    slow_gru_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    actor_feature_dim: int = 128
    critic_feature_dim: int = 128
    priv_hidden_dim: int = 128

    # MLP width
    mlp_width: int = 256

    # Action std
    init_log_std: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0

    # Optional flags
    use_state_dependent_std: bool = False
    use_layer_norm: bool = True


@dataclass
class AuxPredictionConfig:
    pred_payload_vel_dim: int = 3
    pred_cable_dir_dim: int = 3
    pred_cable_dir_rate_dim: int = 3
    pred_mode_logit_dim: int = 2

    pred_mass_dim: int = 1
    pred_length_dim: int = 1
    pred_disturbance_ctx_dim: int = 3
    pred_actuator_ctx_dim: int = 1


@dataclass
class RunningBeliefState:
    actor_fast_h: torch.Tensor
    actor_slow_h: torch.Tensor
    critic_fast_h: torch.Tensor
    critic_slow_h: torch.Tensor

    def detach(self) -> "RunningBeliefState":
        return RunningBeliefState(
            actor_fast_h=self.actor_fast_h.detach(),
            actor_slow_h=self.actor_slow_h.detach(),
            critic_fast_h=self.critic_fast_h.detach(),
            critic_slow_h=self.critic_slow_h.detach(),
        )

    def clone(self) -> "RunningBeliefState":
        return RunningBeliefState(
            actor_fast_h=self.actor_fast_h.clone(),
            actor_slow_h=self.actor_slow_h.clone(),
            critic_fast_h=self.critic_fast_h.clone(),
            critic_slow_h=self.critic_slow_h.clone(),
        )


@dataclass
class ParsedBeliefObs:
    obs: torch.Tensor
    err: torch.Tensor
    prev_act: torch.Tensor
    ref_preview: torch.Tensor
    priv_state: Optional[torch.Tensor] = None


def _check_2d_dim(name: str, tensor: torch.Tensor, expected_last_dim: int) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be 2D [B, D], got shape={tuple(tensor.shape)}."
        )
    if tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dim mismatch: expected {expected_last_dim}, got {tensor.shape[-1]} "
            f"(shape={tuple(tensor.shape)})."
        )


def _check_batch_size(name: str, tensor: torch.Tensor, expected_batch: int) -> None:
    if tensor.shape[0] != expected_batch:
        raise ValueError(
            f"{name} batch mismatch: expected {expected_batch}, got {tensor.shape[0]} "
            f"(shape={tuple(tensor.shape)})."
        )


def _normalize_sequence_mask(
    sequence_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sequence_mask is None:
        return torch.ones(batch_size, 1, device=device, dtype=dtype)

    mask = sequence_mask.to(device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 2 or mask.shape != (batch_size, 1):
        raise ValueError(
            f"sequence_mask must have shape [B] or [B, 1], got {tuple(sequence_mask.shape)} "
            f"for batch_size={batch_size}."
        )
    return mask.to(dtype=dtype)


def _apply_sequence_mask(hidden: torch.Tensor, sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if sequence_mask is None:
        return hidden
    return hidden * sequence_mask


def episode_starts_to_sequence_mask(episode_starts: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    if episode_starts is None:
        return None
    if episode_starts.ndim == 1:
        starts = episode_starts
    elif episode_starts.ndim == 2 and episode_starts.shape[1] == 1:
        starts = episode_starts.squeeze(-1)
    else:
        raise ValueError(
            f"episode_starts must have shape [B] or [B, 1], got {tuple(episode_starts.shape)}."
        )
    if starts.shape[0] != batch_size:
        raise ValueError(
            f"episode_starts batch mismatch: expected {batch_size}, got {starts.shape[0]}."
        )
    starts = starts.to(dtype=torch.float32)
    return (1.0 - starts).unsqueeze(-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        use_layer_norm: bool = True,
        activate_last: bool = False,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if activate_last:
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUCellBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(x, h)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class SquashedNormal:
    """
    Diagonal Gaussian + tanh squashing.
    log_prob uses tanh correction:
        log pi(a) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2),  a = tanh(u)
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)
        self.eps = eps

    @property
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_action = self.normal.rsample()
        squashed_action = torch.tanh(raw_action)
        return squashed_action, raw_action

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_action = self.normal.sample()
        squashed_action = torch.tanh(raw_action)
        return squashed_action, raw_action

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _log_abs_det_jacobian(raw_action: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(x)^2) in a numerically stable form.
        return 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))

    def log_prob(self, squashed_action: torch.Tensor, raw_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if raw_action is None:
            clipped = squashed_action.clamp(-1.0 + self.eps, 1.0 - self.eps)
            raw_action = self._atanh(clipped)
        gaussian_log_prob = self.normal.log_prob(raw_action).sum(dim=-1, keepdim=True)
        correction = self._log_abs_det_jacobian(raw_action).sum(dim=-1, keepdim=True)
        return gaussian_log_prob - correction

    def entropy(self) -> torch.Tensor:
        # No closed-form entropy after tanh. Use pre-squash entropy as approximation.
        return self.normal.entropy().sum(dim=-1, keepdim=True)


class CurrentObservationEncoder(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        in_dim = cfg.obs_dim + cfg.err_dim + cfg.prev_act_dim
        self.encoder = MLP(
            in_dim=in_dim,
            hidden_dims=(cfg.mlp_width,),
            out_dim=cfg.current_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            activate_last=True,
        )

    def forward(self, obs: torch.Tensor, err: torch.Tensor, prev_act: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([obs, err, prev_act], dim=-1))


class PreviewEncoder(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.encoder = MLP(
            in_dim=cfg.ref_dim,
            hidden_dims=(128,),
            out_dim=cfg.preview_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            activate_last=True,
        )

    def forward(self, ref_preview: torch.Tensor) -> torch.Tensor:
        return self.encoder(ref_preview)


class PrivilegedEncoder(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.encoder = MLP(
            in_dim=cfg.priv_dim,
            hidden_dims=(cfg.mlp_width,),
            out_dim=cfg.priv_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            activate_last=True,
        )

    def forward(self, priv_state: torch.Tensor) -> torch.Tensor:
        return self.encoder(priv_state)


class InstantLatentHead(nn.Module):
    def __init__(self, cfg: NetworkConfig, aux_cfg: AuxPredictionConfig):
        super().__init__()
        self.payload_vel_head = MLP(cfg.fast_gru_hidden_dim, (128,), aux_cfg.pred_payload_vel_dim, cfg.use_layer_norm)
        self.cable_dir_head = MLP(cfg.fast_gru_hidden_dim, (128,), aux_cfg.pred_cable_dir_dim, cfg.use_layer_norm)
        self.cable_dir_rate_head = MLP(cfg.fast_gru_hidden_dim, (128,), aux_cfg.pred_cable_dir_rate_dim, cfg.use_layer_norm)
        self.mode_logit_head = MLP(cfg.fast_gru_hidden_dim, (64,), aux_cfg.pred_mode_logit_dim, cfg.use_layer_norm)
        self.out_dim = (
            aux_cfg.pred_payload_vel_dim
            + aux_cfg.pred_cable_dir_dim
            + aux_cfg.pred_cable_dir_rate_dim
            + aux_cfg.pred_mode_logit_dim
        )

    def forward(self, fast_h: torch.Tensor) -> TensorDict:
        payload_vel = self.payload_vel_head(fast_h)
        cable_dir = F.normalize(self.cable_dir_head(fast_h), dim=-1)
        cable_dir_rate = self.cable_dir_rate_head(fast_h)
        mode_logits = self.mode_logit_head(fast_h)
        concat_feat = torch.cat([payload_vel, cable_dir, cable_dir_rate, mode_logits], dim=-1)
        return {
            "payload_vel": payload_vel,
            "cable_dir": cable_dir,
            "cable_dir_rate": cable_dir_rate,
            "mode_logits": mode_logits,
            "concat_feat": concat_feat,
        }


class ContextLatentHead(nn.Module):
    def __init__(self, cfg: NetworkConfig, aux_cfg: AuxPredictionConfig):
        super().__init__()
        self.mass_head = MLP(cfg.slow_gru_hidden_dim, (64,), aux_cfg.pred_mass_dim, cfg.use_layer_norm)
        self.length_head = MLP(cfg.slow_gru_hidden_dim, (64,), aux_cfg.pred_length_dim, cfg.use_layer_norm)
        self.disturbance_head = MLP(cfg.slow_gru_hidden_dim, (64,), aux_cfg.pred_disturbance_ctx_dim, cfg.use_layer_norm)
        self.actuator_ctx_head = MLP(cfg.slow_gru_hidden_dim, (64,), aux_cfg.pred_actuator_ctx_dim, cfg.use_layer_norm)
        self.out_dim = (
            aux_cfg.pred_mass_dim
            + aux_cfg.pred_length_dim
            + aux_cfg.pred_disturbance_ctx_dim
            + aux_cfg.pred_actuator_ctx_dim
        )

    def forward(self, slow_h: torch.Tensor) -> TensorDict:
        mass = F.softplus(self.mass_head(slow_h))
        length = F.softplus(self.length_head(slow_h))
        disturbance_ctx = self.disturbance_head(slow_h)
        actuator_ctx = self.actuator_ctx_head(slow_h)
        concat_feat = torch.cat([mass, length, disturbance_ctx, actuator_ctx], dim=-1)
        return {
            "mass": mass,
            "length": length,
            "disturbance_ctx": disturbance_ctx,
            "actuator_ctx": actuator_ctx,
            "concat_feat": concat_feat,
        }


class BeliefActor(nn.Module):
    def __init__(self, cfg: NetworkConfig, aux_cfg: AuxPredictionConfig):
        super().__init__()
        self.cfg = cfg
        self.current_encoder = CurrentObservationEncoder(cfg)
        self.preview_encoder = PreviewEncoder(cfg)

        fast_input_dim = cfg.current_hidden_dim + cfg.prev_act_dim
        self.fast_gru = GRUCellBlock(fast_input_dim, cfg.fast_gru_hidden_dim)
        self.slow_gru = GRUCellBlock(fast_input_dim, cfg.slow_gru_hidden_dim)

        self.inst_head = InstantLatentHead(cfg, aux_cfg)
        self.ctx_head = ContextLatentHead(cfg, aux_cfg)

        fusion_in_dim = (
            cfg.current_hidden_dim
            + cfg.preview_hidden_dim
            + cfg.fast_gru_hidden_dim
            + cfg.slow_gru_hidden_dim
            + self.inst_head.out_dim
            + self.ctx_head.out_dim
        )
        self.fusion = MLP(
            in_dim=fusion_in_dim,
            hidden_dims=(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
            out_dim=cfg.actor_feature_dim,
            use_layer_norm=cfg.use_layer_norm,
            activate_last=True,
        )
        self.mean_head = MLP(cfg.actor_feature_dim, (64,), cfg.act_dim, cfg.use_layer_norm)

        if cfg.use_state_dependent_std:
            self.log_std_head = MLP(cfg.actor_feature_dim, (64,), cfg.act_dim, cfg.use_layer_norm)
            self.log_std_param = None
        else:
            self.log_std_head = None
            self.log_std_param = nn.Parameter(torch.ones(cfg.act_dim) * cfg.init_log_std)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fast_gru.init_hidden(batch_size, device), self.slow_gru.init_hidden(batch_size, device)

    def forward_step(
        self,
        obs: torch.Tensor,
        err: torch.Tensor,
        prev_act: torch.Tensor,
        ref_preview: torch.Tensor,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> TensorDict:
        _check_2d_dim("actor.obs", obs, self.cfg.obs_dim)
        _check_2d_dim("actor.err", err, self.cfg.err_dim)
        _check_2d_dim("actor.prev_act", prev_act, self.cfg.prev_act_dim)
        _check_2d_dim("actor.ref_preview", ref_preview, self.cfg.ref_dim)
        _check_2d_dim("actor.fast_h", fast_h, self.cfg.fast_gru_hidden_dim)
        _check_2d_dim("actor.slow_h", slow_h, self.cfg.slow_gru_hidden_dim)

        batch_size = obs.shape[0]
        _check_batch_size("actor.err", err, batch_size)
        _check_batch_size("actor.prev_act", prev_act, batch_size)
        _check_batch_size("actor.ref_preview", ref_preview, batch_size)
        _check_batch_size("actor.fast_h", fast_h, batch_size)
        _check_batch_size("actor.slow_h", slow_h, batch_size)
        mask = _normalize_sequence_mask(sequence_mask, batch_size, obs.device, obs.dtype)
        masked_fast_h = _apply_sequence_mask(fast_h, mask)
        masked_slow_h = _apply_sequence_mask(slow_h, mask)

        cur_feat = self.current_encoder(obs, err, prev_act)
        ref_feat = self.preview_encoder(ref_preview)
        gru_input = torch.cat([cur_feat, prev_act], dim=-1)

        new_fast_h = self.fast_gru(gru_input, masked_fast_h)
        new_slow_h = self.slow_gru(gru_input, masked_slow_h)

        inst_pred = self.inst_head(new_fast_h)
        ctx_pred = self.ctx_head(new_slow_h)

        fusion_input = torch.cat(
            [
                cur_feat,
                ref_feat,
                new_fast_h,
                new_slow_h,
                inst_pred["concat_feat"],
                ctx_pred["concat_feat"],
            ],
            dim=-1,
        )
        actor_feat = self.fusion(fusion_input)
        mean = self.mean_head(actor_feat)

        if self.cfg.use_state_dependent_std:
            log_std = self.log_std_head(actor_feat)
            log_std = torch.clamp(log_std, self.cfg.min_log_std, self.cfg.max_log_std)
        else:
            if self.log_std_param is None:
                raise RuntimeError("log_std_param must exist when use_state_dependent_std=False")
            log_std = self.log_std_param.unsqueeze(0).expand_as(mean)

        std = torch.exp(log_std)
        dist = SquashedNormal(mean=mean, std=std)

        if deterministic:
            action = dist.mode
            raw_action = mean
        else:
            action, raw_action = dist.rsample()
        log_prob = dist.log_prob(action, raw_action=raw_action)

        return {
            "action": action,
            "raw_action": raw_action,
            "log_prob": log_prob,
            "mean": mean,
            "log_std": log_std,
            "dist": dist,
            "actor_feat": actor_feat,
            "cur_feat": cur_feat,
            "ref_feat": ref_feat,
            "fast_h": new_fast_h,
            "slow_h": new_slow_h,
            "inst_pred": inst_pred,
            "ctx_pred": ctx_pred,
        }


class AsymmetricBeliefCritic(nn.Module):
    def __init__(self, cfg: NetworkConfig, aux_cfg: AuxPredictionConfig):
        super().__init__()
        self.cfg = cfg
        self.current_encoder = CurrentObservationEncoder(cfg)
        self.preview_encoder = PreviewEncoder(cfg)

        fast_input_dim = cfg.current_hidden_dim + cfg.prev_act_dim
        self.fast_gru = GRUCellBlock(fast_input_dim, cfg.fast_gru_hidden_dim)
        self.slow_gru = GRUCellBlock(fast_input_dim, cfg.slow_gru_hidden_dim)

        self.inst_head = InstantLatentHead(cfg, aux_cfg)
        self.ctx_head = ContextLatentHead(cfg, aux_cfg)

        belief_fusion_in_dim = (
            cfg.current_hidden_dim
            + cfg.preview_hidden_dim
            + cfg.fast_gru_hidden_dim
            + cfg.slow_gru_hidden_dim
            + self.inst_head.out_dim
            + self.ctx_head.out_dim
        )
        self.belief_fusion = MLP(
            in_dim=belief_fusion_in_dim,
            hidden_dims=(cfg.fusion_hidden_dim,),
            out_dim=cfg.critic_feature_dim,
            use_layer_norm=cfg.use_layer_norm,
            activate_last=True,
        )

        self.priv_encoder = PrivilegedEncoder(cfg)
        value_in_dim = cfg.critic_feature_dim + cfg.priv_hidden_dim
        self.value_head = MLP(
            in_dim=value_in_dim,
            hidden_dims=(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
            out_dim=1,
            use_layer_norm=cfg.use_layer_norm,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fast_gru.init_hidden(batch_size, device), self.slow_gru.init_hidden(batch_size, device)

    def forward_step(
        self,
        obs: torch.Tensor,
        err: torch.Tensor,
        prev_act: torch.Tensor,
        ref_preview: torch.Tensor,
        priv_state: torch.Tensor,
        fast_h: torch.Tensor,
        slow_h: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        _check_2d_dim("critic.obs", obs, self.cfg.obs_dim)
        _check_2d_dim("critic.err", err, self.cfg.err_dim)
        _check_2d_dim("critic.prev_act", prev_act, self.cfg.prev_act_dim)
        _check_2d_dim("critic.ref_preview", ref_preview, self.cfg.ref_dim)
        _check_2d_dim("critic.priv_state", priv_state, self.cfg.priv_dim)
        _check_2d_dim("critic.fast_h", fast_h, self.cfg.fast_gru_hidden_dim)
        _check_2d_dim("critic.slow_h", slow_h, self.cfg.slow_gru_hidden_dim)

        batch_size = obs.shape[0]
        _check_batch_size("critic.err", err, batch_size)
        _check_batch_size("critic.prev_act", prev_act, batch_size)
        _check_batch_size("critic.ref_preview", ref_preview, batch_size)
        _check_batch_size("critic.priv_state", priv_state, batch_size)
        _check_batch_size("critic.fast_h", fast_h, batch_size)
        _check_batch_size("critic.slow_h", slow_h, batch_size)
        mask = _normalize_sequence_mask(sequence_mask, batch_size, obs.device, obs.dtype)
        masked_fast_h = _apply_sequence_mask(fast_h, mask)
        masked_slow_h = _apply_sequence_mask(slow_h, mask)

        cur_feat = self.current_encoder(obs, err, prev_act)
        ref_feat = self.preview_encoder(ref_preview)

        gru_input = torch.cat([cur_feat, prev_act], dim=-1)
        new_fast_h = self.fast_gru(gru_input, masked_fast_h)
        new_slow_h = self.slow_gru(gru_input, masked_slow_h)

        inst_pred = self.inst_head(new_fast_h)
        ctx_pred = self.ctx_head(new_slow_h)

        belief_fusion_input = torch.cat(
            [
                cur_feat,
                ref_feat,
                new_fast_h,
                new_slow_h,
                inst_pred["concat_feat"],
                ctx_pred["concat_feat"],
            ],
            dim=-1,
        )
        belief_feat = self.belief_fusion(belief_fusion_input)
        priv_feat = self.priv_encoder(priv_state)
        value = self.value_head(torch.cat([belief_feat, priv_feat], dim=-1))
        return {
            "value": value,
            "belief_feat": belief_feat,
            "priv_feat": priv_feat,
            "fast_h": new_fast_h,
            "slow_h": new_slow_h,
            "inst_pred": inst_pred,
            "ctx_pred": ctx_pred,
        }


class BeliefActorCritic(nn.Module):
    def __init__(self, cfg: NetworkConfig, aux_cfg: AuxPredictionConfig):
        super().__init__()
        self.actor = BeliefActor(cfg, aux_cfg)
        self.critic = AsymmetricBeliefCritic(cfg, aux_cfg)
        self.cfg = cfg

    def init_belief_state(self, batch_size: int, device: torch.device) -> RunningBeliefState:
        actor_fast_h, actor_slow_h = self.actor.init_hidden(batch_size, device)
        critic_fast_h, critic_slow_h = self.critic.init_hidden(batch_size, device)
        return RunningBeliefState(
            actor_fast_h=actor_fast_h,
            actor_slow_h=actor_slow_h,
            critic_fast_h=critic_fast_h,
            critic_slow_h=critic_slow_h,
        )

    def actor_forward(
        self,
        obs: torch.Tensor,
        err: torch.Tensor,
        prev_act: torch.Tensor,
        ref_preview: torch.Tensor,
        belief_state: RunningBeliefState,
        sequence_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> TensorDict:
        out = self.actor.forward_step(
            obs=obs,
            err=err,
            prev_act=prev_act,
            ref_preview=ref_preview,
            fast_h=belief_state.actor_fast_h,
            slow_h=belief_state.actor_slow_h,
            sequence_mask=sequence_mask,
            deterministic=deterministic,
        )
        belief_state.actor_fast_h = out["fast_h"]
        belief_state.actor_slow_h = out["slow_h"]
        return out

    def critic_forward(
        self,
        obs: torch.Tensor,
        err: torch.Tensor,
        prev_act: torch.Tensor,
        ref_preview: torch.Tensor,
        priv_state: torch.Tensor,
        belief_state: RunningBeliefState,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        out = self.critic.forward_step(
            obs=obs,
            err=err,
            prev_act=prev_act,
            ref_preview=ref_preview,
            priv_state=priv_state,
            fast_h=belief_state.critic_fast_h,
            slow_h=belief_state.critic_slow_h,
            sequence_mask=sequence_mask,
        )
        belief_state.critic_fast_h = out["fast_h"]
        belief_state.critic_slow_h = out["slow_h"]
        return out


class BeliefActorCriticPolicy(nn.Module):
    """
    SB3-like recurrent policy interface:
      - forward(obs, state, episode_starts, deterministic)
      - evaluate_actions(obs, actions, state, episode_starts)
      - predict_values(obs, state, episode_starts)
      - get_distribution(obs, state, episode_starts)
    """

    def __init__(
        self,
        observation_space: Optional[spaces.Space],
        action_space: Optional[spaces.Box],
        cfg: NetworkConfig,
        aux_cfg: Optional[AuxPredictionConfig] = None,
        require_privileged_for_value: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.aux_cfg = aux_cfg if aux_cfg is not None else AuxPredictionConfig()
        self.model = BeliefActorCritic(cfg, self.aux_cfg)
        self.observation_space = observation_space
        self.action_space = action_space
        self.require_privileged_for_value = require_privileged_for_value

        if action_space is not None:
            if not isinstance(action_space, spaces.Box):
                raise TypeError("BeliefActorCriticPolicy only supports continuous Box action_space.")
            if action_space.shape is None or len(action_space.shape) != 1:
                raise ValueError(f"action_space must be 1D Box, got shape={action_space.shape}.")
            if int(action_space.shape[0]) != cfg.act_dim:
                raise ValueError(
                    f"action_space dim mismatch: cfg.act_dim={cfg.act_dim}, action_space.shape={action_space.shape}."
                )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> RunningBeliefState:
        dev = self.device if device is None else device
        return self.model.init_belief_state(batch_size=batch_size, device=dev)

    def _split_flat_obs(self, obs_flat: torch.Tensor) -> ParsedBeliefObs:
        total_without_priv = self.cfg.obs_dim + self.cfg.err_dim + self.cfg.prev_act_dim + self.cfg.ref_dim
        total_with_priv = total_without_priv + self.cfg.priv_dim

        _check_2d_dim("obs_flat", obs_flat, obs_flat.shape[-1])
        last_dim = obs_flat.shape[-1]
        if last_dim not in (total_without_priv, total_with_priv):
            raise ValueError(
                f"Flat obs dim mismatch: expected {total_without_priv} (without priv) or "
                f"{total_with_priv} (with priv), got {last_dim}."
            )

        idx = 0
        obs = obs_flat[:, idx : idx + self.cfg.obs_dim]
        idx += self.cfg.obs_dim
        err = obs_flat[:, idx : idx + self.cfg.err_dim]
        idx += self.cfg.err_dim
        prev_act = obs_flat[:, idx : idx + self.cfg.prev_act_dim]
        idx += self.cfg.prev_act_dim
        ref_preview = obs_flat[:, idx : idx + self.cfg.ref_dim]
        idx += self.cfg.ref_dim
        priv_state = obs_flat[:, idx : idx + self.cfg.priv_dim] if last_dim == total_with_priv else None
        return ParsedBeliefObs(obs=obs, err=err, prev_act=prev_act, ref_preview=ref_preview, priv_state=priv_state)

    def _parse_obs(self, obs: Union[torch.Tensor, TensorDict]) -> ParsedBeliefObs:
        if isinstance(obs, torch.Tensor):
            return self._split_flat_obs(obs)
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be Tensor or Dict[str, Tensor], got {type(obs)}.")

        required_keys = ("obs", "prev_act", "ref_preview")
        if self.cfg.err_dim > 0:
            required_keys = ("obs", "err", "prev_act", "ref_preview")
        for key in required_keys:
            if key not in obs:
                raise KeyError(f"Dict obs missing key '{key}'. Required keys: {required_keys}.")

        obs_tensor = obs["obs"]
        _check_2d_dim("obs['obs']", obs_tensor, self.cfg.obs_dim)
        batch_size = obs_tensor.shape[0]
        if "err" in obs:
            err_tensor = obs["err"]
        else:
            if self.cfg.err_dim > 0:
                raise KeyError("Dict obs missing key 'err' while cfg.err_dim > 0.")
            err_tensor = torch.zeros(
                batch_size,
                0,
                device=obs_tensor.device,
                dtype=obs_tensor.dtype,
            )

        parsed = ParsedBeliefObs(
            obs=obs_tensor,
            err=err_tensor,
            prev_act=obs["prev_act"],
            ref_preview=obs["ref_preview"],
            priv_state=obs.get("priv_state", obs.get("priv")),
        )
        _check_2d_dim("obs['err']", parsed.err, self.cfg.err_dim)
        _check_2d_dim("obs['prev_act']", parsed.prev_act, self.cfg.prev_act_dim)
        _check_2d_dim("obs['ref_preview']", parsed.ref_preview, self.cfg.ref_dim)
        _check_batch_size("obs['err']", parsed.err, batch_size)
        _check_batch_size("obs['prev_act']", parsed.prev_act, batch_size)
        _check_batch_size("obs['ref_preview']", parsed.ref_preview, batch_size)
        if parsed.priv_state is not None:
            _check_2d_dim("obs['priv_state']", parsed.priv_state, self.cfg.priv_dim)
            _check_batch_size("obs['priv_state']", parsed.priv_state, batch_size)
        return parsed

    def _ensure_priv_state(self, parsed_obs: ParsedBeliefObs) -> torch.Tensor:
        if parsed_obs.priv_state is not None:
            return parsed_obs.priv_state
        if self.require_privileged_for_value:
            raise ValueError(
                "Privileged state is required for critic/value but was not provided. "
                "Provide dict obs with 'priv_state' (or flat obs containing priv segment)."
            )
        return torch.zeros(parsed_obs.obs.shape[0], self.cfg.priv_dim, device=parsed_obs.obs.device, dtype=parsed_obs.obs.dtype)

    def _to_env_action(self, squashed_action: torch.Tensor) -> torch.Tensor:
        if self.action_space is None:
            return squashed_action
        low = torch.as_tensor(self.action_space.low, device=squashed_action.device, dtype=squashed_action.dtype)
        high = torch.as_tensor(self.action_space.high, device=squashed_action.device, dtype=squashed_action.dtype)
        return low + 0.5 * (squashed_action + 1.0) * (high - low)

    def _from_env_action(self, env_action: torch.Tensor) -> torch.Tensor:
        if self.action_space is None:
            return env_action.clamp(-1.0, 1.0)
        low = torch.as_tensor(self.action_space.low, device=env_action.device, dtype=env_action.dtype)
        high = torch.as_tensor(self.action_space.high, device=env_action.device, dtype=env_action.dtype)
        scaled = 2.0 * (env_action - low) / (high - low + 1e-8) - 1.0
        return scaled.clamp(-1.0, 1.0)

    def act(
        self,
        obs: Union[torch.Tensor, TensorDict],
        state: Optional[RunningBeliefState] = None,
        episode_starts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, RunningBeliefState]:
        parsed = self._parse_obs(obs)
        batch_size = parsed.obs.shape[0]
        if state is None:
            state = self.init_states(batch_size=batch_size, device=parsed.obs.device)
        working_state = state.clone()
        seq_mask = episode_starts_to_sequence_mask(episode_starts, batch_size)

        actor_out = self.model.actor_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            belief_state=working_state,
            sequence_mask=seq_mask,
            deterministic=deterministic,
        )
        env_action = self._to_env_action(actor_out["action"])
        return env_action, actor_out["log_prob"].squeeze(-1), working_state

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        state: Optional[RunningBeliefState] = None,
        episode_starts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RunningBeliefState]:
        parsed = self._parse_obs(obs)
        batch_size = parsed.obs.shape[0]
        if state is None:
            state = self.init_states(batch_size=batch_size, device=parsed.obs.device)
        working_state = state.clone()
        seq_mask = episode_starts_to_sequence_mask(episode_starts, batch_size)

        actor_out = self.model.actor_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            belief_state=working_state,
            sequence_mask=seq_mask,
            deterministic=deterministic,
        )
        priv_state = self._ensure_priv_state(parsed)
        critic_out = self.model.critic_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            priv_state=priv_state,
            belief_state=working_state,
            sequence_mask=seq_mask,
        )
        env_action = self._to_env_action(actor_out["action"])
        return env_action, critic_out["value"], actor_out["log_prob"].squeeze(-1), working_state

    def get_distribution(
        self,
        obs: Union[torch.Tensor, TensorDict],
        state: Optional[RunningBeliefState] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> Tuple[SquashedNormal, RunningBeliefState]:
        parsed = self._parse_obs(obs)
        batch_size = parsed.obs.shape[0]
        if state is None:
            state = self.init_states(batch_size=batch_size, device=parsed.obs.device)
        working_state = state.clone()
        seq_mask = episode_starts_to_sequence_mask(episode_starts, batch_size)
        actor_out = self.model.actor_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            belief_state=working_state,
            sequence_mask=seq_mask,
            deterministic=False,
        )
        return actor_out["dist"], working_state

    def predict_values(
        self,
        obs: Union[torch.Tensor, TensorDict],
        state: Optional[RunningBeliefState] = None,
        episode_starts: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, RunningBeliefState]]:
        parsed = self._parse_obs(obs)
        batch_size = parsed.obs.shape[0]
        if state is None:
            state = self.init_states(batch_size=batch_size, device=parsed.obs.device)
        working_state = state.clone()
        seq_mask = episode_starts_to_sequence_mask(episode_starts, batch_size)
        priv_state = self._ensure_priv_state(parsed)

        critic_out = self.model.critic_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            priv_state=priv_state,
            belief_state=working_state,
            sequence_mask=seq_mask,
        )
        if return_state:
            return critic_out["value"], working_state
        return critic_out["value"]

    def evaluate_actions(
        self,
        obs: Union[torch.Tensor, TensorDict],
        actions: torch.Tensor,
        state: Optional[RunningBeliefState] = None,
        episode_starts: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RunningBeliefState]]:
        parsed = self._parse_obs(obs)
        batch_size = parsed.obs.shape[0]
        if actions.ndim != 2 or actions.shape[0] != batch_size or actions.shape[1] != self.cfg.act_dim:
            raise ValueError(
                f"actions must have shape [B, {self.cfg.act_dim}], got {tuple(actions.shape)}."
            )
        if state is None:
            state = self.init_states(batch_size=batch_size, device=parsed.obs.device)
        working_state = state.clone()
        seq_mask = episode_starts_to_sequence_mask(episode_starts, batch_size)

        actor_out = self.model.actor_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            belief_state=working_state,
            sequence_mask=seq_mask,
            deterministic=False,
        )
        squashed_actions = self._from_env_action(actions)
        log_prob = actor_out["dist"].log_prob(squashed_actions).squeeze(-1)
        entropy = actor_out["dist"].entropy().squeeze(-1)

        priv_state = self._ensure_priv_state(parsed)
        critic_out = self.model.critic_forward(
            obs=parsed.obs,
            err=parsed.err,
            prev_act=parsed.prev_act,
            ref_preview=parsed.ref_preview,
            priv_state=priv_state,
            belief_state=working_state,
            sequence_mask=seq_mask,
        )
        outputs = (critic_out["value"], log_prob, entropy)
        if return_state:
            return outputs + (working_state,)
        return outputs

    def rollout_step(
        self,
        obs: Union[torch.Tensor, TensorDict],
        state: Optional[RunningBeliefState],
        episode_starts: Optional[torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        actions, values, log_prob, new_state = self.forward(
            obs=obs,
            state=state,
            episode_starts=episode_starts,
            deterministic=deterministic,
        )
        return {
            "actions": actions,
            "values": values,
            "log_prob": log_prob,
            "state_actor_fast_h": new_state.actor_fast_h,
            "state_actor_slow_h": new_state.actor_slow_h,
            "state_critic_fast_h": new_state.critic_fast_h,
            "state_critic_slow_h": new_state.critic_slow_h,
        }


def compute_auxiliary_losses(
    actor_out: TensorDict,
    critic_out: Optional[TensorDict],
    batch: TensorDict,
    loss_weights: Dict[str, float],
) -> TensorDict:
    del critic_out
    losses: TensorDict = {}
    total_aux: torch.Tensor = torch.zeros((), device=actor_out["action"].device)

    inst_pred = actor_out["inst_pred"]
    ctx_pred = actor_out["ctx_pred"]

    if "payload_vel_target" in batch:
        l = F.mse_loss(inst_pred["payload_vel"], batch["payload_vel_target"])
        losses["aux_payload_vel"] = l
        total_aux = total_aux + loss_weights.get("payload_vel", 1.0) * l

    if "cable_dir_target" in batch:
        pred = F.normalize(inst_pred["cable_dir"], dim=-1)
        target = F.normalize(batch["cable_dir_target"], dim=-1)
        l = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        losses["aux_cable_dir"] = l
        total_aux = total_aux + loss_weights.get("cable_dir", 1.0) * l

    if "cable_dir_rate_target" in batch:
        l = F.mse_loss(inst_pred["cable_dir_rate"], batch["cable_dir_rate_target"])
        losses["aux_cable_dir_rate"] = l
        total_aux = total_aux + loss_weights.get("cable_dir_rate", 1.0) * l

    if "mode_target" in batch:
        l = F.cross_entropy(inst_pred["mode_logits"], batch["mode_target"].long())
        losses["aux_mode"] = l
        total_aux = total_aux + loss_weights.get("mode", 1.0) * l

    if "mass_target" in batch:
        l = F.mse_loss(ctx_pred["mass"], batch["mass_target"])
        losses["aux_mass"] = l
        total_aux = total_aux + loss_weights.get("mass", 1.0) * l

    if "length_target" in batch:
        l = F.mse_loss(ctx_pred["length"], batch["length_target"])
        losses["aux_length"] = l
        total_aux = total_aux + loss_weights.get("length", 1.0) * l

    if "disturbance_ctx_target" in batch:
        l = F.mse_loss(ctx_pred["disturbance_ctx"], batch["disturbance_ctx_target"])
        losses["aux_disturbance_ctx"] = l
        total_aux = total_aux + loss_weights.get("disturbance_ctx", 1.0) * l

    if "actuator_ctx_target" in batch:
        l = F.mse_loss(ctx_pred["actuator_ctx"], batch["actuator_ctx_target"])
        losses["aux_actuator_ctx"] = l
        total_aux = total_aux + loss_weights.get("actuator_ctx", 1.0) * l

    if "prev_ctx_pred" in batch:
        current_ctx = torch.cat(
            [ctx_pred["mass"], ctx_pred["length"], ctx_pred["disturbance_ctx"], ctx_pred["actuator_ctx"]],
            dim=-1,
        )
        l = F.mse_loss(current_ctx, batch["prev_ctx_pred"])
        losses["aux_ctx_smooth"] = l
        total_aux = total_aux + loss_weights.get("ctx_smooth", 0.1) * l

    losses["aux_total"] = total_aux
    return losses


def compute_ppo_losses(
    actor_out: TensorDict,
    critic_out: TensorDict,
    batch: TensorDict,
    aux_losses: Optional[TensorDict] = None,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    aux_coef: float = 1.0,
) -> TensorDict:
    new_log_prob = actor_out["log_prob"]
    old_log_prob = batch["old_log_prob"]
    adv = batch["advantages"]
    ret = batch["returns"]

    ratio = torch.exp(new_log_prob - old_log_prob)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value = critic_out["value"]
    value_loss = F.mse_loss(value, ret)

    entropy = actor_out["dist"].entropy().mean()
    entropy_loss = -entropy
    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    if aux_losses is not None and "aux_total" in aux_losses:
        total_loss = total_loss + aux_coef * aux_losses["aux_total"]

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "entropy": entropy.detach(),
    }


def example_build_policy() -> BeliefActorCriticPolicy:
    cfg = NetworkConfig(
        obs_dim=32,
        err_dim=12,
        prev_act_dim=4,
        ref_dim=48,
        priv_dim=40,
        act_dim=4,
    )
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(cfg.obs_dim + cfg.err_dim + cfg.prev_act_dim + cfg.ref_dim + cfg.priv_dim,), dtype=float)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(cfg.act_dim,), dtype=float)
    return BeliefActorCriticPolicy(observation_space=obs_space, action_space=act_space, cfg=cfg)


def example_rollout_step() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = example_build_policy().to(device)

    batch_size = 8
    state = policy.init_states(batch_size=batch_size, device=device)
    cfg = policy.cfg

    obs = torch.randn(batch_size, cfg.obs_dim, device=device)
    err = torch.randn(batch_size, cfg.err_dim, device=device)
    prev_act = torch.randn(batch_size, cfg.prev_act_dim, device=device)
    ref_preview = torch.randn(batch_size, cfg.ref_dim, device=device)
    priv_state = torch.randn(batch_size, cfg.priv_dim, device=device)
    episode_starts = torch.zeros(batch_size, device=device)
    episode_starts[0] = 1.0

    obs_dict = {
        "obs": obs,
        "err": err,
        "prev_act": prev_act,
        "ref_preview": ref_preview,
        "priv_state": priv_state,
    }
    actions, values, log_prob, new_state = policy.forward(
        obs=obs_dict,
        state=state,
        episode_starts=episode_starts,
        deterministic=False,
    )
    print("action shape:", tuple(actions.shape))
    print("value shape:", tuple(values.shape))
    print("log_prob shape:", tuple(log_prob.shape))
    print("actor fast state shape:", tuple(new_state.actor_fast_h.shape))


if __name__ == "__main__":
    example_rollout_step()
