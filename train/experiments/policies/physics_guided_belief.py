from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from train.experiments.feature_extractors.belief_actor_critic import (
    AuxPredictionConfig,
    BeliefActorCriticPolicy as BeliefCorePolicy,
    NetworkConfig,
    SquashedNormal,
)
from train.experiments.specs import PolicySpec


class _SB3SquashedDistributionAdapter:
    """
    Minimal distribution adapter so external callers can still use:
      - get_actions()
      - log_prob(actions)
      - entropy()
    """

    def __init__(self, core_policy: BeliefCorePolicy, dist: SquashedNormal):
        self.core_policy = core_policy
        self.dist = dist

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            squashed = self.dist.mode
        else:
            squashed, _ = self.dist.rsample()
        return self.core_policy._to_env_action(squashed)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        squashed = self.core_policy._from_env_action(actions)
        return self.dist.log_prob(squashed).squeeze(-1)

    def entropy(self) -> th.Tensor:
        return self.dist.entropy().squeeze(-1)


class PhysicsGuidedBeliefPolicy(ActorCriticPolicy):
    """
    SB3 ActorCriticPolicy adapter backed by BeliefActorCriticPolicy.

    Observation is expected to be a flat Box that concatenates:
      [obs, err(optional 0-dim), prev_act, ref_preview, priv_state]
    """

    def __init__(
        self,
        *args,
        belief_cfg: Optional[dict[str, Any] | NetworkConfig] = None,
        aux_cfg: Optional[dict[str, Any] | AuxPredictionConfig] = None,
        require_privileged_for_value: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))
        kwargs.setdefault("activation_fn", th.nn.SiLU)
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

        self.require_privileged_for_value = require_privileged_for_value
        self._belief_cfg_dict = self._coerce_network_config_dict(belief_cfg)
        self._aux_cfg_dict = self._coerce_aux_config_dict(aux_cfg)
        self.belief_cfg = NetworkConfig(**self._belief_cfg_dict)
        self.aux_cfg = AuxPredictionConfig(**self._aux_cfg_dict)
        self._validate_observation_layout()

        self.belief_core = BeliefCorePolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            cfg=self.belief_cfg,
            aux_cfg=self.aux_cfg,
            require_privileged_for_value=require_privileged_for_value,
        )

        # Rebuild optimizer to include belief_core parameters.
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=current_lr,
            **self.optimizer_kwargs,
        )

    def _coerce_network_config_dict(self, belief_cfg: Optional[dict[str, Any] | NetworkConfig]) -> Dict[str, Any]:
        if belief_cfg is None:
            raise ValueError(
                "belief_cfg is required. Provide policy_kwargs['belief_cfg'] with "
                "obs/err/prev_act/ref/priv/action dims for BeliefActorCritic."
            )

        if isinstance(belief_cfg, NetworkConfig):
            cfg_dict = dict(vars(belief_cfg))
        elif isinstance(belief_cfg, dict):
            cfg_dict = dict(belief_cfg)
        else:
            raise TypeError(f"belief_cfg must be dict or NetworkConfig, got {type(belief_cfg)}.")

        action_dim = int(np.prod(self.action_space.shape))
        cfg_dict.setdefault("act_dim", action_dim)
        if int(cfg_dict["act_dim"]) != action_dim:
            raise ValueError(
                f"belief_cfg act_dim mismatch: expected {action_dim} from action_space, "
                f"got {cfg_dict['act_dim']}."
            )
        return cfg_dict

    @staticmethod
    def _coerce_aux_config_dict(aux_cfg: Optional[dict[str, Any] | AuxPredictionConfig]) -> Dict[str, Any]:
        if aux_cfg is None:
            return dict(vars(AuxPredictionConfig()))
        if isinstance(aux_cfg, AuxPredictionConfig):
            return dict(vars(aux_cfg))
        if isinstance(aux_cfg, dict):
            return dict(aux_cfg)
        raise TypeError(f"aux_cfg must be dict or AuxPredictionConfig, got {type(aux_cfg)}.")

    def _validate_observation_layout(self) -> None:
        if self.observation_space.shape is None or len(self.observation_space.shape) != 1:
            raise ValueError(
                f"PhysicsGuidedBeliefPolicy requires 1D Box observation, got shape={self.observation_space.shape}."
            )
        obs_dim = int(self.observation_space.shape[0])
        expected_without_priv = (
            self.belief_cfg.obs_dim
            + self.belief_cfg.err_dim
            + self.belief_cfg.prev_act_dim
            + self.belief_cfg.ref_dim
        )
        expected_with_priv = expected_without_priv + self.belief_cfg.priv_dim
        if obs_dim not in (expected_without_priv, expected_with_priv):
            raise ValueError(
                f"Observation dim mismatch for belief policy: got {obs_dim}, expected "
                f"{expected_without_priv} (without priv) or {expected_with_priv} (with priv)."
            )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                belief_cfg=self._belief_cfg_dict,
                aux_cfg=self._aux_cfg_dict,
                require_privileged_for_value=self.require_privileged_for_value,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        actions, values, log_prob, _ = self.belief_core.forward(
            obs=obs,
            state=None,
            episode_starts=None,
            deterministic=deterministic,
        )
        return actions, values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _, _ = self.belief_core.act(
            obs=observation,
            state=None,
            episode_starts=None,
            deterministic=deterministic,
        )
        return actions

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self.belief_core.predict_values(
            obs=obs,
            state=None,
            episode_starts=None,
            return_state=False,
        )

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        values, log_prob, entropy = self.belief_core.evaluate_actions(
            obs=obs,
            actions=actions,
            state=None,
            episode_starts=None,
            return_state=False,
        )
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        dist, _ = self.belief_core.get_distribution(
            obs=obs,
            state=None,
            episode_starts=None,
        )
        return _SB3SquashedDistributionAdapter(self.belief_core, dist)


POLICY_SPEC = PolicySpec(
    name="physics-guided_belief",
    policy=PhysicsGuidedBeliefPolicy,
    description=(
        "Physics-guided belief actor-critic (fast/slow GRU + asymmetric privileged critic) "
        "with tanh-corrected log-prob."
    ),
    policy_kwargs={
        "belief_cfg": {
            "obs_dim": 21,
            "err_dim": 0,
            "prev_act_dim": 3,
            "ref_dim": 30,
            "priv_dim": 14,
            "act_dim": 3,
            "current_hidden_dim": 128,
            "preview_hidden_dim": 64,
            "fast_gru_hidden_dim": 128,
            "slow_gru_hidden_dim": 64,
            "fusion_hidden_dim": 256,
            "actor_feature_dim": 128,
            "critic_feature_dim": 128,
            "priv_hidden_dim": 128,
            "mlp_width": 256,
            "use_layer_norm": True,
            "use_state_dependent_std": False,
            "init_log_std": -0.7,
            "min_log_std": -5.0,
            "max_log_std": 1.0,
        },
        "require_privileged_for_value": True,
        "ortho_init": False,
    },
    algo_kwargs={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "ent_coef": 0.001,
    },
    env_kwargs={
        "observation_mode": "belief",
        "belief_ref_horizon": 5,
    },
)
