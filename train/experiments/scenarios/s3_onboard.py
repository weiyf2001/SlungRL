from collections import deque

import numpy as np
from gymnasium.spaces import Box

from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from envs.utils.rotation_transformations import euler2rot, quat2euler_raw
from train.experiments.specs import ScenarioSpec


class QuadrotorPayloadEnvS3Onboard(QuadrotorPayloadEnvV2):
    """
    S3 onboard scenario:
    - payload mass and cable length are randomized every episode
    - payload position and payload position error are removed from the observation
    - rope direction vector is added to the state as onboard-observable information
    - optional belief observation mode packs:
      [obs, prev_act, ref_preview, privileged_state]
    - no sensor noise, delay, filtering, actuator lag, or disturbances
    """

    obs_curr_dim = 24

    def __init__(self, *args, observation_mode: str = "history", belief_ref_horizon: int = 5, **kwargs):
        if observation_mode not in {"current", "history", "belief"}:
            raise ValueError(f"Unsupported observation_mode '{observation_mode}'.")
        if belief_ref_horizon <= 0:
            raise ValueError(f"belief_ref_horizon must be > 0, got {belief_ref_horizon}.")

        self.observation_mode = observation_mode
        self.belief_ref_horizon = int(belief_ref_horizon)
        # Provide defaults before super().__init__(), because base init calls _set_observation_space().
        self.belief_obs_dim = 21
        self.belief_err_dim = 0
        self.belief_prev_act_dim = 3
        self.belief_ref_dim = self.belief_ref_horizon * 6
        self.belief_priv_dim = 14
        self.belief_total_dim = (
            self.belief_obs_dim
            + self.belief_err_dim
            + self.belief_prev_act_dim
            + self.belief_ref_dim
            + self.belief_priv_dim
        )
        self._defer_env_init_log = True
        super().__init__(*args, **kwargs)
        self.is_io_history = observation_mode == "history"
        self.is_env_randomized = True
        self.is_disturbance = False
        self.is_full_traj = False
        self.is_rotor_dynamics = False
        self.is_action_filter = False
        self.is_ema_action = False
        self.is_delayed = False

        self.s_dim = 21
        self.d_dim = 6
        self.o_dim = self.obs_curr_dim + self.history_len * (self.s_dim + self.d_dim + self.a_dim)
        self.s_buffer = deque(np.zeros((self.history_len, self.s_dim), dtype=np.float32), maxlen=self.history_len)
        self.d_buffer = deque(np.zeros((self.history_len, self.d_dim), dtype=np.float32), maxlen=self.history_len)

        self.belief_obs_dim = self.s_dim
        self.belief_err_dim = 0
        self.belief_prev_act_dim = self.a_dim
        self.belief_ref_dim = self.belief_ref_horizon * 6  # [xQ_ref(3), vQ_ref(3)] * horizon
        self.belief_priv_dim = 14  # [xP(3), vP(3), q(3), dq(3), mP(1), cable_length(1)]
        self.belief_total_dim = (
            self.belief_obs_dim
            + self.belief_err_dim
            + self.belief_prev_act_dim
            + self.belief_ref_dim
            + self.belief_priv_dim
        )

        self.observation_space = self._set_observation_space()
        self._defer_env_init_log = False
        self._init_env()

    def _set_observation_space(self):
        belief_total_dim = (
            getattr(self, "belief_obs_dim", 21)
            + getattr(self, "belief_err_dim", 0)
            + getattr(self, "belief_prev_act_dim", getattr(self, "a_dim", 3))
            + getattr(self, "belief_ref_dim", int(getattr(self, "belief_ref_horizon", 5)) * 6)
            + getattr(self, "belief_priv_dim", 14)
        )
        if self.observation_mode == "current":
            obs_dim = self.obs_curr_dim
        elif self.observation_mode == "history":
            obs_dim = self.o_dim
        else:
            obs_dim = belief_total_dim
        return Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _init_env(self):
        if getattr(self, "_defer_env_init_log", False):
            return
        super()._init_env()

    def _init_history_ff(self):
        s_curr = np.copy(self._get_state_curr())
        d_curr = np.concatenate([self.xQd[0] / self.pos_bound, self.vQd[0] / self.vel_bound])
        a_curr = np.copy(self.action)

        self.s_buffer.extend([s_curr] * self.history_len)
        self.d_buffer.extend([d_curr] * self.history_len)
        self.a_buffer.extend([a_curr] * self.history_len)
        if self.is_record_action:
            self.action_record = np.zeros((self.max_timesteps, self.a_dim))
        if self.is_record_xP:
            self.xP_record = np.zeros((self.max_timesteps, 3))

    def _get_obs(self):
        obs_curr = self._get_obs_curr().astype(np.float32)
        if self.observation_mode == "current":
            return obs_curr
        if self.observation_mode == "belief":
            obs = self.s_curr.astype(np.float32)
            prev_act = self.action.astype(np.float32)
            ref_preview = self._get_ref_preview().astype(np.float32)
            priv_state = self._get_privileged_state().astype(np.float32)
            return np.concatenate([obs, prev_act, ref_preview, priv_state]).astype(np.float32)

        s_buffer = np.asarray(self.s_buffer, dtype=np.float32).flatten()
        d_buffer = np.asarray(self.d_buffer, dtype=np.float32).flatten()
        a_buffer = np.asarray(self.a_buffer, dtype=np.float32).flatten()
        io_history = np.concatenate([s_buffer, d_buffer, a_buffer])
        return np.concatenate([obs_curr, io_history]).astype(np.float32)

    def _get_obs_curr(self):
        self.s_curr = self._get_state_curr()
        return np.concatenate([self.s_curr, self.action])

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3]
        self.RQ = euler2rot(quat2euler_raw(self.data.qpos[3:7]))
        self.xP = self.data.qpos[7:10]
        self.vQ = self.data.qvel[0:3]
        self.ωQ = self.data.qvel[3:6]
        self.vP = self.data.qvel[6:9]

        cable_vec = self.xP - self.xQ
        cable_dist = max(np.linalg.norm(cable_vec), 1e-6)
        q = cable_vec / cable_dist

        return np.concatenate(
            [
                self.xQ / self.pos_bound,
                self.RQ.flatten(),
                self.vQ / self.vel_bound,
                self.ωQ,
                q,
            ]
        )

    def _get_ref_preview(self):
        max_ref_idx = self.xQd.shape[0] - 1
        ref_indices = np.clip(
            np.arange(self.timestep, self.timestep + self.belief_ref_horizon),
            0,
            max_ref_idx,
        )
        xq_ref = self.xQd[ref_indices] / self.pos_bound
        vq_ref = self.vQd[ref_indices] / self.vel_bound
        return np.concatenate([xq_ref, vq_ref], axis=-1).reshape(-1)

    def _get_privileged_state(self):
        xQ_true = self.data.qpos[0:3]
        xP_true = self.data.qpos[7:10]
        vQ_true = self.data.qvel[0:3]
        vP_true = self.data.qvel[6:9]

        cable_vec = xP_true - xQ_true
        cable_dist = max(np.linalg.norm(cable_vec), 1e-6)
        q = cable_vec / cable_dist
        rel_vel = vP_true - vQ_true
        dq = (rel_vel - q * np.dot(q, rel_vel)) / cable_dist

        return np.concatenate(
            [
                xP_true / self.pos_bound,
                vP_true / self.vel_bound,
                q,
                dq / self.vel_bound,
                np.array([self.mP, self.cable_length], dtype=np.float32),
            ]
        ).astype(np.float32)


SCENARIO_SPEC = ScenarioSpec(
    name="s3_onboard",
    env_class=QuadrotorPayloadEnvS3Onboard,
    description="Onboard-focused S3 observation: remove payload position/error and add rope direction vector.",
)
