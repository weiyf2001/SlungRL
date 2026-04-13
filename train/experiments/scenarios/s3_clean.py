from collections import deque

import numpy as np
from gymnasium.spaces import Box

from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from envs.utils.rotation_transformations import euler2rot, quat2euler_raw
from train.experiments.specs import ScenarioSpec


class QuadrotorPayloadEnvS3Clean(QuadrotorPayloadEnvV2):
    """
    S3 clean scenario:
    - payload mass and cable length are randomized every episode
    - all payload states are removed from the observation
    - only quadrotor states and quadrotor tracking errors remain observable
    - no sensor noise, delay, filtering, actuator lag, or disturbances
    """

    obs_curr_dim = 27

    def __init__(self, *args, observation_mode: str = "history", **kwargs):
        if observation_mode not in {"current", "history"}:
            raise ValueError(f"Unsupported observation_mode '{observation_mode}'.")

        self.observation_mode = observation_mode
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

        self.s_dim = 18
        self.d_dim = 6
        self.o_dim = self.obs_curr_dim + self.history_len * (self.s_dim + self.d_dim + self.a_dim)
        self.s_buffer = deque(np.zeros((self.history_len, self.s_dim), dtype=np.float32), maxlen=self.history_len)
        self.d_buffer = deque(np.zeros((self.history_len, self.d_dim), dtype=np.float32), maxlen=self.history_len)
        self.observation_space = self._set_observation_space()
        self._defer_env_init_log = False
        self._init_env()

    def _set_observation_space(self):
        obs_dim = self.obs_curr_dim if self.observation_mode == "current" else self.o_dim
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

        s_buffer = np.asarray(self.s_buffer, dtype=np.float32).flatten()
        d_buffer = np.asarray(self.d_buffer, dtype=np.float32).flatten()
        a_buffer = np.asarray(self.a_buffer, dtype=np.float32).flatten()
        io_history = np.concatenate([s_buffer, d_buffer, a_buffer])
        return np.concatenate([obs_curr, io_history]).astype(np.float32)

    def _get_obs_curr(self):
        self.s_curr = self._get_state_curr()

        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]
        self.e_curr = np.concatenate([self.RQ.T @ self.exQ, self.RQ.T @ self.evQ])

        return np.concatenate([self.s_curr, self.e_curr, self.action])

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3]
        self.RQ = euler2rot(quat2euler_raw(self.data.qpos[3:7]))
        self.xP = self.data.qpos[7:10]
        self.vQ = self.data.qvel[0:3]
        self.ωQ = self.data.qvel[3:6]
        self.vP = self.data.qvel[6:9]
        return np.concatenate(
            [
                self.xQ / self.pos_bound,
                self.RQ.flatten(),
                self.vQ / self.vel_bound,
                self.ωQ,
            ]
        )


SCENARIO_SPEC = ScenarioSpec(
    name="s3_clean",
    env_class=QuadrotorPayloadEnvS3Clean,
    description="Only quadrotor states are observed; payload states and latent parameters are hidden.",
)
