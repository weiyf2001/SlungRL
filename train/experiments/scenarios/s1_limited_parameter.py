from collections import deque

import numpy as np
from gymnasium.spaces import Box
from numpy.linalg import inv, norm
from numpy import abs, asarray, clip, concatenate, copy, dot, exp, mean, ones, pi, round, sum, sqrt, tanh, zeros
from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from envs.utils.rotation_transformations import euler2rot, quat2euler_raw
from train.experiments.specs import ScenarioSpec


class QuadrotorPayloadEnvS1LimitedParameter(QuadrotorPayloadEnvV2):
    """
    S1 limited-parameter scenario:
    - payload mass and cable length are randomized every episode
    - payload mass and cable length are hidden from the observation
    - rope direction vector is included in the observable state
    - no sensor noise, delay, filtering, actuator lag, or disturbances
    - history observation can be enabled so policies can infer hidden parameters
    """

    obs_curr_dim = 36

    def __init__(self, *args, observation_mode: str = "history", **kwargs):
        if observation_mode not in {"current", "history"}:
            raise ValueError(f"Unsupported observation_mode '{observation_mode}'.")

        self.observation_mode = observation_mode
        self._defer_env_init_log = True
        super().__init__(*args, **kwargs)
        self.is_io_history = observation_mode == "history"
        self.is_env_randomized = False
        self.is_disturbance = False
        self.is_full_traj = False
        self.is_rotor_dynamics = False
        self.is_action_filter = False
        self.is_ema_action = False
        self.is_delayed = False

        self.s_dim = 27
        self.o_dim = self.obs_curr_dim + self.history_len * (self.s_dim + self.d_dim + self.a_dim)
        self.s_buffer = deque(np.zeros((self.history_len, self.s_dim), dtype=np.float32), maxlen=self.history_len)
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

        self.exP = self.xP - self.xPd[self.timestep]
        self.evP = self.vP - self.vPd[self.timestep]
        self.e_curr = np.concatenate([self.RQ.T @ self.exP, self.RQ.T @ self.evP])

        return np.concatenate([self.s_curr, self.e_curr, self.action])

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3]
        self.RQ = euler2rot(quat2euler_raw(self.data.qpos[3:7]))
        self.xP = self.data.qpos[7:10]
        self.vQ = self.data.qvel[0:3]
        self.ωQ = self.data.qvel[3:6]
        self.vP = self.data.qvel[6:9]
        cable_vec = self.xP - self.xQ
        cable_dist = max(norm(cable_vec), 1e-6)
        q = cable_vec / cable_dist
        return np.concatenate(
            [
                self.xQ / self.pos_bound,
                self.RQ.flatten(),
                self.xP / self.pos_bound,
                self.vQ / self.vel_bound,
                self.ωQ,
                self.vP / self.vel_bound,
                q,
            ]
        )

    # def _get_reward(self):
    #     xQ = self.data.qpos[0:3]
    #     xP = self.data.qpos[7:10]
    #     vQ = self.data.qvel[0:3]
    #     vP = self.data.qvel[6:9]
    #
    #     cable_vec = xP - xQ
    #     cable_dist = max(norm(cable_vec), 1e-6)
    #     q = cable_vec / cable_dist
    #     rel_vel = vP - vQ
    #     dq = (rel_vel - q * dot(q, rel_vel)) / cable_dist
    #
    #     prev_action = asarray(self.a_buffer[-1], dtype=np.float32) if len(self.a_buffer) else self.action_offset
    #     action_delta = norm(self.raw_action - prev_action)
    #
    #     errors = {
    #         'xP': norm(xP - self.xPd[self.timestep]),
    #         'vP': norm(vP - self.vPd[self.timestep]),
    #         'q': norm(q - self.qd[self.timestep]),
    #         'dq': norm(dq - self.dqd[self.timestep]),
    #         'xQ': norm(xQ - self.xQd[self.timestep]),
    #         'l': abs(cable_dist - self.cable_length),
    #         'ψQ': abs(quat2euler_raw(self.data.qpos[3:7])[2]),
    #         'ωQ': norm(self.data.qvel[3:6]),
    #         'a': norm(self.raw_action),
    #         'Δa': action_delta,
    #     }
    #
    #     weights = {
    #         'xP': 0.35,
    #         'vP': 0.15,
    #         'q': 0.20,
    #         'dq': 0.10,
    #         'xQ': 0.06,
    #         'l': 0.03,
    #         'ψQ': 0.05,
    #         'ωQ': 0.03,
    #         'a': 0.015,
    #         'Δa': 0.015,
    #     }
    #     scales = {
    #         'xP': 0.10,
    #         'vP': 0.35,
    #         'q': 0.20,
    #         'dq': 0.35,
    #         'xQ': 0.20,
    #         'l': 0.05,
    #         'ψQ': pi / 6,
    #         'ωQ': 0.35,
    #         'a': 0.80,
    #         'Δa': 0.25,
    #     }
    #
    #     # Use a slower-decaying kernel than exp(-e / s) so large initial errors still carry learning signal.
    #     rewards = {
    #         key: 1.0 / (1.0 + (errors[key] / max(scales[key], 1e-6)) ** 2)
    #         for key in weights
    #     }
    #     total_reward = float(sum(weights[key] * rewards[key] for key in weights))
    #     return total_reward, rewards, errors


SCENARIO_SPEC = ScenarioSpec(
    name="s1_limited_parameter",
    env_class=QuadrotorPayloadEnvS1LimitedParameter,
    description="Hidden payload mass and cable length with history-based parameter inference.",
)
