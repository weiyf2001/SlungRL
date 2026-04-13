import numpy as np
from gymnasium.spaces import Box

from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from envs.utils.rotation_transformations import euler2rot, quat2euler_raw
from train.experiments.specs import ScenarioSpec


class QuadrotorPayloadEnvS0FullState(QuadrotorPayloadEnvV2):
    """
    S0 oracle scenario:
    - fixed payload parameters
    - no sensor noise
    - no delay, filtering, actuator lag, or disturbances
    - memoryless full-state observation for the current timestep
    """

    obs_curr_dim = 41

    def __init__(self, *args, observation_mode: str = "current", **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_mode = "current" if observation_mode != "current" else observation_mode
        self.is_io_history = False
        self.is_env_randomized = True
        self.is_disturbance = False
        self.is_full_traj = False
        self.is_rotor_dynamics = False
        self.is_action_filter = False
        self.is_ema_action = False
        self.is_delayed = False
        self.observation_space = self._set_observation_space()

    def _set_observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_curr_dim,), dtype=np.float32)

    def _get_obs(self):
        return self._get_obs_curr().astype(np.float32)

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
                self.xP / self.pos_bound,
                self.vQ / self.vel_bound,
                self.ωQ,
                self.vP / self.vel_bound,
                [self.mP, self.cable_length],
            ]
        )


SCENARIO_SPEC = ScenarioSpec(
    name="s0_full_state",
    env_class=QuadrotorPayloadEnvS0FullState,
    description="Oracle full-state payload tracking without engineering degradations.",
)
