import numpy as np
from numpy import clip, pi

from envs.utils.rotation_transformations import euler2rot, quat2euler_raw
from train.experiments.scenarios.s3_clean import QuadrotorPayloadEnvS3Clean
from train.experiments.specs import ScenarioSpec


class QuadrotorPayloadEnvS4Robust(QuadrotorPayloadEnvS3Clean):
    """
    S4 robust scenario:
    - same observable variables as S3 clean
    - payload mass and cable length are randomized every episode
    - sensor noise is injected into the quadrotor state estimate
    - action delay and rotor dynamics are enabled
    """

    def __init__(self, *args, observation_mode: str = "history", **kwargs):
        super().__init__(*args, observation_mode=observation_mode, **kwargs)
        self.is_io_history = observation_mode == "history"
        self.is_env_randomized = True
        self.is_disturbance = False
        self.is_full_traj = False
        self.is_rotor_dynamics = True
        self.is_action_filter = False
        self.is_ema_action = False
        self.is_delayed = True

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3] + clip(np.random.normal(loc=0, scale=0.01, size=3), -0.0025, 0.0025)
        self.RQ = euler2rot(
            quat2euler_raw(self.data.qpos[3:7])
            + clip(np.random.normal(loc=0, scale=pi / 60, size=3), -pi / 120, pi / 120)
        )
        self.xP = self.data.qpos[7:10]
        self.vQ = self.data.qvel[0:3] + clip(np.random.normal(loc=0, scale=0.02, size=3), -0.005, 0.005)
        self.ωQ = self.data.qvel[3:6] + clip(np.random.normal(loc=0, scale=pi / 30, size=3), -pi / 60, pi / 60)
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
    name="s4_robust",
    env_class=QuadrotorPayloadEnvS4Robust,
    description="S3-clean observation with sensor noise, action delay, and actuator lag enabled.",
)
