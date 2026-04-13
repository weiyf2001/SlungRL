from collections import deque

from numpy import concatenate, zeros

from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from envs.utils.env_randomizer_v2a import EnvRandomizerV2A


class QuadrotorPayloadEnvV2A(QuadrotorPayloadEnvV2):
    def __init__(self, *args, **kwargs):
        self._suppress_init_banner = True
        super().__init__(*args, **kwargs)

        self.env_randomizer = EnvRandomizerV2A(model=self.model)
        self.is_env_randomized = True

        self.s_dim = 24
        self.o_dim = 300 if self.control_scheme in ["srt", "ctbr"] else 294
        self.s_buffer = deque(zeros((self.history_len, self.s_dim)), maxlen=self.history_len)
        self.observation_space = self._set_observation_space()

        self._suppress_init_banner = False
        self._init_env()

    def _init_env(self):
        if getattr(self, "_suppress_init_banner", False):
            return
        super()._init_env()

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3] + self._clip_position_noise()
        self.RQ = self._get_noisy_rotation()
        self.xP = self.data.qpos[7:10] + self._clip_position_noise()
        self.vQ = self.data.qvel[0:3] + self._clip_velocity_noise()
        self.ωQ = self.data.qvel[3:6] + self._clip_angular_noise()
        self.vP = self.data.qvel[6:9] + self._clip_velocity_noise()

        # v2a intentionally hides payload mass and cable length from the observation.
        return concatenate(
            [
                self.xQ / self.pos_bound,
                self.RQ.flatten(),
                self.xP / self.pos_bound,
                self.vQ / self.vel_bound,
                self.ωQ,
                self.vP / self.vel_bound,
            ]
        )

    def _clip_position_noise(self):
        return self._clip_noise(scale=0.01, bound=0.0025)

    def _clip_velocity_noise(self):
        return self._clip_noise(scale=0.02, bound=0.005)

    def _clip_angular_noise(self):
        return self._clip_noise(scale=self._pi() / 30, bound=self._pi() / 60)

    def _get_noisy_rotation(self):
        from numpy.random import normal
        from numpy import clip

        from envs.utils.rotation_transformations import euler2rot, quat2euler_raw

        return euler2rot(
            quat2euler_raw(self.data.qpos[3:7]) + clip(normal(loc=0, scale=self._pi() / 60, size=3), -self._pi() / 120, self._pi() / 120)
        )

    def _clip_noise(self, scale, bound):
        from numpy import clip
        from numpy.random import normal

        return clip(normal(loc=0, scale=scale, size=3), -bound, bound)

    @staticmethod
    def _pi():
        from numpy import pi

        return pi


QuadrotorPayloadEnv = QuadrotorPayloadEnvV2A
