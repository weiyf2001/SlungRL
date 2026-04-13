import numpy as np
from numpy.linalg import norm
from numpy.random import uniform

from envs.utils.env_randomizer import EnvRandomizer, random_deviation_quaternion


class EnvRandomizerV2A(EnvRandomizer):
    def __init__(self, model):
        super().__init__(model)
        self.payload_mass_range = (0.05, 0.20)
        self.tendon_length_range = (0.50, 1.00)
        self.tau_scale_range = (0.8, 1.2)

    def randomize_env(self, model):
        model = self.reset_env(model=model)

        for i in range(self.nbody):
            if self.has_payload and i == self.payload_body_id:
                continue

            model.body_ipos[i] = self._default_body_ipos[i] + uniform(
                size=3,
                low=-self.ipos_noise_scale,
                high=self.ipos_noise_scale,
            )
            body_iquat = random_deviation_quaternion(self._default_body_iquat[i], self.iquat_noise_scale)
            model.body_iquat[i] = body_iquat / norm(body_iquat)
            model.body_mass[i] = self._default_body_mass[i] * (
                1.0 + uniform(low=-self.mass_noise_scale, high=self.mass_noise_scale)
            )
            model.body_inertia[i] = self._default_body_inertia[i] * (
                1.0 + uniform(size=3, low=-self.inertia_noise_scale, high=self.inertia_noise_scale)
            )

        for gear in model.actuator_gear:
            gear *= 1.0 + uniform(
                low=-self.actuator_gear_noise_scale,
                high=self.actuator_gear_noise_scale,
                size=len(gear),
            )

        payload_mass = self._default_payload_mass
        tendon_length = self._default_tendon_max_length
        if self.has_payload:
            payload_mass = uniform(low=self.payload_mass_range[0], high=self.payload_mass_range[1])
            tendon_length = uniform(low=self.tendon_length_range[0], high=self.tendon_length_range[1])
            model.body_ipos[self.payload_body_id] = self._default_hook_core_site_pos + np.array([0.0, 0.0, -tendon_length])
            model.body_mass[self.payload_body_id] = payload_mass
            model.tendon_range[0][1] = tendon_length

        tau_scale = uniform(low=self.tau_scale_range[0], high=self.tau_scale_range[1])
        tau_up = self._default_tau_up * tau_scale
        tau_down = self._default_tau_down * tau_scale

        return model, payload_mass, tendon_length, tau_up, tau_down
