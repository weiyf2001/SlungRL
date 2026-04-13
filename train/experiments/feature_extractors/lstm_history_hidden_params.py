import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LimitedParameterLSTMFeaturesExtractor(BaseFeaturesExtractor):
    OBSERVATION_LAYOUTS = {
        234: {
            "obs_curr_dim": 39,
            "s_dim": 24,
            "d_dim": 12,
            "a_dim": 3,
            "history_len": 5,
        },
        213: {
            "obs_curr_dim": 33,
            "s_dim": 21,
            "d_dim": 12,
            "a_dim": 3,
            "history_len": 5,
        },
        162: {
            "obs_curr_dim": 27,
            "s_dim": 18,
            "d_dim": 6,
            "a_dim": 3,
            "history_len": 5,
        },
    }

    def __init__(
        self,
        observation_space: spaces.Box,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        projection_dim: int = 32,
    ):
        if observation_space.shape is None:
            raise ValueError(
                "LimitedParameterLSTMFeaturesExtractor requires a fixed observation dimension."
            )

        self.expected_obs_dim = observation_space.shape[0]
        if self.expected_obs_dim not in self.OBSERVATION_LAYOUTS:
            supported_dims = ", ".join(str(dim) for dim in sorted(self.OBSERVATION_LAYOUTS))
            raise ValueError(
                f"LimitedParameterLSTMFeaturesExtractor expects observation dim in {{{supported_dims}}}, "
                f"got {observation_space.shape}."
            )

        layout = self.OBSERVATION_LAYOUTS[self.expected_obs_dim]
        self.history_len = layout["history_len"]
        self.s_dim = layout["s_dim"]
        self.d_dim = layout["d_dim"]
        self.a_dim = layout["a_dim"]
        self.obs_curr_dim = layout["obs_curr_dim"]

        self.state_history_dim = self.s_dim + self.d_dim
        features_dim = self.obs_curr_dim + projection_dim * 2
        super().__init__(observation_space, features_dim)

        self.state_history_lstm = nn.LSTM(
            input_size=self.state_history_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.action_history_lstm = nn.LSTM(
            input_size=self.a_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.state_projection = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, projection_dim),
            nn.SiLU(),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, projection_dim),
            nn.SiLU(),
        )

    def _split_observation(self, obs_full: th.Tensor):
        batch_size = obs_full.shape[0]
        obs_curr = obs_full[:, : self.obs_curr_dim]

        idx = self.obs_curr_dim
        s_history = obs_full[:, idx : idx + self.s_dim * self.history_len].reshape(batch_size, self.history_len, self.s_dim)
        idx += self.s_dim * self.history_len
        d_history = obs_full[:, idx : idx + self.d_dim * self.history_len].reshape(batch_size, self.history_len, self.d_dim)
        idx += self.d_dim * self.history_len
        a_history = obs_full[:, idx : idx + self.a_dim * self.history_len].reshape(batch_size, self.history_len, self.a_dim)
        return obs_curr, s_history, d_history, a_history

    @staticmethod
    def _last_hidden(output: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]) -> th.Tensor:
        _, (hidden_state, _) = output
        return hidden_state[-1]

    def forward(self, obs_full: th.Tensor) -> th.Tensor:
        obs_curr, s_history, d_history, a_history = self._split_observation(obs_full)
        state_history = th.cat([s_history, d_history], dim=-1)
        state_latent = self.state_projection(self._last_hidden(self.state_history_lstm(state_history)))
        action_latent = self.action_projection(self._last_hidden(self.action_history_lstm(a_history)))
        return th.cat([obs_curr, state_latent, action_latent], dim=1)
