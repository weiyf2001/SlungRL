import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeaturesExtractorV3(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        projection_dim: int = 32,
    ):
        self.history_len = 5
        self.future_len = 5
        self.s_dim = 26
        self.d_dim = 12
        self.a_dim = 3
        self.ff_dim = 12
        self.obs_curr_dim = 41
        self.expected_obs_dim = 306

        if observation_space.shape is None or observation_space.shape[0] != self.expected_obs_dim:
            raise ValueError(
                f"CustomFeaturesExtractorV3 expects observation dim {self.expected_obs_dim}, "
                f"got {observation_space.shape}."
            )

        self.state_history_dim = self.s_dim + self.d_dim
        features_dim = self.obs_curr_dim + projection_dim * 3
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
        self.future_lstm = nn.LSTM(
            input_size=self.ff_dim,
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
        self.future_projection = nn.Sequential(
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
        idx += self.a_dim * self.history_len
        ff = obs_full[:, idx : idx + self.ff_dim * self.future_len].reshape(batch_size, self.future_len, self.ff_dim)

        return obs_curr, s_history, d_history, a_history, ff

    @staticmethod
    def _last_hidden(output: tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]) -> th.Tensor:
        _, (hidden_state, _) = output
        return hidden_state[-1]

    def forward(self, obs_full: th.Tensor) -> th.Tensor:
        obs_curr, s_history, d_history, a_history, ff = self._split_observation(obs_full)

        state_history = th.cat([s_history, d_history], dim=-1)

        state_latent = self.state_projection(self._last_hidden(self.state_history_lstm(state_history)))
        action_latent = self.action_projection(self._last_hidden(self.action_history_lstm(a_history)))
        future_latent = self.future_projection(self._last_hidden(self.future_lstm(ff)))

        return th.cat([obs_curr, state_latent, action_latent, future_latent], dim=1)
