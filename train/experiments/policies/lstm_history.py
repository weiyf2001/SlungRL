from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th

from train.experiments.feature_extractors import LimitedParameterLSTMFeaturesExtractor
from train.experiments.specs import PolicySpec


class HistoryLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LimitedParameterLSTMFeaturesExtractor)


POLICY_SPEC = PolicySpec(
    name="lstm_history",
    policy=HistoryLSTMPolicy,
    description="LSTM feature extractor over structured observation history and action history.",
    policy_kwargs={
        "activation_fn": th.nn.SiLU,
        "net_arch": {
            "pi": [256, 128, 64],
            "vf": [256, 128, 64],
        },
    },
    algo_kwargs={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "ent_coef": 0.001,
    },
    env_kwargs={
        "observation_mode": "history",
    },
)
