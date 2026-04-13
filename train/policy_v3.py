from stable_baselines3.common.policies import ActorCriticPolicy

from train.feature_extractor_v3 import CustomFeaturesExtractorV3


class CustomActorCriticPolicyV3(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=CustomFeaturesExtractorV3)
