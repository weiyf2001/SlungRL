import torch as th

from train.experiments.specs import PolicySpec


POLICY_SPEC = PolicySpec(
    name="mlp_history",
    policy="MlpPolicy",
    description="Feed-forward MLP for s1_limited_parameter over flattened state/reference/action history.",
    policy_kwargs={
        "activation_fn": th.nn.SiLU,
        "net_arch": {
            "pi": [384, 192, 96],
            "vf": [384, 192, 96],
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
