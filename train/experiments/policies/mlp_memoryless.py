import torch as th

from train.experiments.specs import PolicySpec


POLICY_SPEC = PolicySpec(
    name="mlp_memoryless",
    policy="MlpPolicy",
    description="Feed-forward MLP baseline for s0_full_state with current-step observation only.",
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
        "observation_mode": "current",
    },
)
