from train.experiments.policies import POLICY_SPECS
from train.experiments.scenarios import SCENARIO_SPECS
from train.experiments.specs import PolicySpec, ScenarioSpec


def get_scenario_spec(name: str) -> ScenarioSpec:
    if name not in SCENARIO_SPECS:
        available = ", ".join(sorted(SCENARIO_SPECS))
        raise KeyError(f"Unknown scenario '{name}'. Available scenarios: {available}")
    return SCENARIO_SPECS[name]


def get_policy_spec(name: str) -> PolicySpec:
    if name in POLICY_SPECS:
        return POLICY_SPECS[name]

    alias_candidates = [name.replace("-", "_"), name.replace("_", "-")]
    for alias in alias_candidates:
        if alias in POLICY_SPECS:
            return POLICY_SPECS[alias]

    available = ", ".join(sorted(POLICY_SPECS))
    raise KeyError(f"Unknown policy '{name}'. Available policies: {available}")
