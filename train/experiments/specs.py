from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    env_class: type
    description: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy: str | type
    description: str
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
