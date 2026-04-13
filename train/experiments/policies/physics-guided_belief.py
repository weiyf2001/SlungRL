"""
Compatibility shim.

Python module imports cannot use '-' in module names.
Use train.experiments.policies.physics_guided_belief for runtime imports.
"""

from train.experiments.policies.physics_guided_belief import *  # noqa: F401,F403
