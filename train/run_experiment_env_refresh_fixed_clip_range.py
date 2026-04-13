"""
Variant of run_experiment_env_refresh.py with fixed PPO clip range.

The original script reassigns model.clip_range before each refresh chunk via
global_linear_schedule_for_chunk(...), which linearly decays clip_range to 0.
This script keeps the same training flow but overrides that schedule builder
to return a constant schedule.
"""

try:
    from train import run_experiment_env_refresh as base
except ModuleNotFoundError:
    import run_experiment_env_refresh as base


def fixed_clip_schedule_for_chunk(initial_value, chunk_end_step=None, total_steps=None, **_kwargs):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    fixed_value = float(initial_value)

    def schedule(_progress):
        return fixed_value

    return schedule


base.global_linear_schedule_for_chunk = fixed_clip_schedule_for_chunk


if __name__ == "__main__":
    base.main()
