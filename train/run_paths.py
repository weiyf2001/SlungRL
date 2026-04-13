import os

from stable_baselines3.common.logger import configure


def get_run_directories(version: str, experiment_id: str) -> dict[str, str]:
    run_dir = os.path.join("runs", version, experiment_id)
    tb_dir = os.path.join(run_dir, "tb")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tb_dir": tb_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def get_checkpoint_path(version: str, experiment_id: str, checkpoint_name: str = "best_model") -> str:
    return os.path.join("runs", version, experiment_id, "checkpoints", checkpoint_name)


def configure_experiment_logger(model, tb_dir: str) -> None:
    logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
