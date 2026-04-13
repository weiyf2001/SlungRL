import argparse
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from envs.quadrotor_env import QuadrotorEnv
from envs.quadrotor_mini_env import QuadrotorMiniEnv
from envs.quadrotor_payload_env_v2a import QuadrotorPayloadEnvV2A
from train.run_paths import configure_experiment_logger, get_checkpoint_path, get_run_directories


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO v2a with hidden payload mass and cable length.")
    parser.add_argument("--id", type=str, default="untitled", help="Provide experiment name and ID.")
    parser.add_argument("--visualize", type=bool, default=False, help="Choose visualization option.")
    parser.add_argument("--device", type=str, default="cuda", help="Provide device info.")
    parser.add_argument("--num_envs", type=int, default=1, help="Provide number of parallel environments.")
    parser.add_argument(
        "--vec_env",
        type=str,
        default="subproc",
        choices=["dummy", "subproc"],
        help="Vectorized env backend: dummy (single-process) or subproc (multi-process).",
    )
    parser.add_argument("--num_steps", type=int, default=int(1e8), help="Provide number of training steps.")
    parser.add_argument("--env", type=str, default="payload", help="Choose environment [falcon, mini, payload].")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load a pretrained v2a model ID from runs/v2a/<id>/checkpoints.",
    )
    return vars(parser.parse_args())


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "subreward" in info:
                for key, value in info["subreward"].items():
                    self.logger.record(f"subreward/{key}", value)
            if "reward_error" in info:
                for key, value in info["reward_error"].items():
                    self.logger.record(f"reward_error/{key}", value)
        return True


class EvalCallbackWithTimestamp(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward_prev = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.best_model_save_path is not None and self.best_mean_reward > self.best_mean_reward_prev:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_save_path = os.path.join(self.best_model_save_path, f"best_model_{timestamp}")
            self.model.save(new_save_path)
            print(f"New best model saved to {new_save_path}")
            self.best_mean_reward_prev = self.best_mean_reward
        return result


def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def schedule(progress):
        return progress * initial_value

    return schedule


def main():
    args_dict = parse_arguments()
    print(args_dict)

    experiment_id = args_dict["id"]
    run_paths = get_run_directories("v2a", experiment_id)
    tb_path = run_paths["tb_dir"]
    checkpoints_path = run_paths["checkpoints_dir"]
    render_mode = "human" if args_dict["visualize"] else None

    env_dict = {
        "falcon": QuadrotorEnv,
        "mini": QuadrotorMiniEnv,
        "payload": QuadrotorPayloadEnvV2A,
    }
    quad_env = env_dict.get(args_dict["env"], QuadrotorEnv)

    def create_env(seed=0):
        def _init():
            return quad_env(render_mode=render_mode, env_num=seed)

        set_random_seed(seed)
        return _init

    num_envs = args_dict["num_envs"]
    env_fns = [create_env(seed=i) for i in range(num_envs)]
    vec_env_cls = SubprocVecEnv if args_dict["vec_env"] == "subproc" else DummyVecEnv
    env = VecMonitor(vec_env_cls(env_fns))

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)
    eval_callback = EvalCallbackWithTimestamp(
        env,
        callback_on_new_best=stop_callback,
        eval_freq=2500,
        best_model_save_path=checkpoints_path,
        verbose=1,
    )
    callback_list = CallbackList([RewardLoggingCallback(), eval_callback])

    activation_fn = th.nn.SiLU
    net_arch = {"pi": [256, 128, 64], "vf": [256, 128, 64]}

    horizon_len = 64
    n_steps = horizon_len if num_envs >= 16 else 256
    batch_size = min(32 * num_envs, 4096)

    model = PPO(
        "MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=linear_schedule(0.05),
        ent_coef=0.001,
        verbose=1,
        policy_kwargs={"activation_fn": activation_fn, "net_arch": net_arch},
        tensorboard_log=None,
        device=args_dict["device"],
    )
    configure_experiment_logger(model, tb_path)

    if args_dict["checkpoint"] is not None:
        checkpoint_path = get_checkpoint_path("v2a", args_dict["checkpoint"])
        model.set_parameters(checkpoint_path)

    model.learn(
        total_timesteps=args_dict["num_steps"],
        progress_bar=True,
        callback=callback_list,
    )
    model.save(os.path.join(checkpoints_path, "final_model"))


if __name__ == "__main__":
    main()
