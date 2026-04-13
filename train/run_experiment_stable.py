import argparse
import copy
import os
import sys
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from train.experiments.registry import get_policy_spec, get_scenario_spec
from train.run_paths import configure_experiment_logger, get_checkpoint_path, get_run_directories


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a scenario+policy experiment.")
    parser.add_argument("--scenario", type=str, default="s0_full_state", help="Scenario registry key.")
    parser.add_argument("--policy", type=str, default="mlp_memoryless", help="Policy registry key.")
    parser.add_argument("--id", type=str, default="untitled", help="Experiment run identifier.")
    parser.add_argument("--visualize", type=bool, default=False, help="Render the environment.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments.")
    parser.add_argument(
        "--vec_env",
        type=str,
        default="subproc",
        choices=["dummy", "subproc"],
        help="Vectorized environment backend.",
    )
    parser.add_argument("--num_steps", type=int, default=int(8e7), help="Total training timesteps.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint run id under the same scenario/policy namespace.",
    )
    return vars(parser.parse_args())


class RewardLoggingCallback(BaseCallback):
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


def build_env_factory(env_class, render_mode, env_kwargs, seed):
    def _init():
        return env_class(render_mode=render_mode, env_num=seed, **env_kwargs)

    set_random_seed(seed)
    return _init


def main():
    args_dict = parse_arguments()
    scenario_spec = get_scenario_spec(args_dict["scenario"])
    policy_spec = get_policy_spec(args_dict["policy"])

    print(
        {
            **args_dict,
            "scenario_description": scenario_spec.description,
            "policy_description": policy_spec.description,
        }
    )

    version = os.path.join("experiments", scenario_spec.name, policy_spec.name)
    run_paths = get_run_directories(version, args_dict["id"])
    tb_path = run_paths["tb_dir"]
    checkpoints_path = run_paths["checkpoints_dir"]
    render_mode = "human" if args_dict["visualize"] else None

    env_kwargs = copy.deepcopy(scenario_spec.default_kwargs)
    env_kwargs.update(copy.deepcopy(policy_spec.env_kwargs))
    num_envs = args_dict["num_envs"]
    env_fns = [
        build_env_factory(
            env_class=scenario_spec.env_class,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
            seed=seed,
        )
        for seed in range(num_envs)
    ]
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

    horizon_len = 64
    n_steps = horizon_len if num_envs >= 16 else 256
    batch_size = min(32 * num_envs, 4096)

    algo_kwargs = copy.deepcopy(policy_spec.algo_kwargs)
    model = PPO(
        policy_spec.policy,
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=linear_schedule(0.05),
        verbose=1,
        policy_kwargs=copy.deepcopy(policy_spec.policy_kwargs),
        tensorboard_log=None,
        device=args_dict["device"],
        **algo_kwargs,
    )
    configure_experiment_logger(model, tb_path)

    if args_dict["checkpoint"] is not None:
        checkpoint_path = get_checkpoint_path(version, args_dict["checkpoint"])
        model.set_parameters(checkpoint_path)

    model.learn(
        total_timesteps=args_dict["num_steps"],
        progress_bar=True,
        callback=callback_list,
    )
    model.save(os.path.join(checkpoints_path, "final_model"))


if __name__ == "__main__":
    main()
