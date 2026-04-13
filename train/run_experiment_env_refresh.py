import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from train.experiments.registry import get_policy_spec, get_scenario_spec
from train.run_paths import configure_experiment_logger, get_checkpoint_path, get_run_directories


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a scenario+policy experiment with periodic train/eval environment refresh."
    )
    parser.add_argument("--scenario", type=str, default="s0_full_state", help="Scenario registry key.")
    parser.add_argument("--policy", type=str, default="mlp_memoryless", help="Policy registry key.")
    parser.add_argument("--id", type=str, default="untitled", help="Experiment run identifier.")
    parser.add_argument("--visualize", type=bool, default=False, help="Render the environment.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of vectorized environments.")
    parser.add_argument(
        "--vec_env",
        type=str,
        default="subproc",
        choices=["dummy", "subproc"],
        help="Vectorized environment backend.",
    )
    parser.add_argument("--num_steps", type=int, default=int(8e7), help="Total training timesteps.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Optional override for PPO learning rate. If unset, use policy default.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint run id under the same scenario/policy namespace.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=2500,
        help="Evaluation frequency in environment steps.",
    )
    parser.add_argument(
        "--env_refresh_steps",
        type=int,
        default=int(1e7),
        help="Recreate train/eval envs every N training timesteps.",
    )
    parser.add_argument(
        "--train_seed_base",
        type=int,
        default=0,
        help="Base seed offset for train envs.",
    )
    parser.add_argument(
        "--eval_seed_base",
        type=int,
        default=10_000,
        help="Base seed offset for eval envs.",
    )
    parser.add_argument(
        "--seed_stride",
        type=int,
        default=1000,
        help="Seed offset increment per refresh round.",
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
    def __init__(self, *args, global_best_mean_reward=-np.inf, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward_prev = global_best_mean_reward

    def _on_step(self) -> bool:
        should_run_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_run_eval:
            rng_state = capture_global_rng_state()
            try:
                result = super()._on_step()
            finally:
                restore_global_rng_state(rng_state)
        else:
            result = super()._on_step()

        if self.best_mean_reward > self.best_mean_reward_prev:
            if self.best_model_save_path is not None:
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


def global_linear_schedule_for_chunk(initial_value, chunk_end_step, total_steps):
    """
    Build a per-chunk schedule that maps SB3's progress to global linear decay.

    With `reset_num_timesteps=False`, SB3 uses:
      progress = 1 - num_timesteps / chunk_end_step
    where `chunk_end_step = chunk_start_step + chunk_steps`.
    We invert that relation to obtain global progress:
      global_progress = 1 - num_timesteps / total_steps
                      = (1 - chunk_end_step / total_steps)
                        + (chunk_end_step / total_steps) * progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0 for global linear schedule")
    if chunk_end_step <= 0:
        raise ValueError("chunk_end_step must be > 0 for global linear schedule")

    coeff = chunk_end_step / total_steps
    offset = 1.0 - coeff

    def schedule(progress):
        global_progress = offset + coeff * progress
        return max(0.0, initial_value * global_progress)

    return schedule


def capture_global_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_global_rng_state(state):
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if "torch_cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])


def build_env_factory(env_class, render_mode, env_kwargs, seed):
    def _init():
        return env_class(render_mode=render_mode, env_num=seed, **env_kwargs)

    return _init


def build_vec_env(env_class, render_mode, env_kwargs, num_envs, vec_env_type, seed_offset=0):
    env_fns = [
        build_env_factory(
            env_class=env_class,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
            seed=seed_offset + seed,
        )
        for seed in range(num_envs)
    ]
    vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else DummyVecEnv
    return VecMonitor(vec_env_cls(env_fns))


def build_eval_env_with_rng_isolation(env_class, render_mode, env_kwargs, seed_offset):
    rng_state = capture_global_rng_state()
    try:
        return build_vec_env(
            env_class=env_class,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
            num_envs=1,
            vec_env_type="dummy",
            seed_offset=seed_offset,
        )
    finally:
        restore_global_rng_state(rng_state)


def build_env_pair(args_dict, scenario_spec, render_mode, env_kwargs, refresh_round):
    train_seed_offset = args_dict["train_seed_base"] + refresh_round * args_dict["seed_stride"]
    eval_seed_offset = args_dict["eval_seed_base"] + refresh_round * args_dict["seed_stride"]

    train_env = build_vec_env(
        env_class=scenario_spec.env_class,
        render_mode=render_mode,
        env_kwargs=env_kwargs,
        num_envs=args_dict["num_envs"],
        vec_env_type=args_dict["vec_env"],
        seed_offset=train_seed_offset,
    )
    eval_env = build_eval_env_with_rng_isolation(
        env_class=scenario_spec.env_class,
        render_mode=render_mode,
        env_kwargs=env_kwargs,
        seed_offset=eval_seed_offset,
    )
    return train_env, eval_env, train_seed_offset, eval_seed_offset


def main():
    args_dict = parse_arguments()
    if args_dict["env_refresh_steps"] <= 0:
        raise ValueError("--env_refresh_steps must be > 0")
    if args_dict["seed_stride"] <= 0:
        raise ValueError("--seed_stride must be > 0")

    scenario_spec = get_scenario_spec(args_dict["scenario"])

    raw_policy_name = args_dict["policy"]
    policy_name_candidates = []
    for candidate in (raw_policy_name, raw_policy_name.replace("-", "_"), raw_policy_name.replace("_", "-")):
        if candidate not in policy_name_candidates:
            policy_name_candidates.append(candidate)

    policy_spec = None
    for candidate in policy_name_candidates:
        try:
            policy_spec = get_policy_spec(candidate)
            args_dict["policy"] = candidate
            break
        except KeyError:
            continue
    if policy_spec is None:
        raise KeyError(
            f"Unknown policy '{raw_policy_name}'. Tried aliases: {policy_name_candidates}"
        )

    if policy_spec.name == "physics-guided_belief" and scenario_spec.name != "s3_onboard":
        raise ValueError(
            "physics-guided_belief is preconfigured for scenario 's3_onboard'. "
            "Use --scenario s3_onboard or provide a custom belief_cfg in policy spec."
        )

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

    horizon_len = 64
    n_steps = horizon_len if args_dict["num_envs"] >= 16 else 256
    batch_size = min(32 * args_dict["num_envs"], 4096)
    algo_kwargs = copy.deepcopy(policy_spec.algo_kwargs)
    if args_dict["learning_rate"] is not None:
        algo_kwargs["learning_rate"] = args_dict["learning_rate"]

    model = None
    global_best_mean_reward = -np.inf
    refresh_round = 0
    total_target_steps = args_dict["num_steps"]
    stop_requested = False

    while not stop_requested and (model.num_timesteps if model is not None else 0) < total_target_steps:
        trained_steps = model.num_timesteps if model is not None else 0
        chunk_steps = min(args_dict["env_refresh_steps"], total_target_steps - trained_steps)
        if chunk_steps <= 0:
            break

        train_env, eval_env, train_seed_offset, eval_seed_offset = build_env_pair(
            args_dict=args_dict,
            scenario_spec=scenario_spec,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
            refresh_round=refresh_round,
        )

        print(
            f"[refresh {refresh_round}] "
            f"train_seed_offset={train_seed_offset}, eval_seed_offset={eval_seed_offset}, "
            f"chunk_steps={chunk_steps}, trained_steps={trained_steps}/{total_target_steps}"
        )

        try:
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)
            eval_callback = EvalCallbackWithTimestamp(
                eval_env,
                callback_on_new_best=stop_callback,
                eval_freq=args_dict["eval_freq"],
                best_model_save_path=checkpoints_path,
                verbose=1,
                global_best_mean_reward=global_best_mean_reward,
            )
            callback_list = CallbackList([RewardLoggingCallback(), eval_callback])

            if model is None:
                model = PPO(
                    policy_spec.policy,
                    env=train_env,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    clip_range=0.05,
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
            else:
                model.set_env(train_env)

            # Keep clip_range globally linear across all refresh chunks.
            before_steps = model.num_timesteps
            model.clip_range = global_linear_schedule_for_chunk(
                initial_value=0.05,
                chunk_end_step=before_steps + chunk_steps,
                total_steps=total_target_steps,
            )

            model.learn(
                total_timesteps=chunk_steps,
                progress_bar=True,
                callback=callback_list,
                reset_num_timesteps=False,
            )
            after_steps = model.num_timesteps
            steps_gained = after_steps - before_steps
            global_best_mean_reward = max(global_best_mean_reward, eval_callback.best_mean_reward_prev)

            print(
                f"[refresh {refresh_round}] done: steps_gained={steps_gained}, "
                f"global_steps={after_steps}/{total_target_steps}, global_best_mean_reward={global_best_mean_reward:.4f}"
            )

            if steps_gained < chunk_steps:
                print("Training stopped early by callback.")
                stop_requested = True
        finally:
            train_env.close()
            eval_env.close()

        refresh_round += 1

    if model is None:
        raise RuntimeError("No model was created. Check --num_steps setting.")

    model.save(os.path.join(checkpoints_path, "final_model"))
    print(f"Final model saved to {os.path.join(checkpoints_path, 'final_model')}")


if __name__ == "__main__":
    main()
