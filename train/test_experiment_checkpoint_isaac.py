import argparse
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from train.experiments.registry import get_policy_spec, get_scenario_spec


def parse_arguments() -> argparse.Namespace:
    default_checkpoint = (
        "/home/ubuntu/weiyf/roverfly/runs/experiments/s0_full_state/"
        "mlp_memoryless/big_distur4/checkpoints/best_model.zip"
    )
    default_output_dir = "/home/ubuntu/weiyf/roverfly/runs/eval_plots/isaac_eval"

    parser = argparse.ArgumentParser(
        description="Load a checkpoint and evaluate in Isaac Sim / Isaac Lab."
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Checkpoint path (.zip preferred).")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Isaac Gymnasium task id, e.g. Isaac-Reach-Franka-v0 or your custom task id.",
    )
    parser.add_argument(
        "--cfg-entry-point",
        type=str,
        default="env_cfg_entry_point",
        help="Config entry point used by load_cfg_from_registry.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario key to preload custom modules used by checkpoint.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional policy key to preload custom modules used by checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollout episodes.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Hard step cap per episode.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch Isaac app headless.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live viewport (typically requires --no-headless).",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["human", "rgb_array", "depth_array", "none"],
        help="Explicit render mode override. If omitted, mode is inferred.",
    )
    parser.add_argument(
        "--enable-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable camera rendering in Isaac app launcher.",
    )
    parser.add_argument(
        "--use-sb3-wrapper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap env with Isaac Lab SB3 VecEnv wrapper.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device for model loading.")
    parser.add_argument("--seed", type=int, default=66, help="Environment seed.")
    parser.add_argument(
        "--obs-key",
        type=str,
        default="policy",
        help="When observation is dict, which key is fed to policy and used for logging.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory for rollout artifacts (csv/plot/gif).",
    )
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout GIF for each episode.",
    )
    parser.add_argument("--gif-fps", type=int, default=25, help="GIF frame rate.")
    parser.add_argument("--gif-stride", type=int, default=4, help="Store one frame every N steps.")
    parser.add_argument(
        "--record-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save observation trajectory csv.",
    )
    parser.add_argument(
        "--record-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save action trajectory csv + plot.",
    )
    parser.add_argument(
        "--xp-obs-slice",
        type=str,
        default=None,
        help='Optional payload position slice on observation, e.g. "7:10".',
    )
    return parser.parse_args()


def resolve_checkpoint_path(raw_path: str) -> tuple[str, bool]:
    path_obj = Path(raw_path).expanduser()
    candidates: list[Path] = []

    if path_obj.is_dir():
        candidates.extend([path_obj / "best_model.zip", path_obj / "best_model", path_obj / "best_model.pt"])
    else:
        suffix = path_obj.suffix.lower()
        if suffix == "":
            candidates.extend([Path(f"{path_obj}.zip"), path_obj, Path(f"{path_obj}.pt")])
        elif suffix == ".pt":
            candidates.extend([path_obj.with_suffix(".zip"), path_obj])
        elif suffix == ".zip":
            candidates.extend([path_obj, path_obj.with_suffix(".pt")])
        else:
            candidates.append(path_obj)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve()), str(candidate) != str(path_obj)

    raise FileNotFoundError(f"Checkpoint not found. Tried: {[str(c) for c in candidates]}")


def infer_scenario_policy_from_path(checkpoint_path: str) -> tuple[str | None, str | None]:
    parts = Path(checkpoint_path).parts
    for idx, token in enumerate(parts):
        if token == "experiments" and idx + 2 < len(parts):
            return parts[idx + 1], parts[idx + 2]
    return None, None


def parse_slice(slice_expr: str | None) -> slice | None:
    if slice_expr is None:
        return None
    chunks = slice_expr.split(":")
    if len(chunks) != 2:
        raise ValueError(f'Invalid --xp-obs-slice "{slice_expr}". Use start:end, e.g. 7:10.')
    start = int(chunks[0]) if chunks[0] != "" else None
    end = int(chunks[1]) if chunks[1] != "" else None
    return slice(start, end)


def load_isaac_runtime(headless: bool, enable_cameras: bool) -> tuple[Any, str]:
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
        return app_launcher.app, "isaaclab"
    except ImportError:
        from omni.isaac.lab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
        return app_launcher.app, "omni.isaac.lab"


def import_isaac_helpers(runtime_flavor: str):
    if runtime_flavor == "isaaclab":
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import load_cfg_from_registry

        try:
            from isaaclab_rl.sb3 import Sb3VecEnvWrapper
        except ImportError:
            Sb3VecEnvWrapper = None
    else:
        import omni.isaac.lab_tasks  # noqa: F401
        from omni.isaac.lab_tasks.utils import load_cfg_from_registry

        try:
            from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
        except ImportError:
            Sb3VecEnvWrapper = None

    import gymnasium as gym

    return gym, load_cfg_from_registry, Sb3VecEnvWrapper


def infer_render_mode(args: argparse.Namespace) -> str | None:
    if args.render_mode == "none":
        return None
    if args.render_mode is not None:
        return args.render_mode
    if args.visualize:
        return "human"
    if args.save_gif:
        return "rgb_array"
    return None


def make_isaac_env(
    gym_module,
    load_cfg_from_registry,
    sb3_wrapper_cls,
    args: argparse.Namespace,
    render_mode: str | None,
):
    cfg = None
    try:
        cfg = load_cfg_from_registry(args.task, args.cfg_entry_point)
    except Exception as exc:
        print(
            f"[warn] load_cfg_from_registry(task={args.task}, entry={args.cfg_entry_point}) failed: {exc}. "
            f"Falling back to gym.make without cfg."
        )

    env_kwargs = {}
    if cfg is not None:
        env_kwargs["cfg"] = cfg
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym_module.make(args.task, **env_kwargs)
    if args.use_sb3_wrapper:
        if sb3_wrapper_cls is None:
            raise ImportError("SB3 wrapper requested but not importable in current Isaac Lab version.")
        env = sb3_wrapper_cls(env)
    return env


def try_reset(env, seed: int):
    try:
        reset_out = env.reset(seed=seed)
    except TypeError:
        reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0], reset_out[1]
    return reset_out, {}


def normalize_obs(obs: Any, obs_key: str) -> np.ndarray:
    if isinstance(obs, dict):
        if obs_key in obs:
            arr = obs[obs_key]
        elif "policy" in obs:
            arr = obs["policy"]
        elif len(obs) == 1:
            arr = next(iter(obs.values()))
        else:
            raise KeyError(f"Observation is dict with keys {list(obs.keys())}, missing --obs-key={obs_key}.")
    else:
        arr = obs
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim > 1 and arr_np.shape[0] == 1:
        return arr_np[0]
    return arr_np.reshape(-1)


def normalize_action(action: Any) -> np.ndarray:
    act = np.asarray(action, dtype=np.float32)
    if act.ndim > 1 and act.shape[0] == 1:
        return act[0]
    return act.reshape(-1)


def parse_step_output(step_out: Any):
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"Unexpected step output type: {type(step_out)}")

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        reward_scalar = float(np.mean(np.asarray(reward)))
        terminated_flag = bool(np.any(np.asarray(terminated)))
        truncated_flag = bool(np.any(np.asarray(truncated)))
        return obs, reward_scalar, terminated_flag, truncated_flag, info

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        reward_scalar = float(np.mean(np.asarray(reward)))
        done_flag = bool(np.any(np.asarray(done)))
        truncated_flag = False
        if isinstance(info, dict):
            trunc = info.get("TimeLimit.truncated", False)
            truncated_flag = bool(np.any(np.asarray(trunc)))
        return obs, reward_scalar, done_flag, truncated_flag, info

    raise RuntimeError(f"Unexpected step output length: {len(step_out)}")


def render_rgb_frame(env) -> np.ndarray | None:
    try:
        frame = env.render()
    except Exception:
        frame = None
    if frame is None:
        return None
    frame_np = np.asarray(frame)
    if frame_np.ndim == 3:
        return frame_np
    return None


def export_gif(output_dir: str, file_tag: str, frames: list[np.ndarray], fps: int) -> None:
    if len(frames) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"{file_tag}_sim.gif")
    imageio.mimsave(gif_path, frames, fps=max(1, int(fps)))
    print(f"[Saved] Isaac GIF -> {gif_path}")


def export_rollout_csv(
    output_dir: str,
    file_tag: str,
    obs_records: list[np.ndarray],
    action_records: list[np.ndarray],
    rewards: list[float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if len(obs_records) > 0:
        obs_arr = np.asarray(obs_records, dtype=np.float32)
        obs_df = pd.DataFrame(obs_arr, columns=[f"obs_{i}" for i in range(obs_arr.shape[1])])
        obs_df.insert(0, "step", np.arange(len(obs_df), dtype=np.int32))
        obs_path = os.path.join(output_dir, f"{file_tag}_obs.csv")
        obs_df.to_csv(obs_path, index=False)
        print(f"[Saved] Obs CSV -> {obs_path}")

    if len(action_records) > 0:
        act_arr = np.asarray(action_records, dtype=np.float32)
        act_df = pd.DataFrame(act_arr, columns=[f"action_{i}" for i in range(act_arr.shape[1])])
        act_df.insert(0, "step", np.arange(len(act_df), dtype=np.int32))
        act_path = os.path.join(output_dir, f"{file_tag}_action.csv")
        act_df.to_csv(act_path, index=False)
        print(f"[Saved] Action CSV -> {act_path}")

    if len(rewards) > 0:
        rew_arr = np.asarray(rewards, dtype=np.float32)
        rew_df = pd.DataFrame({"step": np.arange(len(rew_arr), dtype=np.int32), "reward": rew_arr})
        rew_path = os.path.join(output_dir, f"{file_tag}_reward.csv")
        rew_df.to_csv(rew_path, index=False)
        print(f"[Saved] Reward CSV -> {rew_path}")


def export_action_plot(output_dir: str, file_tag: str, action_records: list[np.ndarray]) -> None:
    if len(action_records) == 0:
        return
    action_arr = np.asarray(action_records, dtype=np.float32)
    t = np.arange(action_arr.shape[0], dtype=np.int32)
    plt.figure(figsize=(10, 4))
    for idx in range(action_arr.shape[1]):
        plt.plot(t, action_arr[:, idx], label=f"a{idx}")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.title("Action Trajectory")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{file_tag}_action.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Action plot -> {out_path}")


def export_xp_from_obs(
    output_dir: str,
    file_tag: str,
    obs_records: list[np.ndarray],
    obs_slice: slice | None,
) -> None:
    if obs_slice is None or len(obs_records) == 0:
        return
    obs_arr = np.asarray(obs_records, dtype=np.float32)
    xp_arr = obs_arr[:, obs_slice]
    if xp_arr.ndim != 2 or xp_arr.shape[1] != 3:
        print(
            f"[warn] skip xP export: obs slice {obs_slice} produces shape {xp_arr.shape}, expected (*, 3)."
        )
        return

    t = np.arange(xp_arr.shape[0], dtype=np.int32)
    xp_df = pd.DataFrame({"step": t, "xP_x": xp_arr[:, 0], "xP_y": xp_arr[:, 1], "xP_z": xp_arr[:, 2]})
    xp_csv = os.path.join(output_dir, f"{file_tag}_xP.csv")
    xp_df.to_csv(xp_csv, index=False)
    print(f"[Saved] xP CSV -> {xp_csv}")

    plt.figure(figsize=(10, 4))
    plt.plot(t, xp_arr[:, 0], label="x")
    plt.plot(t, xp_arr[:, 1], label="y")
    plt.plot(t, xp_arr[:, 2], label="z")
    plt.xlabel("Step")
    plt.ylabel("xP")
    plt.title("Payload Position from Observation Slice")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    xp_png = os.path.join(output_dir, f"{file_tag}_xP_xyz.png")
    plt.savefig(xp_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Saved] xP plot -> {xp_png}")


def preload_registry_modules(scenario_name: str | None, policy_name: str | None) -> None:
    if scenario_name is None or policy_name is None:
        return
    scenario_spec = get_scenario_spec(scenario_name)
    policy_spec = get_policy_spec(policy_name)
    print(
        {
            "scenario": scenario_name,
            "scenario_description": scenario_spec.description,
            "policy": policy_name,
            "policy_description": policy_spec.description,
        }
    )


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path, used_fallback = resolve_checkpoint_path(args.checkpoint)
    inferred_scenario, inferred_policy = infer_scenario_policy_from_path(checkpoint_path)
    scenario_name = args.scenario or inferred_scenario
    policy_name = args.policy or inferred_policy
    preload_registry_modules(scenario_name, policy_name)

    render_mode = infer_render_mode(args)
    xp_slice = parse_slice(args.xp_obs_slice)
    enable_cameras = bool(args.enable_cameras or args.save_gif or render_mode in {"rgb_array", "depth_array"})

    print(
        {
            "checkpoint_input": args.checkpoint,
            "checkpoint_resolved": checkpoint_path,
            "checkpoint_fallback_used": used_fallback,
            "task": args.task,
            "cfg_entry_point": args.cfg_entry_point,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "headless": args.headless,
            "visualize": args.visualize,
            "render_mode": render_mode,
            "use_sb3_wrapper": args.use_sb3_wrapper,
            "deterministic": args.deterministic,
            "device": args.device,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "save_gif": args.save_gif,
            "gif_fps": args.gif_fps,
            "gif_stride": args.gif_stride,
            "obs_key": args.obs_key,
            "xp_obs_slice": args.xp_obs_slice,
        }
    )

    simulation_app = None
    env = None
    try:
        simulation_app, runtime_flavor = load_isaac_runtime(headless=args.headless, enable_cameras=enable_cameras)
        gym_module, load_cfg_from_registry, sb3_wrapper_cls = import_isaac_helpers(runtime_flavor)
        env = make_isaac_env(
            gym_module=gym_module,
            load_cfg_from_registry=load_cfg_from_registry,
            sb3_wrapper_cls=sb3_wrapper_cls,
            args=args,
            render_mode=render_mode,
        )

        model = PPO.load(checkpoint_path, device=args.device)

        episode_rewards = []
        print("Isaac evaluation start...")
        for episode_idx in range(1, args.episodes + 1):
            file_tag = f"seed_{int(args.seed):03d}_ep_{episode_idx:03d}"
            obs, _info = try_reset(env, seed=args.seed + episode_idx - 1)

            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            episode_save_gif = bool(args.save_gif)
            obs_records: list[np.ndarray] = []
            action_records: list[np.ndarray] = []
            reward_records: list[float] = []
            gif_frames: list[np.ndarray] = []

            if episode_save_gif:
                frame0 = render_rgb_frame(env)
                if frame0 is not None:
                    gif_frames.append(frame0)
                else:
                    print("[warn] render returned no RGB frame; gif capture disabled for this run.")
                    episode_save_gif = False

            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=args.deterministic)
                step_out = env.step(action)
                obs, reward, terminated, truncated, info = parse_step_output(step_out)
                episode_reward += reward
                episode_steps += 1

                if args.record_obs or xp_slice is not None:
                    obs_records.append(normalize_obs(obs, args.obs_key))
                if args.record_action:
                    action_records.append(normalize_action(action))
                reward_records.append(reward)

                if episode_save_gif and episode_steps % max(1, args.gif_stride) == 0:
                    frame = render_rgb_frame(env)
                    if frame is not None:
                        gif_frames.append(frame)

                if episode_steps >= args.max_steps:
                    truncated = True

            export_rollout_csv(
                output_dir=args.output_dir,
                file_tag=file_tag,
                obs_records=obs_records if args.record_obs else [],
                action_records=action_records if args.record_action else [],
                rewards=reward_records,
            )
            if args.record_action:
                export_action_plot(args.output_dir, file_tag, action_records)
            export_xp_from_obs(args.output_dir, file_tag, obs_records, xp_slice)
            if episode_save_gif:
                export_gif(args.output_dir, file_tag, gif_frames, args.gif_fps)

            episode_rewards.append(episode_reward)
            print(
                f"[episode {episode_idx}/{args.episodes}] "
                f"reward={episode_reward:.4f}, steps={episode_steps}, "
                f"terminated={terminated}, truncated={truncated}"
            )

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
        print(f"Evaluation done. Mean reward: {mean_reward:.6f}, Std reward: {std_reward:.6f}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
