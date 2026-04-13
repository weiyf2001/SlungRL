import argparse
import copy
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from train.experiments.registry import get_policy_spec, get_scenario_spec


def parse_arguments():
    default_checkpoint = (
        "/home/ubuntu/weiyf/roverfly/runs/experiments/s0_full_state/"
        "mlp_memoryless/big_distur4/checkpoints/best_model.zip"
    )
    default_output_dir = "/home/ubuntu/weiyf/roverfly/runs/eval_plots/big_distur4"

    parser = argparse.ArgumentParser(
        description="Load a trained checkpoint and run MuJoCo evaluation with optional visualization."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="Checkpoint file path or stem. Prefer .zip; .pt input is auto-mapped to same-name .zip first.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario key override (auto-inferred from checkpoint path when omitted).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy key override (auto-inferred from checkpoint path when omitted).",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render MuJoCo window during rollout.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device for model loading.")
    parser.add_argument("--seed", type=int, default=66, help="Environment seed (env_num).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory for exported xP/action figures and CSV files.",
    )
    parser.add_argument(
        "--record-xp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable payload trajectory recording and plot/csv export on episode end.",
    )
    parser.add_argument(
        "--record-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable action recording and plot/csv export on episode end.",
    )
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save MuJoCo rollout GIF for each episode to output directory.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=25,
        help="FPS used when encoding GIF files.",
    )
    parser.add_argument(
        "--gif-stride",
        type=int,
        default=4,
        help="Store one frame every N steps when building GIF.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(raw_path: str) -> tuple[str, bool]:
    path_obj = Path(raw_path).expanduser()
    candidates = []

    if path_obj.is_dir():
        candidates.extend(
            [
                path_obj / "best_model.zip",
                path_obj / "best_model",
                path_obj / "best_model.pt",
            ]
        )
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

    raise FileNotFoundError(
        f"Checkpoint not found. Tried: {[str(c) for c in candidates]}"
    )


def infer_scenario_policy_from_path(checkpoint_path: str) -> tuple[str | None, str | None]:
    parts = Path(checkpoint_path).parts
    for idx, token in enumerate(parts):
        if token == "experiments" and idx + 2 < len(parts):
            return parts[idx + 1], parts[idx + 2]
    return None, None


def export_episode_artifacts(
    env,
    output_dir: str,
    episode_idx: int,
    file_tag: str,
    xp_records: list[np.ndarray],
    action_records: list[np.ndarray],
    record_xp: bool,
    record_action: bool,
) -> None:
    if not record_xp and not record_action:
        return

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, file_tag)
    max_steps = int(getattr(env, "max_timesteps", 0))

    if record_xp:
        if len(xp_records) == 0:
            print(f"[episode {episode_idx}] skip xP export: empty trajectory.")
        else:
            xp_array = np.asarray(xp_records, dtype=np.float32)
            if max_steps > 0:
                xp_array = xp_array[:max_steps]
            env.xP_record = xp_array
            env.plot_xP(save_path=f"{prefix}_xP_xyz.png")
            env.plot_xP_3d(save_path=f"{prefix}_xP_3d.png")
            env.save_xP_csv(save_path=f"{prefix}_xP.csv")

    if record_action:
        if len(action_records) == 0:
            print(f"[episode {episode_idx}] skip action export: empty trajectory.")
        else:
            action_array = np.asarray(action_records, dtype=np.float32)
            if max_steps > 0:
                action_array = action_array[:max_steps]
            env.action_record = action_array
            env.plot_action(save_path=f"{prefix}_action.png")
            env.save_action_csv(save_path=f"{prefix}_action.csv")


def export_episode_gif(
    output_dir: str,
    episode_idx: int,
    file_tag: str,
    frames: list[np.ndarray],
    gif_fps: int,
) -> None:
    if len(frames) == 0:
        print(f"[episode {episode_idx}] skip gif export: empty frames.")
        return
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"{file_tag}_sim.gif")
    imageio.mimsave(gif_path, frames, fps=max(1, gif_fps))
    print(f"[Saved] MuJoCo GIF -> {gif_path}")


def main():
    args = parse_arguments()

    checkpoint_path, used_fallback = resolve_checkpoint_path(args.checkpoint)
    inferred_scenario, inferred_policy = infer_scenario_policy_from_path(checkpoint_path)
    scenario_name = args.scenario or inferred_scenario
    policy_name = args.policy or inferred_policy

    if scenario_name is None or policy_name is None:
        raise ValueError(
            "Unable to infer scenario/policy from checkpoint path. "
            "Please provide --scenario and --policy explicitly."
        )

    scenario_spec = get_scenario_spec(scenario_name)
    policy_spec = get_policy_spec(policy_name)

    render_mode = "human" if args.visualize else None
    env_kwargs = copy.deepcopy(scenario_spec.default_kwargs)
    env_kwargs.update(copy.deepcopy(policy_spec.env_kwargs))

    def _make_env():
        env = scenario_spec.env_class(
            render_mode=render_mode,
            env_num=args.seed,
            **env_kwargs,
        )
        env.is_record_xP = bool(args.record_xp)
        env.is_record_action = bool(args.record_action)
        return env

    print(
        {
            "checkpoint_input": args.checkpoint,
            "checkpoint_resolved": checkpoint_path,
            "checkpoint_fallback_used": used_fallback,
            "scenario": scenario_name,
            "policy": policy_name,
            "episodes": args.episodes,
            "visualize": args.visualize,
            "deterministic": args.deterministic,
            "record_xp": args.record_xp,
            "record_action": args.record_action,
            "save_gif": args.save_gif,
            "gif_fps": args.gif_fps,
            "gif_stride": args.gif_stride,
            "output_dir": args.output_dir,
            "device": args.device,
            "seed": args.seed,
        }
    )

    model = PPO.load(checkpoint_path, device=args.device)
    env = _make_env()
    # Script-level recording avoids missing plots on early-terminated episodes.
    env.is_record_xP = False
    env.is_record_action = False

    episode_rewards = []
    print("Evaluation start...")
    for episode_idx in range(1, args.episodes + 1):
        file_tag = f"seed_{int(args.seed):03d}_ep_{episode_idx:03d}"
        obs, _info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        xp_records = []
        action_records = []
        gif_frames = []

        if args.save_gif:
            frame0 = env.mujoco_renderer.render("rgb_array")
            gif_frames.append(frame0)

        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1

            if args.record_xp:
                xp_records.append(np.copy(env.data.qpos[7:10]))
            if args.record_action:
                action_records.append(np.asarray(action, dtype=np.float32).copy())
            if args.save_gif and (episode_steps % max(1, args.gif_stride) == 0):
                frame = env.mujoco_renderer.render("rgb_array")
                gif_frames.append(frame)

        export_episode_artifacts(
            env=env,
            output_dir=args.output_dir,
            episode_idx=episode_idx,
            file_tag=file_tag,
            xp_records=xp_records,
            action_records=action_records,
            record_xp=args.record_xp,
            record_action=args.record_action,
        )
        if args.save_gif:
            export_episode_gif(
                output_dir=args.output_dir,
                episode_idx=episode_idx,
                file_tag=file_tag,
                frames=gif_frames,
                gif_fps=args.gif_fps,
            )

        episode_rewards.append(episode_reward)
        print(
            f"[episode {episode_idx}/{args.episodes}] "
            f"reward={episode_reward:.4f}, steps={episode_steps}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    env.close()
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    print(f"Evaluation done. Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    main()
