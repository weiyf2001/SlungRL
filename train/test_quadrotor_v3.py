import argparse
import os
import sys

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "envs"))

from envs.ppo.ppo import PPO
from envs.quadrotor_payload_env_v2 import QuadrotorPayloadEnvV2
from train.policy_v3 import CustomActorCriticPolicyV3
from train.run_paths import get_checkpoint_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate PPO v3 with an LSTM feature extractor.")
    parser.add_argument("--id", type=str, default="untitled", help="Provide experiment name and ID.")
    parser.add_argument("--visualize", type=bool, default=False, help="Choose visualization option.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    return vars(parser.parse_args())


def main():
    args_dict = parse_arguments()
    print(args_dict)

    model_path = get_checkpoint_path("v3", args_dict["id"])
    loaded_model = PPO.load(model_path, custom_objects={"policy_class": CustomActorCriticPolicyV3})

    render_mode = "human" if args_dict["visualize"] else None
    env = VecMonitor(DummyVecEnv([lambda: QuadrotorPayloadEnvV2(render_mode=render_mode)]))

    print("Evaluation start!")
    mean_reward, std_reward = evaluate_policy(
        loaded_model,
        env,
        n_eval_episodes=args_dict["episodes"],
        render=bool(render_mode),
    )
    env.close()
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    main()
