"""Render short rollout videos of a saved agent into videos/.

The proposal calls for side-by-side clips of an aggressive agent and a
cautious one, so run this once with a speed_heavy model and once with a
safety_heavy model.

Usage:
    python record_videos.py --config speed_heavy --seed 0 --episodes 3
"""

import argparse
from pathlib import Path
import gymnasium as gym
import highway_env 
from stable_baselines3 import DQN
from train import load_settings

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_name = f"{args.config}_seed{args.seed}"
    
    settings, rewards = load_settings(args.config)
    base_env = gym.make(settings["env_id"], config=rewards, render_mode="rgb_array")
    
    env = gym.wrappers.RecordVideo(base_env, video_folder="videos", 
                    name_prefix=run_name, episode_trigger=lambda ep_id: True, 
                    disable_logger=True)
    

    model_path = Path("models") / f"{run_name}.zip"
    if not model_path.is_file():
        raise SystemExit(f"RecordVideos: {model_path} not found.")
    
    model = DQN.load(model_path, env=env)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()

    print(f"[{run_name}]: {args.episodes} videos finished rendering.")

if __name__ == "__main__":
    main()
