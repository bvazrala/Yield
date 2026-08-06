"""Evaluate one saved agent over held-out episodes and append a row to results/metrics.csv.

Metrics follow proposal section 4: collision rate, average speed, mean return,
mean episode length, lane changes per episode, average following distance.

Usage:
    python evaluate.py --config balanced --seed 0
    python evaluate.py --config balanced --seed 0 --episodes 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from tqdm import trange

from train import load_settings, make_env

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "metrics.csv"
FIELDS = [
    "config",
    "seed",
    "episodes",
    "collision_rate",
    "avg_speed",
    "mean_return",
    "mean_length",
    "lane_changes_per_episode",
    "avg_following_distance",
]

# Evaluation episodes use a seed range the agents never saw during training,
# and the SAME range for every agent, so all agents face identical traffic.
EVAL_SEED_BASE = 10_000


def append_row(row: dict) -> None:
    """Append one result row, creating the CSV and header when necessary."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def following_distance(env) -> float | None:
    """Return the current bumper-to-bumper gap to the nearest car ahead.
    Only a front vehicle in the ego vehicle's current lane is considered.
    ``lane_distance_to`` measures center-to-center longitudinal distance, so
    half of both vehicle lengths is removed to obtain the physical gap.
    ``None`` means there is currently no vehicle ahead in that lane.
    """
    base_env = env.unwrapped
    ego_vehicle = base_env.vehicle
    road = base_env.road

    if ego_vehicle is None or road is None:
        return None

    front_vehicle, _ = road.neighbour_vehicles(ego_vehicle)
    if front_vehicle is None:
        return None

    center_distance = float(ego_vehicle.lane_distance_to(front_vehicle))
    vehicle_lengths = 0.5 * float(ego_vehicle.LENGTH + front_vehicle.LENGTH)
    return max(0.0, center_distance - vehicle_lengths)

def evaluate_agent(model: DQN, env, episodes: int) -> dict[str, float]:
    """Run deterministic held-out rollouts and return aggregate metrics.
    Speed and following distance are averaged within each episode first and
    then across episodes. This keeps long non-collision episodes from receiving
    more weight solely because they contain more simulator steps.
    """
    collisions: list[float] = []
    average_speeds: list[float] = []
    returns: list[float] = []
    lengths: list[float] = []
    lane_changes: list[float] = []
    average_following_distances: list[float] = []

    for episode in trange(episodes, desc="Evaluating", unit="episode"):
        observation, _ = env.reset(seed=EVAL_SEED_BASE + episode)
        previous_lane_index = env.unwrapped.vehicle.lane_index

        episode_return = 0.0
        episode_length = 0
        episode_lane_changes = 0
        episode_crashed = False
        speed_samples: list[float] = []
        distance_samples: list[float] = []

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            ego_vehicle = env.unwrapped.vehicle
            episode_return += float(reward)
            episode_length += 1
            episode_crashed = episode_crashed or bool(
                info.get("crashed", ego_vehicle.crashed))
            speed_samples.append(float(info.get("speed", ego_vehicle.speed)))

            current_lane_index = ego_vehicle.lane_index
            if current_lane_index != previous_lane_index:
                episode_lane_changes += 1
                previous_lane_index = current_lane_index

            distance = following_distance(env)
            if distance is not None and np.isfinite(distance):
                distance_samples.append(distance)

        collisions.append(float(episode_crashed))
        average_speeds.append(fmean(speed_samples) if speed_samples else 0.0)
        returns.append(episode_return)
        lengths.append(float(episode_length))
        lane_changes.append(float(episode_lane_changes))
        if distance_samples:
            average_following_distances.append(fmean(distance_samples))

    return {
        "collision_rate": fmean(collisions),
        "avg_speed": fmean(average_speeds),
        "mean_return": fmean(returns),
        "mean_length": fmean(lengths),
        "lane_changes_per_episode": fmean(lane_changes),
        "avg_following_distance": (
            fmean(average_following_distances)
            if average_following_distances
            else float("nan")
        ),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Reward configuration name from configs/rewards.yaml",
        )
    parser.add_argument("--seed", type=int, required=True), help="Training seed"
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of held-out episodes (default: eval_episodes from YAML)",
    )
    args = parser.parse_args()

    settings, rewards = load_settings(args.config)
    episodes = (
        int(settings["eval_episodes"])
        if args.episodes is None
        else args.episodes
    )
    if episodes <= 0:
        parser.error("--episodes must be greater than zero")

    run_name = f"{args.config}_seed{args.seed}"
    model_path = ROOT / "models" / f"{run_name}.zip"
    if not model_path.is_file():
        raise SystemExit(
            f"Saved model not found: {model_path}\n"
            f"Train it first with: python train.py --config {args.config} "
            f"--seed {args.seed}"
        )

    env = make_env(settings, rewards)
    try:
        model = DQN.load(model_path, env=env)
        metrics = evaluate_agent(model, env, episodes)
    finally:
        env.close()

    row: dict[str, Any] = {
        "config": args.config,
        "seed": args.seed,
        "episodes": episodes,
        **{name: round(value, 6) for name, value in metrics.items()},
    }
    append_row(row)

    print(f"\nEvaluation complete: {run_name}")
    for field in FIELDS[3:]:
        print(f"  {field}: {row[field]}")
    print(f"Appended results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
