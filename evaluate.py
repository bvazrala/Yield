"""Evaluate one saved agent over held-out episodes and append a row to results/metrics.csv.

Metrics follow proposal section 4: collision rate, average speed, mean return,
mean episode length, lane changes per episode, average following distance.

Usage:
    python evaluate.py --config balanced --seed 0
"""

import argparse
import csv
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401

RESULTS_FILE = Path("results/metrics.csv")
FIELDS = [
    "config", "seed", "episodes",
    "collision_rate", "avg_speed", "mean_return", "mean_length",
    "lane_changes_per_episode", "avg_following_distance",
]

# Evaluation episodes use a seed range the agents never saw during training,
# and the SAME range for every agent, so all agents face identical traffic.
EVAL_SEED_BASE = 10_000


def append_row(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    run_name = f"{args.config}_seed{args.seed}"
    model_path = Path(f"models/{run_name}.zip")

    # ------------------------------------------------------------------
    # TODO(team, week 2): implement the evaluation loop.
    #
    # 1. model = DQN.load(model_path)
    # 2. Rebuild the env exactly as in train.py (same reward config). For
    #    episode i, call env.reset(seed=EVAL_SEED_BASE + i) so every agent
    #    sees the same traffic situations.
    # 3. Step with model.predict(obs, deterministic=True) until the episode
    #    ends. Useful signals along the way:
    #      - info["crashed"] says whether the episode ended in a collision
    #      - env.unwrapped.vehicle exposes the ego car (speed, lane_index)
    #    Following distance is the stretch metric. Skip it at first and
    #    backfill the column later if time allows.
    # 4. Aggregate per-episode numbers into the FIELDS above and call
    #    append_row({...}).
    # ------------------------------------------------------------------
    raise SystemExit(
        f"evaluate.py is scaffolding: fill in the TODO block, then this will "
        f"evaluate {run_name} over {args.episodes} episodes."
    )


if __name__ == "__main__":
    main()
