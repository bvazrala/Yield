"""Train one DQN agent on highway-fast-v0 with a chosen reward configuration.

Each run is defined by a reward configuration name from configs/rewards.yaml
and a random seed. The step budget lives in the YAML file and is frozen for
the whole project. Use --steps only for smoke tests.

Usage:
    python train.py --config balanced --seed 0
    python train.py --config safety_heavy --seed 2 --steps 3000
"""

import argparse
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401  (import registers the highway environments)
import yaml

CONFIG_FILE = Path(__file__).parent / "configs" / "rewards.yaml"


def load_settings(config_name: str) -> tuple[dict, dict]:
    """Return (shared settings, reward overrides for the chosen configuration)."""
    with open(CONFIG_FILE) as f:
        data = yaml.safe_load(f)
    if config_name not in data["configurations"]:
        raise SystemExit(
            f"Unknown config '{config_name}'. Options: {list(data['configurations'])}"
        )
    return data, data["configurations"][config_name]


def make_env(settings: dict, rewards: dict):
    """Create the training environment with this run's reward weights."""
    return gymnasium.make(settings["env_id"], config=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Name from configs/rewards.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override the frozen step budget (smoke tests only)",
    )
    args = parser.parse_args()

    settings, rewards = load_settings(args.config)
    total_steps = args.steps or settings["total_steps"]
    run_name = f"{args.config}_seed{args.seed}"

    env = make_env(settings, rewards)

    # ------------------------------------------------------------------
    # TODO(team, week 1-2): build and train the agent. Keep it boring.
    #
    # 1. Copy DQN hyperparameters from the published highway-env DQN example
    #    (Farama-Foundation/HighwayEnv docs, "Getting Started") into a dict
    #    here, then FREEZE them for the whole project. No tuning.
    # 2. model = DQN("MlpPolicy", env, seed=args.seed,
    #                tensorboard_log=f"logs/{run_name}", **hyperparams)
    # 3. Attach a CheckpointCallback saving to models/checkpoints/{run_name}
    #    every ~10k steps so a crashed overnight run loses minutes, not hours.
    # 4. model.learn(total_timesteps=total_steps, progress_bar=True)
    # 5. model.save(f"models/{run_name}")
    # ------------------------------------------------------------------
    raise SystemExit(
        f"train.py is scaffolding: fill in the TODO block, then this command "
        f"will train {run_name} for {total_steps} steps."
    )


if __name__ == "__main__":
    main()
