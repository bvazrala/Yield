"""Train one DQN agent on highway-fast-v0 with a chosen reward configuration.

Usage:
    python train.py --config balanced --seed 0
    python train.py --config safety_heavy --seed 2 --steps 3000

Outputs:
    models/<run>.zip                final agent
    models/<run>.json               run metadata
    models/checkpoints/<run>/       periodic checkpoints
    logs/<run>/                     TensorBoard events and monitor.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gymnasium
import highway_env  # noqa: F401  (registers the highway environments)
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "configs" / "rewards.yaml"

# Frozen for the whole study. The reward specification is the only variable.
HYPERPARAMS: dict[str, Any] = dict(
    policy="MlpPolicy",
    learning_rate=5e-4,
    buffer_size=15_000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)


def load_settings(config_name: str) -> tuple[dict, dict]:
    """Return shared settings and the environment overrides for one configuration."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    configurations = data["configurations"]
    if config_name not in configurations:
        raise SystemExit(f"Unknown config '{config_name}'. Options: {list(configurations)}")
    return data, configurations[config_name]


def make_env(settings: dict, overrides: dict, seed: int, monitor_path: Path):
    env_config = dict(settings.get("env_config", {}))
    env_config.update(overrides)
    env = gymnasium.make(settings["env_id"], config=env_config)
    env = Monitor(env, filename=str(monitor_path))
    env.reset(seed=seed)
    return env


def package_versions() -> dict:
    versions = {}
    for package in ("gymnasium", "highway-env", "stable-baselines3", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Name from configs/rewards.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Override the step budget (smoke tests only)")
    args = parser.parse_args()

    settings, overrides = load_settings(args.config)
    total_steps = args.steps or settings["total_steps"]
    run_name = f"{args.config}_seed{args.seed}"

    log_dir = ROOT / "logs" / run_name
    checkpoint_dir = ROOT / "models" / "checkpoints" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    env = make_env(settings, overrides, args.seed, log_dir / "monitor")

    model = DQN(env=env, seed=args.seed, tensorboard_log=str(log_dir), **HYPERPARAMS)

    checkpoint = CheckpointCallback(
        save_freq=max(total_steps // 5, 1),
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
    )

    print(f"[{run_name}] {total_steps:,} steps on {settings['env_id']} with {overrides}")

    # The progress bar is noise when stdout is piped to a log file.
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint,
        progress_bar=sys.stdout.isatty(),
    )

    model.save(ROOT / "models" / run_name)

    metadata = {
        "run_name": run_name,
        "config": args.config,
        "seed": args.seed,
        "env_id": settings["env_id"],
        "total_steps": total_steps,
        "is_smoke_test": args.steps is not None,
        "env_overrides": overrides,
        "hyperparameters": {k: v for k, v in HYPERPARAMS.items() if k != "policy_kwargs"},
        "net_arch": HYPERPARAMS["policy_kwargs"]["net_arch"],
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }
    (ROOT / "models" / f"{run_name}.json").write_text(json.dumps(metadata, indent=2))

    env.close()
    print(f"[{run_name}] saved models/{run_name}.zip")
    print(f"[{run_name}] next: python evaluate.py --config {args.config} --seed {args.seed}")


if __name__ == "__main__":
    main()
    