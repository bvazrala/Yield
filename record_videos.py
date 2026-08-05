"""Render short rollout videos of a saved agent into videos/.

The proposal calls for side-by-side clips of an aggressive agent and a
cautious one, so run this once with a speed_heavy model and once with a
safety_heavy model.

Usage:
    python record_videos.py --config speed_heavy --seed 0 --episodes 3
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_name = f"{args.config}_seed{args.seed}"

    # ------------------------------------------------------------------
    # TODO(team, week 5):
    # 1. Load the model as in evaluate.py.
    # 2. Rebuild the env with render_mode="rgb_array", then wrap it:
    #      env = gymnasium.wrappers.RecordVideo(
    #          env, video_folder="videos", name_prefix=run_name)
    # 3. Roll out --episodes episodes with deterministic actions.
    #
    # Rendering is for saved policies only. Never render during training.
    # ------------------------------------------------------------------
    raise SystemExit(f"record_videos.py is scaffolding: fill in the TODO block for {run_name}.")


if __name__ == "__main__":
    main()
