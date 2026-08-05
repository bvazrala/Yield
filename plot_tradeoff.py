"""Build the headline plot: average speed against collision rate, one point per
reward configuration, error bars across seeds.

Reads results/metrics.csv (written by evaluate.py) and saves
results/tradeoff.png. This is the week 3-4 starter plot. Refine labels,
annotations, and styling during the week 5 analysis pass.
"""

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df = pd.read_csv("results/metrics.csv")
    grouped = (
        df.groupby("config")
        .agg(
            collision_mean=("collision_rate", "mean"),
            collision_std=("collision_rate", "std"),
            speed_mean=("avg_speed", "mean"),
            speed_std=("avg_speed", "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        grouped["speed_mean"], grouped["collision_mean"],
        xerr=grouped["speed_std"], yerr=grouped["collision_std"],
        fmt="o", capsize=4,
    )
    for _, row in grouped.iterrows():
        ax.annotate(
            row["config"], (row["speed_mean"], row["collision_mean"]),
            textcoords="offset points", xytext=(6, 6),
        )
    ax.set_xlabel("Average speed")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety and performance tradeoff across reward configurations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/tradeoff.png", dpi=200)
    print("Wrote results/tradeoff.png")


if __name__ == "__main__":
    main()
