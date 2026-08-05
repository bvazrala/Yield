"""Week 1 sanity check: the RL stack must solve CartPole before we touch highway-env.

If this finishes with mean reward at or above 400, the pipeline (gymnasium,
stable-baselines3, torch, tensorboard) works, and every later problem is a
highway problem rather than an install problem.

Hyperparameters come from rl-baselines3-zoo's tuned CartPole DQN settings.
Takes a few minutes on a laptop CPU.
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def main() -> None:
    env = gym.make("CartPole-v1")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=2.3e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=256,
        gradient_steps=128,
        target_update_interval=10,
        exploration_fraction=0.16,
        exploration_final_eps=0.04,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="logs/cartpole",
        verbose=1,
    )
    model.learn(total_timesteps=50_000, progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"\nCartPole evaluation: {mean_reward:.1f} +/- {std_reward:.1f} over 20 episodes")
    if mean_reward >= 400:
        print("PASS: the training pipeline works. Move on to highway-env.")
    else:
        print(
            "NOT THERE YET: expected mean reward >= 400. "
            "Check package versions before touching highway-env."
        )


if __name__ == "__main__":
    main()
