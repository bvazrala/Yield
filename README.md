# Learning to Drive in Traffic

Measuring the safety and performance tradeoff in reinforcement learning.

CS 175: Project in AI, Summer 2026, Group 7.

| Team member | UCI NetID |

| Bala Kausik Vazrala | bvazrala |
| Christopher James Cho | cjcho2 |
| Harrison Nguyen Xuan Hiep Le | harrisnl |

## Overview

We train DQN agents in the highway-env simulator where the only difference between agents is how heavily the reward penalizes collisions relative to rewarding speed. Five reward configurations, three random seeds each, and a frozen budget of 50,000 training steps per run. The headline result is a tradeoff curve of average speed against collision rate. Full details are in [insert google doc link later].

## Repository structure

```
configs/rewards.yaml     The five reward configurations and shared settings
train.py                 Train one agent (one config, one seed)
evaluate.py              Evaluate a saved agent over 200 held-out episodes
validate_cartpole.py     Week 1 sanity check for the whole RL stack
plot_tradeoff.py         Build the speed vs collision rate plot from results
record_videos.py         Render rollout videos of saved agents
scripts/run_matrix.sh    Queue the full 5 x 3 experiment matrix sequentially
docs/                    Project proposal
models/                  Saved models and checkpoints (git ignored)
logs/                    Training logs and TensorBoard events (git ignored)
videos/                  Rendered rollouts (git ignored)
results/                 Metrics CSV and plots (tracked in git)
```

## Setup

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Once the install works on all three laptops, freeze exact versions so every machine trains on identical code:

```
pip freeze > requirements.lock.txt
```

## Order of operations

1. `python validate_cartpole.py` must pass before anything else. If CartPole trains, every later problem is a highway problem, not an install problem.
2. Fill in the TODO blocks in `train.py` and `evaluate.py`. This is the week 1-2 engineering work.
3. Smoke test one run end to end: `python train.py --config balanced --seed 0 --steps 3000`.
4. Full matrix: `bash scripts/run_matrix.sh`, or `bash scripts/run_matrix.sh 1` to run only seed 1 on this machine.
5. `python evaluate.py --config <name> --seed <n>` for each finished run, then `python plot_tradeoff.py`.

## Overnight runs

Start inside tmux so the run survives a closed terminal:

```
tmux new -s highway
bash scripts/run_matrix.sh
```

Detach with Ctrl-b then d. Reattach in the morning with `tmux attach -t highway`. Live curves: `tensorboard --logdir logs`.

tmux does not survive a sleeping machine. On macOS run the script under `caffeinate -i` and leave the lid open. On Windows set the power plan to never sleep while plugged in and pause updates. On Linux disable automatic suspend. Watch the first run for ten minutes before bed: episodes completing, rewards logging, a checkpoint appearing on disk.

## Milestones

| Week | Milestone |
|---|---|
| 1 | Basic design: installs, CartPole validation, step budget frozen |
| 2 | Alpha: baselines measured, default reward agent, automated evaluation |
| 3-4 | Beta: full 5 x 3 matrix through overnight runs, first tradeoff plot |
| 5 | Analysis: stress tests, transfer experiments, rollout videos |
| 6 | Final: showcase run on highway-v0, report, buffer for reruns |

## References

1. E. Leurent. An Environment for Autonomous Driving Decision-Making. GitHub repository, 2018. https://github.com/eleurent/highway-env
2. A. Raffin et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268), 2021.
3. M. Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032, 2024.
4. V. Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 2015.
