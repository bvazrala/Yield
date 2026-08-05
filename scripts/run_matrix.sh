#!/usr/bin/env bash
# Queue the full experiment matrix sequentially: 5 reward configs x 3 seeds.
#
# Run inside tmux so it survives a closed terminal:
#   tmux new -s highway
#   bash scripts/run_matrix.sh
#   Ctrl-b then d to detach, tmux attach -t highway to check on it.
#
# Splitting across laptops: pass a seed to run only that slice here.
#   bash scripts/run_matrix.sh 0     # laptop 1
#   bash scripts/run_matrix.sh 1     # laptop 2
#   bash scripts/run_matrix.sh 2     # laptop 3
#
# No 'set -e' on purpose: a crashed run should advance to the next one, not
# waste the rest of the night. pipefail keeps the failure message accurate.
set -o pipefail

CONFIGS=(speed_heavy speed_lean balanced safety_lean safety_heavy)
if [[ $# -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(0 1 2); fi

mkdir -p logs

for seed in "${SEEDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    echo "=== training ${cfg} seed ${seed} ==="
    python -u train.py --config "${cfg}" --seed "${seed}" 2>&1 \
      | tee "logs/${cfg}_seed${seed}.log" \
      || echo "!!! ${cfg} seed ${seed} failed, continuing with the next run"
  done
done

echo "Matrix complete. Next: evaluate.py per run, then plot_tradeoff.py."
