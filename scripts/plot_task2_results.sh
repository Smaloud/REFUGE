#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.plot_results \
  --experiment CE outputs/checkpoints/task2_refuge_loss_ce/history.json outputs/checkpoints/task2_refuge_loss_ce/eval_val/metrics.json \
  --experiment Dice outputs/checkpoints/task2_refuge_loss_dice/history.json outputs/checkpoints/task2_refuge_loss_dice/eval_val/metrics.json \
  --experiment baseline outputs/checkpoints/task2_refuge_baseline/history.json outputs/checkpoints/task2_refuge_baseline/eval_val/metrics.json \
  --experiment lr_1e-4 outputs/checkpoints/task2_refuge_lr_1e4/history.json outputs/checkpoints/task2_refuge_lr_1e4/eval_val/metrics.json \
  --experiment batch2 outputs/checkpoints/task2_refuge_batch2/history.json outputs/checkpoints/task2_refuge_batch2/eval_val/metrics.json \
  --experiment step_scheduler outputs/checkpoints/task2_refuge_scheduler_step/history.json outputs/checkpoints/task2_refuge_scheduler_step/eval_val/metrics.json \
  --experiment topology outputs/checkpoints/task2_refuge_topology/history.json outputs/checkpoints/task2_refuge_topology/eval_val/metrics.json \
  --curve-output outputs/figures/task2_training_curves.png \
  --bar-output outputs/figures/task2_metrics_bar.png
