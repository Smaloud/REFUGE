#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.plot_results \
  --experiment baseline outputs/checkpoints/task2_refuge_baseline/history.json outputs/checkpoints/task2_refuge_baseline/eval_val/metrics.json \
  --experiment topology outputs/checkpoints/task2_refuge_topology/history.json outputs/checkpoints/task2_refuge_topology/eval_val/metrics.json \
  --curve-output outputs/figures/task2_training_curves.png \
  --bar-output outputs/figures/task2_metrics_bar.png

