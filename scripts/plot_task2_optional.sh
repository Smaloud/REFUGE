#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.plot_optional_cases \
  --baseline-config configs/task2_refuge_baseline.yaml \
  --baseline-ckpt outputs/checkpoints/task2_refuge_baseline/best_model.pt \
  --topology-config configs/task2_refuge_topology.yaml \
  --topology-ckpt outputs/checkpoints/task2_refuge_topology/best_model.pt \
  --split val \
  --num-cases 4 \
  --output outputs/figures/task2_optional_cases.png

