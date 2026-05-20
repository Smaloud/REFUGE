#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.evaluate \
  --config configs/task2_refuge_baseline.yaml \
  --checkpoint outputs/checkpoints/task2_refuge_baseline/best_model.pt \
  --split val \
  --postprocess

