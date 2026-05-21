#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python3 -m refuge_seg.summarize_experiments \
  --output reports/task2_experiment_summary.md
