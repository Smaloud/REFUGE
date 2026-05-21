#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.train --config configs/task2_refuge_baseline.yaml
python3 -m refuge_seg.train --config configs/task2_refuge_lr_1e4.yaml
python3 -m refuge_seg.train --config configs/task2_refuge_batch2.yaml
python3 -m refuge_seg.train --config configs/task2_refuge_scheduler_step.yaml
