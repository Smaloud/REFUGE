#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.train --config configs/task2_refuge_loss_ce.yaml
python3 -m refuge_seg.train --config configs/task2_refuge_loss_dice.yaml
python3 -m refuge_seg.train --config configs/task2_refuge_baseline.yaml
