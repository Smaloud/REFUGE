#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

python3 -m refuge_seg.plot_dataset \
  --data-root REFUGE \
  --split train \
  --image-size 512 \
  --num-samples 6 \
  --output outputs/figures/task2_dataset_overview.png

python3 -m refuge_seg.plot_dataset_assets \
  --data-root REFUGE \
  --mode all \
  --stats-output outputs/figures/task2_dataset_stats.png \
  --structure-output outputs/figures/task2_structure_example.png

