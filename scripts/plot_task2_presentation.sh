#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

args=(
  --data-root REFUGE
  --output-dir outputs/figures
)

for summary in \
  outputs/checkpoints/task2_refuge_baseline/eval_val/prediction_summary.json \
  outputs/checkpoints/task2_refuge_topology/eval_val/prediction_summary.json \
  outputs/predictions/test/prediction_summary.json
do
  if [[ -f "$summary" ]]; then
    args+=(--summary "$summary")
  fi
done

python3 -m refuge_seg.plot_presentation_assets "${args[@]}"
