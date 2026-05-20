#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python3 -m refuge_seg.check_data --data-root REFUGE
