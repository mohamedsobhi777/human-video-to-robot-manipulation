#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts logs
# Prevent accidental concurrent Isaac processes contending for this laptop GPU.
flock -n artifacts/.isaac.lock env -u PYTHONPATH .venv/bin/python \
    scripts/run_baseline.py --headless --video \
    --wandb-mode online --entity mohamedsobhi777 "$@"
