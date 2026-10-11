#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts logs
# Prevent accidental concurrent Isaac processes contending for this laptop GPU.
flock -n artifacts/.isaac.lock env -u PYTHONPATH PYTHONUNBUFFERED=1 .venv/bin/python \
    scripts/evaluate.py --headless --wandb-mode online --entity mohamedsobhi777 "$@"
