#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts
export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python scripts/smoke_isaac.py --headless --enable_cameras "$@" \
    > logs/isaac-smoke.log 2>&1
grep SMOKE_PASS logs/isaac-smoke.log
