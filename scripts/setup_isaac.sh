#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LAB_COMMIT=3c6e67bb5c7ada942a6d1884ab69338f57596f77
uv venv --python 3.11 --seed --allow-existing .venv
if [ ! -d .deps/IsaacLab/.git ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab
    git -C .deps/IsaacLab checkout --detach "$LAB_COMMIT"
fi
test "$(git -C .deps/IsaacLab rev-parse HEAD)" = "$LAB_COMMIT"
if [ -f requirements-isaac.lock.txt ]; then
    uv pip install --python .venv/bin/python \
        --build-constraint constraints-build.txt \
        --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
        -r requirements-isaac.lock.txt
    uv pip check --python .venv/bin/python
    exit 0
fi
uv pip install --python .venv/bin/python \
    'isaacsim[all,extscache]==5.1.0' \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python \
    --build-constraint constraints-build.txt \
    -e .deps/IsaacLab/source/isaaclab \
    -e .deps/IsaacLab/source/isaaclab_assets \
    -e .deps/IsaacLab/source/isaaclab_tasks \
    -e .deps/IsaacLab/source/isaaclab_rl \
    'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0' \
    'wheel==0.45.1' 'transformers==4.57.6'
