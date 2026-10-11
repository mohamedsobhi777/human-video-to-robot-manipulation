#!/usr/bin/env bash
# One-command ManiSkill3 setup for a fresh Google Colab GPU VM (verified 2026-10-11 on A100,
# driver 580.82.07). Colab ships Python 3.13 and only NVIDIA compute libraries, so this
# (1) installs the NVIDIA user-space graphics libraries matching the host driver for Vulkan
# rendering and (2) creates a Python 3.11 env, since mani_skill pins wheels without cp313.
set -euo pipefail
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if ! vulkaninfo --summary >/dev/null 2>&1; then
  apt-get -qq update >/dev/null && apt-get -qq install -y libvulkan1 vulkan-tools >/dev/null
  cd /content
  wget -q "https://us.download.nvidia.com/tesla/${DRIVER}/NVIDIA-Linux-x86_64-${DRIVER}.run" \
    || wget -q "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER}/NVIDIA-Linux-x86_64-${DRIVER}.run"
  sh "NVIDIA-Linux-x86_64-${DRIVER}.run" --no-kernel-module --no-kernel-module-source --no-x-check \
    --no-nouveau-check --no-systemd --no-check-for-alternate-installs --skip-module-load --silent --install-libglvnd
fi
vulkaninfo --summary 2>&1 | grep -m1 deviceName
unset UV_PYTHON UV_SYSTEM_PYTHON PIP_TARGET
pip install -q uv
uv python install 3.11 -q
rm -rf /content/ms
uv venv -q -p "$(uv python find 3.11)" /content/ms
uv pip install -q --python /content/ms/bin/python mani_skill torch
/content/ms/bin/python -c 'import mani_skill, sapien, torch; print("mani_skill", mani_skill.__version__, "sapien", sapien.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
