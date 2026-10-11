#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The host ROS Python 3.12 plugins do not belong in this Python 3.11 environment.
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests "$@"
