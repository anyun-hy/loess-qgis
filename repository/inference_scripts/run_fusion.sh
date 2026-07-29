#!/bin/bash
# Usage: run_fusion.sh --run-spec <json> --profile <json> [--device <device>] [--resume]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/fusion_runtime.py" "$@"

