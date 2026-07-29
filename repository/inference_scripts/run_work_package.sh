#!/bin/bash
# Usage: run_work_package.sh --run-spec <json> --package-id <id> [--device <device>] [--resume]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/work_package_runtime.py" "$@"
