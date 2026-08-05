#!/bin/bash
# Production usage: run_work_package.sh --run-spec <json> --worker-id <id> \
#   [--device <device>] [--max-open-frontier-units <count>] [--resume]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/work_package_runtime.py" "$@"
