#!/bin/bash
# Usage: run_finalize_partition_rasters.sh --run-spec <json>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/finalize_partition_rasters.py" "$@"
