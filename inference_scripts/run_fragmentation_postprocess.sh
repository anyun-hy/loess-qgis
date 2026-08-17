#!/bin/bash
# Usage: run_fragmentation_postprocess.sh --run-spec <json> --stream-id <fusion:id>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/fragmentation_postprocess.py" "$@"
