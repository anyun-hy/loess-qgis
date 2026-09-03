#!/bin/bash
# Usage: run_unit_confidence.sh --run-spec <json> --stream-id <id> --unit-id <id> --job-id <n> --lease-token <token>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/unit_confidence.py" "$@"
