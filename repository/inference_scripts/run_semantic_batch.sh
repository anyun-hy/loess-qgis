#!/bin/bash
# Usage: run_semantic_batch.sh --run-spec <json> --model-id <id> [--device <device>] [--resume]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/semantic_batch.py" "$@"

