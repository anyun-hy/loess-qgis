#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [ "$#" -eq 0 ]; then
  set -- --config "$LOESS_CONFIG_ROOT/config.yaml" --asset-base-dir "$LOESS_CONFIG_ASSET_ROOT"
fi

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python -m loess_runtime.sam.sam3_interactive_worker "$@"
