#!/bin/bash
# Usage: run_env_check.sh [output_dir] [report_json]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if ! "$CONDA_EXE" --version >/dev/null 2>&1; then
  echo "Conda executable not found: $CONDA_EXE. Edit config.sh -> CONDA_EXE." >&2
  exit 127
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python -m loess_runtime.system.check_environment "$@"
fi

ARGS=(
  --scripts-dir "$LOESS_CONFIG_ROOT"
  --asset-base-dir "$LOESS_CONFIG_ASSET_ROOT"
  --conda-env "$CONDA_ENV"
)
[ -n "${1:-}" ] && ARGS+=(--output-dir "$1")
[ -n "${2:-}" ] && ARGS+=(--report-json "$2")

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python -m loess_runtime.system.check_environment "${ARGS[@]}"
