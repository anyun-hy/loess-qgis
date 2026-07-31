#!/bin/bash
# Usage: run_env_check.sh [output_dir]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if ! "$CONDA_EXE" --version >/dev/null 2>&1; then
  echo "Conda executable not found: $CONDA_EXE. Edit config.sh -> CONDA_EXE." >&2
  exit 127
fi

ARGS=(--scripts-dir "$SCRIPT_DIR" --conda-env "$CONDA_ENV")
[ -n "${1:-}" ] && ARGS+=(--output-dir "$1")

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/check_environment.py" "${ARGS[@]}"
