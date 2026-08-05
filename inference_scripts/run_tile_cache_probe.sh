#!/bin/bash
# Measure one real Tile with the production materializer before Run creation.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if ! "$CONDA_EXE" --version >/dev/null 2>&1; then
  echo "Conda executable not found: $CONDA_EXE" >&2
  exit 127
fi

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
  python "$SCRIPT_DIR/tile_cache_probe.py" "$@"
