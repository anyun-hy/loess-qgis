# Shared launcher config for Ubuntu/QGIS 3.44 and macOS/QGIS 4.2.
CONDA_ENV="${CONDA_ENV:-qgis}"
LOESS_PLATFORM="${LOESS_PLATFORM:-auto}"
LOESS_INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOESS_PROJECT_ROOT="$(cd "${LOESS_INFERENCE_DIR}/.." && pwd)"
LOESS_RUNTIME_ROOT="${LOESS_PROJECT_ROOT}/runtime"

# An initialized project carries a minimal, hash-bound copy of the three pure
# Python modules shared with the QGIS plugin.  Source checkouts continue to use
# their adjacent qgis_plugins tree; deployed projects resolve the same import
# names from this runtime root without rewriting Python files.
if [ -d "${LOESS_RUNTIME_ROOT}/labeling_tool/core" ]; then
  export PYTHONPATH="${LOESS_RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [ "$LOESS_PLATFORM" = "auto" ]; then
  case "$(uname -s)" in
    Darwin) LOESS_PLATFORM="macos" ;;
    Linux) LOESS_PLATFORM="ubuntu" ;;
    *) LOESS_PLATFORM="unsupported" ;;
  esac
fi

case "$LOESS_PLATFORM" in
  macos)
    CONDA_EXE="${CONDA_EXE:-/opt/anaconda3/bin/conda}"
    LOESS_ENV_LOCK="environment-macos-qgis4.yml"
    ;;
  ubuntu)
    CONDA_EXE="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
    LOESS_ENV_LOCK="environment-ubuntu-cu124.yml"
    ;;
  *)
    echo "Unsupported LOESS_PLATFORM: $LOESS_PLATFORM" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

if [ ! -x "$CONDA_EXE" ]; then
  CONDA_EXE=""
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  else
    for candidate in \
      "$HOME/miniconda3/bin/conda" \
      "$HOME/anaconda3/bin/conda" \
      "/opt/conda/bin/conda" \
      "/opt/anaconda3/bin/conda" \
      "/usr/local/bin/conda"; do
      if [ -x "$candidate" ]; then
        CONDA_EXE="$candidate"
        break
      fi
    done
  fi
fi

CONDA_EXE="${CONDA_EXE:-conda}"

# QGIS may inherit an incomplete activated-Conda state (for example,
# CONDA_SHLVL=1 without CONDA_PREFIX).  `conda run` cannot deactivate that
# phantom environment, so launch every inference command from a clean state.
unset CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE CONDA_ROOT _CE_CONDA _CE_M
export CONDA_SHLVL=0

export CONDA_ENV CONDA_EXE LOESS_PLATFORM LOESS_ENV_LOCK
export LOESS_INFERENCE_DIR LOESS_PROJECT_ROOT LOESS_RUNTIME_ROOT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
if [ "$LOESS_PLATFORM" = "ubuntu" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export CUDA_LAUNCH_BLOCKING=0
fi
