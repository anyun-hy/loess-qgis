# Shared launcher config for Ubuntu/QGIS 3.44 inference scripts.
CONDA_ENV="${CONDA_ENV:-qgis}"
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

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

export CONDA_ENV CONDA_EXE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING=0
