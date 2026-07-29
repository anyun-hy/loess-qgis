#!/usr/bin/env bash
set -euo pipefail

PROFILE="${QGIS_PROFILE:-default}"
PLUGIN_DIR_OVERRIDE="${QGIS_PLUGIN_DIR:-}"
CONDA_EXE="${CONDA_EXE:-${HOME}/anaconda3/bin/conda}"
ENV_NAME="qgis"
CREATE_ENV=0
DRY_RUN=0
WEIGHTS_SOURCE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="${SCRIPT_DIR}/qgis_plugins/labeling_tool"
INFERENCE_DIR="${SCRIPT_DIR}/inference_scripts"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
ENV_FILE="${INFERENCE_DIR}/environment-linux-cu124.yml"

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Install the Ubuntu QGIS 3.44 plugin and verify the RTX 3090 inference runtime.

Options:
  --profile NAME       QGIS3 profile name. Default: default
  --plugin-dir PATH    Override the QGIS3 plugin directory
  --conda-exe PATH     Conda executable
  --create-env         Create/update ${ENV_NAME} and install Torch 2.6 cu124
  --weights-dir PATH   Link formal model files from PATH into linux/weights
  --dry-run            Print actions without changing files
  -h, --help           Show this help
EOF
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?Missing value for --profile}"
      shift 2
      ;;
    --plugin-dir)
      PLUGIN_DIR_OVERRIDE="${2:?Missing value for --plugin-dir}"
      shift 2
      ;;
    --conda-exe)
      CONDA_EXE="${2:?Missing value for --conda-exe}"
      shift 2
      ;;
    --create-env)
      CREATE_ENV=1
      shift
      ;;
    --weights-dir)
      WEIGHTS_SOURCE="${2:?Missing value for --weights-dir}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${PROFILE}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid QGIS profile name: ${PROFILE}" >&2
  exit 2
}

if [[ "$(uname -s)" != "Linux" && "${DRY_RUN}" -ne 1 ]]; then
  echo "This installer only runs on Linux." >&2
  exit 1
fi

QGIS_PLUGIN_ROOT="${PLUGIN_DIR_OVERRIDE:-${HOME}/.local/share/QGIS/QGIS3/profiles/${PROFILE}/python/plugins}"
DEST_PLUGIN="${QGIS_PLUGIN_ROOT}/labeling_tool"
case "${DEST_PLUGIN}" in
  ""|"/"|"${HOME}"|"${HOME}/"|"/usr"|"/usr/"|"/opt"|"/opt/")
    echo "Refusing unsafe plugin destination: ${DEST_PLUGIN}" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" -eq 0 ]]; then
  qgis_version=""
  for executable in qgis qgis_process; do
    if command -v "${executable}" >/dev/null 2>&1; then
      qgis_version="$("${executable}" --version 2>&1 | head -n 1)"
      [[ -n "${qgis_version}" ]] && break
    fi
  done
  if [[ "${qgis_version}" != *"3.44."* ]]; then
    echo "QGIS 3.44 was not detected: ${qgis_version:-command not found}" >&2
    exit 1
  fi
  echo "Detected ${qgis_version}"
fi

if [[ ! -x "${CONDA_EXE}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  elif [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "Conda executable not found: ${CONDA_EXE}" >&2
    exit 1
  fi
fi

if [[ "${CREATE_ENV}" -eq 1 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]] || ! "${CONDA_EXE}" run -n "${ENV_NAME}" python -V >/dev/null 2>&1; then
    run_cmd "${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  else
    run_cmd "${CONDA_EXE}" env update -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
  run_cmd "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
  run_cmd "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade \
    sam3==0.1.4 timm==1.0.28 tqdm==4.67.3 ftfy==6.3.1 \
    regex==2026.7.10 iopath==0.1.10 typing_extensions==4.15.0 \
    huggingface-hub==1.23.0 einops==0.8.2 pycocotools==2.0.11 \
    safetensors==0.8.0 psutil==7.2.2
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${CONDA_EXE}" run -n "${ENV_NAME}" python -c '
import sys
import torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+", 1)[0] == "2.6.0", torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
name = torch.cuda.get_device_name(0)
assert "3090" in name, name
print(f"Inference runtime: Python {sys.version.split()[0]}, Torch {torch.__version__}, GPU {name}")
'
fi

required_assets=(
  upernet_swin_b.torchscript.pt
  setr_vit.torchscript.pt
  upernet_mambaout_b.torchscript.pt
  sam3.pt
)

if [[ -n "${WEIGHTS_SOURCE}" ]]; then
  for asset in "${required_assets[@]}"; do
    source_path="${WEIGHTS_SOURCE}/${asset}"
    if [[ "${DRY_RUN}" -eq 0 && ! -f "${source_path}" ]]; then
      echo "Missing formal asset: ${source_path}" >&2
      exit 1
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      source_abs="${source_path}"
    else
      source_abs="$(realpath "${source_path}")"
    fi
    run_cmd ln -sfn "${source_abs}" "${WEIGHTS_DIR}/${asset}"
  done
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  declare -A expected_sha=(
    [upernet_swin_b.torchscript.pt]="e1b28d88821f0a35e17e399c89d01ef010c8a73dd1dce797f4db2f2b17425214"
    [setr_vit.torchscript.pt]="7e47de54db003a802a36f78bb26295954201467234fbb084e00db4925a74a12f"
    [upernet_mambaout_b.torchscript.pt]="76819fa558ce4261033fc6b0d65353778ec2a86cb4b0a0ce4a5ba8fd03ae1054"
    [sam3.pt]="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
    [fusion_profile.json]="1e4a3086d016b26c074499a2035fea28ad0a4056e1cf3539fb04eb5e2f8c615d"
  )
  for asset in "${required_assets[@]}" fusion_profile.json; do
    path="${WEIGHTS_DIR}/${asset}"
    [[ -f "${path}" ]] || { echo "Missing formal asset: ${path}" >&2; exit 1; }
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected_sha[${asset}]}" ]] || {
      echo "SHA256 mismatch: ${path}" >&2
      exit 1
    }
  done
fi

run_cmd find "${PLUGIN_SRC}" "${INFERENCE_DIR}" -type f -name '._*' -delete
run_cmd find "${INFERENCE_DIR}" -type f -name '*.sh' -exec chmod +x '{}' +
run_cmd "${CONDA_EXE}" run -n "${ENV_NAME}" python -m compileall -q \
  "${PLUGIN_SRC}" "${INFERENCE_DIR}"
run_cmd mkdir -p "${QGIS_PLUGIN_ROOT}"
if command -v rsync >/dev/null 2>&1 || [[ "${DRY_RUN}" -eq 1 ]]; then
  run_cmd rsync -rl --delete --omit-dir-times \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '._*' \
    "${PLUGIN_SRC}/" "${DEST_PLUGIN}/"
else
  run_cmd rm -rf "${DEST_PLUGIN}"
  run_cmd mkdir -p "${DEST_PLUGIN}"
  run_cmd cp -a "${PLUGIN_SRC}/." "${DEST_PLUGIN}/"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete; no files were changed."
else
  echo "Installed Linux plugin: ${DEST_PLUGIN}"
  echo "Inference scripts: ${INFERENCE_DIR}"
  echo "Restart QGIS 3.44, enable '半自动标注工具', then select the inference_scripts path above."
fi
