#!/usr/bin/env bash
set -euo pipefail

PLUGIN_NAME="labeling_tool"
PLATFORM="${LOESS_PLATFORM:-auto}"
PROFILE="${QGIS_PROFILE:-default}"
PLUGIN_DIR_OVERRIDE="${QGIS_PLUGIN_DIR:-}"
CONDA_EXE="${CONDA_EXE:-}"
ENV_NAME="${CONDA_ENV:-qgis}"
CREATE_ENV=0
CHECK_ONLY=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="${SCRIPT_DIR}/qgis_plugins/${PLUGIN_NAME}"
INFERENCE_DIR="${SCRIPT_DIR}/inference_scripts"

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Install the shared QGIS plugin from this Ubuntu Git repository.

Options:
  --platform NAME      auto, ubuntu, or macos. Default: auto
  --profile NAME       QGIS profile name. Default: default
  --plugin-dir PATH    Override the QGIS plugin directory
  --conda-exe PATH     Override the Conda executable
  --create-env         Create or update the platform inference environment
  --check-only         Validate source, host, environment, and manifest only
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="${2:?Missing value for --platform}"
      shift 2
      ;;
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
    --check-only)
      CHECK_ONLY=1
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

if [[ "${PLATFORM}" == "auto" ]]; then
  case "$(uname -s)" in
    Darwin) PLATFORM="macos" ;;
    Linux) PLATFORM="ubuntu" ;;
    *)
      echo "Unsupported operating system: $(uname -s)" >&2
      exit 1
      ;;
  esac
fi

case "${PLATFORM}" in
  ubuntu)
    EXPECTED_QGIS="3.44."
    ENV_FILE="${INFERENCE_DIR}/environment-ubuntu-cu124.yml"
    DEFAULT_PLUGIN_ROOT="${HOME}/.local/share/QGIS/QGIS3/profiles/${PROFILE}/python/plugins"
    DEFAULT_CONDA_EXE="${HOME}/anaconda3/bin/conda"
    ;;
  macos)
    EXPECTED_QGIS="4.2."
    ENV_FILE="${INFERENCE_DIR}/environment-macos-qgis4.yml"
    DEFAULT_PLUGIN_ROOT="${HOME}/Library/Application Support/QGIS/QGIS4/profiles/${PROFILE}/python/plugins"
    DEFAULT_CONDA_EXE="/opt/anaconda3/bin/conda"
    ;;
  *)
    echo "Invalid platform: ${PLATFORM}; expected auto, ubuntu, or macos" >&2
    exit 2
    ;;
esac

QGIS_PLUGIN_ROOT="${PLUGIN_DIR_OVERRIDE:-${DEFAULT_PLUGIN_ROOT}}"
DEST_PLUGIN="${QGIS_PLUGIN_ROOT}/${PLUGIN_NAME}"

case "${DEST_PLUGIN}" in
  ""|"/"|"${HOME}"|"${HOME}/"|"/usr"|"/usr/"|"/opt"|"/opt/")
    echo "Refusing unsafe plugin destination: ${DEST_PLUGIN}" >&2
    exit 1
    ;;
esac

[[ -d "${PLUGIN_SRC}" ]] || { echo "Plugin source not found: ${PLUGIN_SRC}" >&2; exit 1; }
[[ -f "${PLUGIN_SRC}/metadata.txt" ]] || { echo "Missing plugin metadata" >&2; exit 1; }
[[ -f "${PLUGIN_SRC}/__init__.py" ]] || { echo "Missing plugin entry point" >&2; exit 1; }
[[ -f "${ENV_FILE}" ]] || { echo "Missing platform environment: ${ENV_FILE}" >&2; exit 1; }
grep -q '^version=0\.4\.0$' "${PLUGIN_SRC}/metadata.txt" || {
  echo "Plugin metadata is not the unified 0.4.0 release" >&2
  exit 1
}
grep -q '^qgisMinimumVersion=3\.44$' "${PLUGIN_SRC}/metadata.txt"
grep -q '^qgisMaximumVersion=4\.99$' "${PLUGIN_SRC}/metadata.txt"

detect_qgis_version() {
  local candidate version
  for candidate in \
    "${QGIS_PROCESS_EXE:-}" \
    "$(command -v qgis_process 2>/dev/null || true)" \
    "$(command -v qgis 2>/dev/null || true)" \
    /Applications/QGIS*.app/Contents/MacOS/qgis_process; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    version="$("${candidate}" --version 2>&1 | grep -Eo 'QGIS [0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true)"
    if [[ -n "${version}" ]]; then
      printf '%s\n' "${version}"
      return 0
    fi
  done
  return 1
}

qgis_version="$(detect_qgis_version || true)"
if [[ "${qgis_version}" != *"${EXPECTED_QGIS}"* ]]; then
  echo "Expected QGIS ${EXPECTED_QGIS}x for ${PLATFORM}; detected: ${qgis_version:-none}" >&2
  exit 1
fi

if [[ -z "${CONDA_EXE}" ]]; then
  CONDA_EXE="${DEFAULT_CONDA_EXE}"
fi
if [[ ! -x "${CONDA_EXE}" ]]; then
  CONDA_EXE="$(command -v conda 2>/dev/null || true)"
fi
[[ -x "${CONDA_EXE}" ]] || { echo "Conda executable not found" >&2; exit 1; }

if [[ "${CREATE_ENV}" -eq 1 ]]; then
  if "${CONDA_EXE}" run -n "${ENV_NAME}" python -V >/dev/null 2>&1; then
    "${CONDA_EXE}" env update -n "${ENV_NAME}" -f "${ENV_FILE}"
  else
    "${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
  if [[ "${PLATFORM}" == "ubuntu" ]]; then
    "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124
  fi
fi

PYTHON_BIN="$("${CONDA_EXE}" run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
"${PYTHON_BIN}" -m compileall -q "${PLUGIN_SRC}" "${INFERENCE_DIR}"

STAGE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/loess-plugin-install.XXXXXX")"
STAGE_PLUGIN="${STAGE_PARENT}/${PLUGIN_NAME}"
cleanup() {
  rm -rf "${STAGE_PARENT}"
}
trap cleanup EXIT

mkdir -p "${STAGE_PLUGIN}"
rsync -rl --delete --omit-dir-times \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '._*' \
  "${PLUGIN_SRC}/" "${STAGE_PLUGIN}/"

GIT_SHA="${LOESS_GIT_SHA:-}"
if [[ -z "${GIT_SHA}" ]] && git -C "${SCRIPT_DIR}/.." rev-parse HEAD >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${SCRIPT_DIR}/.." rev-parse HEAD)"
fi
[[ "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Cannot determine canonical Git SHA; run from the Ubuntu repository or set LOESS_GIT_SHA" >&2
  exit 1
}

LOESS_STAGE_PLUGIN="${STAGE_PLUGIN}" \
LOESS_GIT_SHA="${GIT_SHA}" \
LOESS_PLATFORM="${PLATFORM}" \
LOESS_PROFILE="${PROFILE}" \
"${PYTHON_BIN}" -c '
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["LOESS_STAGE_PLUGIN"])
files = {}
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    rel = path.relative_to(root).as_posix()
    files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "git_sha": os.environ["LOESS_GIT_SHA"],
    "plugin_version": "0.4.0",
    "platform": os.environ["LOESS_PLATFORM"],
    "qgis_profile": os.environ["LOESS_PROFILE"],
    "files": files,
}
(root / "deployment_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
'

echo "Validated ${PLUGIN_NAME} 0.4.0"
echo "  platform: ${PLATFORM}"
echo "  QGIS: ${qgis_version}"
echo "  Git SHA: ${GIT_SHA}"
echo "  environment: ${ENV_FILE}"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Check-only complete; installed plugin was not changed."
  exit 0
fi

mkdir -p "${QGIS_PLUGIN_ROOT}"
STAGED_DEST="${QGIS_PLUGIN_ROOT}/.${PLUGIN_NAME}.new.$$"
OLD_DEST="${QGIS_PLUGIN_ROOT}/.${PLUGIN_NAME}.old.$$"
rm -rf "${STAGED_DEST}" "${OLD_DEST}"
mv "${STAGE_PLUGIN}" "${STAGED_DEST}"

if [[ -e "${DEST_PLUGIN}" ]]; then
  mv "${DEST_PLUGIN}" "${OLD_DEST}"
fi
if mv "${STAGED_DEST}" "${DEST_PLUGIN}"; then
  rm -rf "${OLD_DEST}"
else
  [[ ! -e "${DEST_PLUGIN}" && -e "${OLD_DEST}" ]] && mv "${OLD_DEST}" "${DEST_PLUGIN}"
  echo "Atomic plugin installation failed; previous installation was restored" >&2
  exit 1
fi

test -f "${DEST_PLUGIN}/metadata.txt"
test -f "${DEST_PLUGIN}/deployment_manifest.json"
echo "Installed shared plugin: ${DEST_PLUGIN}"
