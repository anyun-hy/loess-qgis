#!/usr/bin/env bash
set -euo pipefail

PLUGIN_NAME="labeling_tool"
PLATFORM="${LOESS_PLATFORM:-auto}"
PROFILE="${QGIS_PROFILE:-default}"
PLUGIN_DIR_OVERRIDE="${QGIS_PLUGIN_DIR:-}"
CHECK_ONLY=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_SRC="${SOURCE_ROOT}/qgis_plugins/${PLUGIN_NAME}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || true)}"

usage() {
  cat <<EOF
Usage: ./bash/install_plugin.sh [options]

Install only the QGIS plugin. This command never creates a Loess project,
deploys inference scripts, creates weight/output directories, or changes Conda.

Options:
  --platform NAME      auto, ubuntu, or macos. Default: auto
  --profile NAME       QGIS profile name. Default: default
  --plugin-dir PATH    Override the QGIS plugin directory
  --check-only         Validate without changing the installed plugin
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
    DEFAULT_PLUGIN_ROOT="${HOME}/.local/share/QGIS/QGIS3/profiles/${PROFILE}/python/plugins"
    ;;
  macos)
    EXPECTED_QGIS="4.2."
    DEFAULT_PLUGIN_ROOT="${HOME}/Library/Application Support/QGIS/QGIS4/profiles/${PROFILE}/python/plugins"
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

[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "Python 3 executable not found; set PYTHON_BIN" >&2
  exit 1
}
[[ -d "${PLUGIN_SRC}" ]] || {
  echo "Plugin source not found: ${PLUGIN_SRC}" >&2
  exit 1
}
[[ -f "${PLUGIN_SRC}/metadata.txt" ]] || {
  echo "Missing plugin metadata" >&2
  exit 1
}
[[ -f "${PLUGIN_SRC}/__init__.py" ]] || {
  echo "Missing plugin entry point" >&2
  exit 1
}
for shared_name in run_spec.py run_state_db.py ownership_neighbors.py; do
  [[ -f "${PLUGIN_SRC}/core/${shared_name}" ]] || {
    echo "Missing shared runtime source: ${shared_name}" >&2
    exit 1
  }
done
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

GIT_SHA="${LOESS_GIT_SHA:-}"
if [[ -z "${GIT_SHA}" ]] && git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
fi
[[ "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Cannot determine canonical Git SHA; set LOESS_GIT_SHA when installing an archive" >&2
  exit 1
}

STAGE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/loess-plugin-install.XXXXXX")"
STAGE_PLUGIN="${STAGE_PARENT}/${PLUGIN_NAME}"
cleanup() {
  rm -rf -- "${STAGE_PARENT}"
}
trap cleanup EXIT

mkdir -p "${STAGE_PLUGIN}"
rsync -rlpt --delete --omit-dir-times \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '._*' \
  "${PLUGIN_SRC}/" "${STAGE_PLUGIN}/"
PYTHONPYCACHEPREFIX="${STAGE_PARENT}/compile-cache" \
  "${PYTHON_BIN}" -m compileall -q "${STAGE_PLUGIN}"
rm -rf -- "${STAGE_PARENT}/compile-cache"

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

shared = {}
aggregate = hashlib.sha256()
for name in sorted(("run_spec.py", "run_state_db.py", "ownership_neighbors.py")):
    canonical = f"qgis_plugins/labeling_tool/core/{name}"
    digest = files[f"core/{name}"]
    shared[canonical] = digest
    aggregate.update(canonical.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(digest.encode("ascii"))
    aggregate.update(b"\n")

payload = {
    "schema_version": 2,
    "deployment_kind": "qgis_plugin",
    "git_sha": os.environ["LOESS_GIT_SHA"],
    "plugin_version": "0.4.0",
    "platform": os.environ["LOESS_PLATFORM"],
    "qgis_profile": os.environ["LOESS_PROFILE"],
    "shared_runtime": {
        "import_root": "labeling_tool.core",
        "sha256": aggregate.hexdigest(),
        "files": shared,
    },
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

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Check-only complete; installed plugin was not changed."
  exit 0
fi

mkdir -p "${QGIS_PLUGIN_ROOT}"
STAGED_DEST="${QGIS_PLUGIN_ROOT}/.${PLUGIN_NAME}.new.$$"
OLD_DEST="${QGIS_PLUGIN_ROOT}/.${PLUGIN_NAME}.old.$$"
rm -rf -- "${STAGED_DEST}" "${OLD_DEST}"
mv "${STAGE_PLUGIN}" "${STAGED_DEST}"

if [[ -e "${DEST_PLUGIN}" ]]; then
  mv "${DEST_PLUGIN}" "${OLD_DEST}"
fi
if mv "${STAGED_DEST}" "${DEST_PLUGIN}"; then
  rm -rf -- "${OLD_DEST}"
else
  [[ ! -e "${DEST_PLUGIN}" && -e "${OLD_DEST}" ]] && mv "${OLD_DEST}" "${DEST_PLUGIN}"
  echo "Atomic plugin installation failed; previous installation was restored" >&2
  exit 1
fi

test -f "${DEST_PLUGIN}/metadata.txt"
test -f "${DEST_PLUGIN}/deployment_manifest.json"
echo "Installed QGIS plugin only: ${DEST_PLUGIN}"
