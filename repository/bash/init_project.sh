#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${LOESS_PLATFORM:-auto}"
PROJECT_ROOT="${LOESS_PROJECT_ROOT:-}"
CONDA_EXE="${CONDA_EXE:-}"
ENV_NAME="${CONDA_ENV:-qgis}"
CREATE_ENV=0
CHECK_ONLY=0
CHECK_ASSETS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_SRC="${SOURCE_ROOT}/inference_scripts"
PLUGIN_SRC="${SOURCE_ROOT}/qgis_plugins/labeling_tool"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || true)}"
RECOMMENDED_ROOT="${HOME}/Desktop/loess-project"

usage() {
  cat <<EOF
Usage: ./bash/init_project.sh [options]

Initialize or update only a Loess inference project. This command never
installs or modifies a QGIS plugin/profile.

Options:
  --project-root PATH  Project directory. If omitted, prompt with:
                       ${RECOMMENDED_ROOT}
  --platform NAME      auto, ubuntu, or macos. Default: auto
  --conda-exe PATH     Override the Conda executable
  --create-env         Create or update the platform inference environment
  --check-only         Validate the source and target without writing
  --check-assets       Fail unless required weight assets exist and hashes match
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="${2:?Missing value for --project-root}"
      shift 2
      ;;
    --platform)
      PLATFORM="${2:?Missing value for --platform}"
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
    --check-assets)
      CHECK_ASSETS=1
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

if [[ -z "${PROJECT_ROOT}" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Project root [${RECOMMENDED_ROOT}]: " PROJECT_ROOT
    PROJECT_ROOT="${PROJECT_ROOT:-${RECOMMENDED_ROOT}}"
  else
    echo "Missing --project-root (recommended: ${RECOMMENDED_ROOT})" >&2
    exit 2
  fi
fi

[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "Python 3 executable not found; set PYTHON_BIN" >&2
  exit 1
}
PROJECT_ROOT="$("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
' "${PROJECT_ROOT}")"

"${PYTHON_BIN}" -c '
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
source = Path(sys.argv[2])
home = Path.home().resolve()
if target in (Path("/"), home):
    raise SystemExit(f"Refusing unsafe project root: {target}")
try:
    target.relative_to(source)
except ValueError:
    pass
else:
    raise SystemExit(f"Project root cannot be inside the source repository: {target}")
try:
    source.relative_to(target)
except ValueError:
    pass
else:
    raise SystemExit(f"Project root cannot contain the source repository: {target}")
' "${PROJECT_ROOT}" "${SOURCE_ROOT}"

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
    ENV_FILE="${INFERENCE_SRC}/environment-ubuntu-cu124.yml"
    DEFAULT_CONDA_EXE="${HOME}/anaconda3/bin/conda"
    ;;
  macos)
    ENV_FILE="${INFERENCE_SRC}/environment-macos-qgis4.yml"
    DEFAULT_CONDA_EXE="/opt/anaconda3/bin/conda"
    ;;
  *)
    echo "Invalid platform: ${PLATFORM}; expected auto, ubuntu, or macos" >&2
    exit 2
    ;;
esac

[[ -d "${INFERENCE_SRC}" ]] || {
  echo "Inference source not found: ${INFERENCE_SRC}" >&2
  exit 1
}
[[ -f "${ENV_FILE}" ]] || {
  echo "Missing platform environment lock: ${ENV_FILE}" >&2
  exit 1
}
for shared_name in run_spec.py run_state_db.py ownership_neighbors.py; do
  [[ -f "${PLUGIN_SRC}/core/${shared_name}" ]] || {
    echo "Missing shared runtime source: ${shared_name}" >&2
    exit 1
  }
done

MANIFEST="${PROJECT_ROOT}/project_manifest.json"
if [[ -e "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"Invalid project manifest {path}: {exc}")
if payload.get("deployment_kind") != "loess_project":
    raise SystemExit(f"Refusing project with foreign manifest: {path}")
' "${MANIFEST}"
elif [[ -e "${PROJECT_ROOT}/inference_scripts" || -e "${PROJECT_ROOT}/runtime" ]]; then
  echo "Refusing to overwrite unmanaged inference_scripts/runtime without project_manifest.json" >&2
  exit 1
fi

GIT_SHA="${LOESS_GIT_SHA:-}"
if [[ -z "${GIT_SHA}" ]] && git -C "${SOURCE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
fi
[[ "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Cannot determine canonical Git SHA; set LOESS_GIT_SHA when initializing from an archive" >&2
  exit 1
}

check_assets() {
  "${PYTHON_BIN}" -c '
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
missing = []
mismatched = []
for asset in manifest.get("required_assets", []):
    path = root / asset["path"]
    if not path.is_file():
        missing.append(asset["path"])
        continue
    expected = str(asset.get("sha256") or "")
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            mismatched.append(asset["path"])
if missing or mismatched:
    if missing:
        print("Missing required assets: " + ", ".join(missing), file=sys.stderr)
    if mismatched:
        print("Asset SHA256 mismatch: " + ", ".join(mismatched), file=sys.stderr)
    raise SystemExit(3)
print("Required weight assets are present and valid.")
' "${MANIFEST}" "${PROJECT_ROOT}"
}

verify_existing_project() {
  "${PYTHON_BIN}" -c '
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
source_root = Path(sys.argv[3])
expected_git = sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if manifest.get("git_sha") != expected_git:
    raise SystemExit(
        "Project Git SHA differs from this source; rerun init_project.sh "
        f"to update it ({manifest.get('git_sha')} != {expected_git})"
    )
if Path(str(manifest.get("project_root") or "")).resolve() != root:
    raise SystemExit("Project was moved; rerun init_project.sh at its current path")

shared = manifest.get("shared_runtime") or {}
shared_files = shared.get("files") or {}
aggregate = hashlib.sha256()
for canonical, expected in sorted(shared_files.items()):
    name = Path(canonical).name
    source = source_root / canonical
    deployed = root / "runtime" / "labeling_tool" / "core" / name
    for label, path in (("source", source), ("project", deployed)):
        if not path.is_file():
            raise SystemExit(f"Missing shared runtime {label} file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Shared runtime {label} SHA256 mismatch: {path}")
    aggregate.update(canonical.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(expected.encode("ascii"))
    aggregate.update(b"\n")
if aggregate.hexdigest() != shared.get("sha256"):
    raise SystemExit("Shared runtime aggregate SHA256 mismatch")

expected_inference = manifest.get("inference_files") or {}
actual_inference = {}
inference_root = root / "inference_scripts"
for path in sorted(p for p in inference_root.rglob("*") if p.is_file()):
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        continue
    rel = path.relative_to(inference_root).as_posix()
    actual_inference[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_inference != expected_inference:
    raise SystemExit(
        "Managed inference_scripts differ from project_manifest.json; "
        "rerun init_project.sh"
    )
print("Existing project manifest and managed files are valid.")
' "${MANIFEST}" "${PROJECT_ROOT}" "${SOURCE_ROOT}" "${GIT_SHA}"
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Project initialization source is valid."
  echo "  platform: ${PLATFORM}"
  echo "  project root: ${PROJECT_ROOT}"
  echo "  Git SHA: ${GIT_SHA}"
  if [[ -f "${MANIFEST}" ]]; then
    verify_existing_project
  fi
  if [[ "${CHECK_ASSETS}" -eq 1 ]]; then
    [[ -f "${MANIFEST}" ]] || {
      echo "Cannot check assets before the project is initialized" >&2
      exit 1
    }
    check_assets
  fi
  echo "Check-only complete; project files were not changed."
  exit 0
fi

if [[ "${CREATE_ENV}" -eq 1 ]]; then
  if [[ -z "${CONDA_EXE}" ]]; then
    CONDA_EXE="${DEFAULT_CONDA_EXE}"
  fi
  if [[ ! -x "${CONDA_EXE}" ]]; then
    CONDA_EXE="$(command -v conda 2>/dev/null || true)"
  fi
  [[ -x "${CONDA_EXE}" ]] || {
    echo "Conda executable not found" >&2
    exit 1
  }
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

mkdir -p "${PROJECT_ROOT}"
STAGE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.loess-project-init.XXXXXX")"
OLD_INFERENCE="${PROJECT_ROOT}/.inference_scripts.old.$$"
OLD_RUNTIME="${PROJECT_ROOT}/.runtime.old.$$"
DEPLOY_COMMITTED=0
cleanup() {
  rm -rf -- "${STAGE_ROOT}"
  if [[ "${DEPLOY_COMMITTED}" -eq 1 ]]; then
    rm -rf -- "${OLD_INFERENCE}" "${OLD_RUNTIME}"
  fi
}
trap cleanup EXIT

mkdir -p "${STAGE_ROOT}/inference_scripts"
rsync -rlpt --delete --omit-dir-times \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '._*' \
  "${INFERENCE_SRC}/" "${STAGE_ROOT}/inference_scripts/"

mkdir -p "${STAGE_ROOT}/runtime/labeling_tool/core"
install -m 0644 "${PLUGIN_SRC}/__init__.py" \
  "${STAGE_ROOT}/runtime/labeling_tool/__init__.py"
install -m 0644 "${PLUGIN_SRC}/core/__init__.py" \
  "${STAGE_ROOT}/runtime/labeling_tool/core/__init__.py"
for shared_name in run_spec.py run_state_db.py ownership_neighbors.py; do
  install -m 0644 "${PLUGIN_SRC}/core/${shared_name}" \
    "${STAGE_ROOT}/runtime/labeling_tool/core/${shared_name}"
done

PYTHONPYCACHEPREFIX="${STAGE_ROOT}/compile-cache" "${PYTHON_BIN}" -m compileall -q \
  "${STAGE_ROOT}/inference_scripts" "${STAGE_ROOT}/runtime"
rm -rf -- "${STAGE_ROOT}/compile-cache"

LOESS_STAGE_ROOT="${STAGE_ROOT}" \
LOESS_GIT_SHA="${GIT_SHA}" \
LOESS_PLATFORM="${PLATFORM}" \
LOESS_PROJECT_ROOT="${PROJECT_ROOT}" \
"${PYTHON_BIN}" -c '
import hashlib
import json
import os
import re
from pathlib import Path

stage = Path(os.environ["LOESS_STAGE_ROOT"])
config = (stage / "inference_scripts" / "config.yaml").read_text(encoding="utf-8")

shared = {}
aggregate = hashlib.sha256()
for name in sorted(("run_spec.py", "run_state_db.py", "ownership_neighbors.py")):
    canonical = f"qgis_plugins/labeling_tool/core/{name}"
    path = stage / "runtime" / "labeling_tool" / "core" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    shared[canonical] = digest
    aggregate.update(canonical.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(digest.encode("ascii"))
    aggregate.update(b"\n")

inference_files = {}
for path in sorted(p for p in (stage / "inference_scripts").rglob("*") if p.is_file()):
    rel = path.relative_to(stage / "inference_scripts").as_posix()
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        continue
    inference_files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

assets = []
section = ""
pending_artifact = None
for raw_line in config.splitlines():
    line = raw_line.strip()
    if line == "semantic_models:":
        section = "models"
        continue
    if line == "fusion_profiles:":
        section = "fusion"
        continue
    if line == "sam3:":
        section = "sam3"
        continue
    if section == "models" and line.startswith("artifact:"):
        pending_artifact = line.split(":", 1)[1].strip().strip("\"'\''")
    elif section == "models" and pending_artifact and line.startswith("sha256:"):
        digest = line.split(":", 1)[1].strip().strip("\"'\''")
        assets.append({
            "kind": "semantic_model",
            "path": f"weights/{pending_artifact}",
            "sha256": digest,
        })
        pending_artifact = None
    elif section == "fusion" and line.startswith("file:"):
        value = line.split(":", 1)[1].strip().strip("\"'\''")
        assets.append({
            "kind": "fusion_profile",
            "path": "weights/" + Path(value).name,
            "sha256": "",
        })
    elif section == "sam3" and line.startswith("checkpoint:"):
        value = line.split(":", 1)[1].strip().strip("\"'\''")
        assets.append({
            "kind": "sam3_checkpoint",
            "path": "weights/" + Path(value).name,
            "sha256": "",
        })

payload = {
    "schema_version": 1,
    "deployment_kind": "loess_project",
    "git_sha": os.environ["LOESS_GIT_SHA"],
    "platform": os.environ["LOESS_PLATFORM"],
    "project_root": os.environ["LOESS_PROJECT_ROOT"],
    "managed_paths": ["inference_scripts", "runtime"],
    "paths": {
        "inference_scripts": "inference_scripts",
        "runtime": "runtime",
        "weights": "weights",
        "input_rasters": "input/rasters",
        "input_ranges": "input/ranges",
        "qgis": "qgis",
        "output": "output",
    },
    "shared_runtime": {
        "import_root": "labeling_tool.core",
        "sha256": aggregate.hexdigest(),
        "files": shared,
    },
    "inference_files": inference_files,
    "required_assets": assets,
}
(stage / "project_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
'

rm -rf -- "${OLD_INFERENCE}" "${OLD_RUNTIME}"
if [[ -e "${PROJECT_ROOT}/inference_scripts" ]]; then
  mv "${PROJECT_ROOT}/inference_scripts" "${OLD_INFERENCE}"
fi
if [[ -e "${PROJECT_ROOT}/runtime" ]]; then
  mv "${PROJECT_ROOT}/runtime" "${OLD_RUNTIME}"
fi

rollback() {
  rm -rf -- "${PROJECT_ROOT}/inference_scripts" "${PROJECT_ROOT}/runtime"
  [[ -e "${OLD_INFERENCE}" ]] && mv "${OLD_INFERENCE}" "${PROJECT_ROOT}/inference_scripts"
  [[ -e "${OLD_RUNTIME}" ]] && mv "${OLD_RUNTIME}" "${PROJECT_ROOT}/runtime"
}

if ! mv "${STAGE_ROOT}/inference_scripts" "${PROJECT_ROOT}/inference_scripts"; then
  rollback
  exit 1
fi
if ! mv "${STAGE_ROOT}/runtime" "${PROJECT_ROOT}/runtime"; then
  rollback
  exit 1
fi
if ! mv "${STAGE_ROOT}/project_manifest.json" "${PROJECT_ROOT}/project_manifest.json"; then
  rollback
  exit 1
fi
DEPLOY_COMMITTED=1

mkdir -p \
  "${PROJECT_ROOT}/weights" \
  "${PROJECT_ROOT}/input/rasters" \
  "${PROJECT_ROOT}/input/ranges" \
  "${PROJECT_ROOT}/qgis" \
  "${PROJECT_ROOT}/output/runs" \
  "${PROJECT_ROOT}/output/cache"

if [[ ! -e "${PROJECT_ROOT}/weights/README_WEIGHTS.md" ]]; then
  install -m 0644 "${SCRIPT_DIR}/templates/README_WEIGHTS.md" \
    "${PROJECT_ROOT}/weights/README_WEIGHTS.md"
fi
if [[ ! -e "${PROJECT_ROOT}/input/README.md" ]]; then
  install -m 0644 "${SCRIPT_DIR}/templates/README_INPUT.md" \
    "${PROJECT_ROOT}/input/README.md"
fi
if [[ ! -e "${PROJECT_ROOT}/qgis/README.md" ]]; then
  install -m 0644 "${SCRIPT_DIR}/templates/README_QGIS.md" \
    "${PROJECT_ROOT}/qgis/README.md"
fi

echo "Initialized Loess project only: ${PROJECT_ROOT}"
echo "  platform: ${PLATFORM}"
echo "  Git SHA: ${GIT_SHA}"
echo "  inference scripts: ${PROJECT_ROOT}/inference_scripts"
echo "  shared runtime: ${PROJECT_ROOT}/runtime/labeling_tool/core"
echo "  weights are not included; see weights/README_WEIGHTS.md"

if [[ "${CHECK_ASSETS}" -eq 1 ]]; then
  check_assets
else
  if ! check_assets; then
    echo "Project initialization is complete; weight assets are still pending." >&2
  fi
fi
