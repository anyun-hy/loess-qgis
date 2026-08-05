#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${LOESS_PLATFORM:-auto}"
PROJECT_ROOT="${LOESS_PROJECT_ROOT:-}"
CONDA_EXE="${CONDA_EXE:-}"
ENV_NAME="${CONDA_ENV:-qgis}"
CONDA_EXE_EXPLICIT=0
CONDA_ENV_EXPLICIT=0
CREATE_ENV=0
CHECK_ONLY=0
CHECK_ASSETS=0
ALLOW_DIRTY=0
REBIND_PROJECT_ROOT=0

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
  --conda-env NAME     Override the Conda environment name. Default: qgis
  --create-env         Create or update the platform inference environment
  --check-only         Validate the source and target without writing
  --check-assets       Fail unless required weight assets exist and hashes match
  --rebind-project-root
                       Explicitly accept a moved project whose identity marker
                       still matches its manifest
  --allow-dirty        Allow a development deployment from modified source;
                       the manifest records the actual source bundle SHA256
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
      CONDA_EXE_EXPLICIT=1
      shift 2
      ;;
    --conda-env)
      ENV_NAME="${2:?Missing value for --conda-env}"
      CONDA_ENV_EXPLICIT=1
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
    --rebind-project-root)
      REBIND_PROJECT_ROOT=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
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

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid Conda environment name: ${ENV_NAME}" >&2
  exit 2
}

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
identity_args=(--project-root "${PROJECT_ROOT}")
if [[ "${REBIND_PROJECT_ROOT}" -eq 1 ]]; then
  identity_args+=(--allow-rebind)
fi
PROJECT_IDENTITY="$("${PYTHON_BIN}" "${SCRIPT_DIR}/project_identity.py" "${identity_args[@]}")"
PROJECT_ID="$("${PYTHON_BIN}" -c '
import json
import sys
print(json.loads(sys.argv[1])["project_id"])
' "${PROJECT_IDENTITY}")"
CREATE_IDENTITY="$("${PYTHON_BIN}" -c '
import json
import sys
print("1" if json.loads(sys.argv[1])["create_identity"] else "0")
' "${PROJECT_IDENTITY}")"
if [[ "${CONDA_EXE_EXPLICIT}" -eq 0 ]]; then
  CONDA_EXE="$("${PYTHON_BIN}" -c '
import json
import sys
print(json.loads(sys.argv[1])["conda_executable"])
' "${PROJECT_IDENTITY}")"
fi
if [[ "${CONDA_ENV_EXPLICIT}" -eq 0 ]]; then
  saved_environment="$("${PYTHON_BIN}" -c '
import json
import sys
print(json.loads(sys.argv[1])["conda_environment"])
' "${PROJECT_IDENTITY}")"
  if [[ -n "${saved_environment}" ]]; then
    ENV_NAME="${saved_environment}"
  fi
fi

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid persisted Conda environment name: ${ENV_NAME}" >&2
  exit 2
}
if [[ -n "${CONDA_EXE}" ]]; then
  CONDA_EXE="$("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
' "${CONDA_EXE}")"
  [[ -x "${CONDA_EXE}" ]] || {
    echo "Configured Conda executable is not executable: ${CONDA_EXE}" >&2
    exit 1
  }
fi

source_args=(inspect --source-root "${SOURCE_ROOT}")
if [[ "${ALLOW_DIRTY}" -eq 1 || "${LOESS_ALLOW_DIRTY:-0}" == "1" ]]; then
  source_args+=(--allow-dirty)
fi
SOURCE_INFO="$("${PYTHON_BIN}" "${SCRIPT_DIR}/deployment_source.py" "${source_args[@]}")"
GIT_SHA="$("${PYTHON_BIN}" -c '
import json
import sys
print(json.loads(sys.argv[1])["git_sha"])
' "${SOURCE_INFO}")"

check_assets() {
  "${PYTHON_BIN}" -c '
import hashlib
import json
import re
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
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        mismatched.append(asset["path"] + " (missing trusted SHA256)")
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
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
expected_platform = sys.argv[5]
expected_conda_exe = sys.argv[6]
expected_conda_env = sys.argv[7]
expected_source = json.loads(sys.argv[8])
expected_project_id = sys.argv[9]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if manifest.get("schema_version") != 2:
    raise SystemExit("Project manifest must use schema 2")
if manifest.get("project_id") != expected_project_id:
    raise SystemExit("Project identity differs from the validated identity marker")
actual_git = manifest.get("git_sha")
if actual_git != expected_git:
    raise SystemExit(
        "Project Git SHA differs from this source; rerun init_project.sh "
        f"to update it ({actual_git} != {expected_git})"
    )
if Path(str(manifest.get("project_root") or "")).resolve() != root:
    raise SystemExit("Project was moved; rerun init_project.sh at its current path")
if manifest.get("platform") != expected_platform:
    raise SystemExit("Project platform differs; rerun init_project.sh")
if manifest.get("source") != expected_source:
    raise SystemExit(
        "Project source bundle differs from this source; rerun init_project.sh"
    )

launcher = manifest.get("launcher") or {}
launcher_path = root / "runtime" / "loess_launcher.sh"
if launcher.get("path") != "runtime/loess_launcher.sh":
    raise SystemExit("Project launcher path is invalid")
if launcher.get("conda_executable") != expected_conda_exe:
    raise SystemExit("Project Conda executable differs; rerun init_project.sh")
if launcher.get("conda_environment") != expected_conda_env:
    raise SystemExit("Project Conda environment differs; rerun init_project.sh")
if not launcher_path.is_file() or launcher_path.is_symlink():
    raise SystemExit(f"Project launcher is missing or unsafe: {launcher_path}")
launcher_digest = hashlib.sha256(launcher_path.read_bytes()).hexdigest()
if launcher_digest != launcher.get("sha256"):
    raise SystemExit("Project launcher SHA256 mismatch")

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
' "${MANIFEST}" "${PROJECT_ROOT}" "${SOURCE_ROOT}" "${GIT_SHA}" \
    "${PLATFORM}" "${CONDA_EXE}" "${ENV_NAME}" "${SOURCE_INFO}" "${PROJECT_ID}"
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
  CONDA_EXE="$("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
' "${CONDA_EXE}")"
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
OLD_MANIFEST="${PROJECT_ROOT}/.project_manifest.json.old.$$"
DEPLOY_COMMITTED=0
NEW_INFERENCE=0
NEW_RUNTIME=0
NEW_MANIFEST=0
NEW_IDENTITY=0
IN_CLEANUP=0
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  if [[ "${IN_CLEANUP}" -eq 1 ]]; then
    exit "${exit_code}"
  fi
  IN_CLEANUP=1
  if [[ "${DEPLOY_COMMITTED}" -eq 0 ]]; then
    if [[ "${NEW_INFERENCE}" -eq 1 && -e "${PROJECT_ROOT}/inference_scripts" ]]; then
      rm -rf -- "${PROJECT_ROOT}/inference_scripts"
    fi
    if [[ "${NEW_RUNTIME}" -eq 1 && -e "${PROJECT_ROOT}/runtime" ]]; then
      rm -rf -- "${PROJECT_ROOT}/runtime"
    fi
    if [[ "${NEW_MANIFEST}" -eq 1 && -e "${MANIFEST}" ]]; then
      rm -f -- "${MANIFEST}"
    fi
    if [[ "${NEW_IDENTITY}" -eq 1 && -e "${PROJECT_ROOT}/.loess-project-id" ]]; then
      rm -f -- "${PROJECT_ROOT}/.loess-project-id"
    fi
    if [[ -e "${OLD_INFERENCE}" ]]; then
      mv "${OLD_INFERENCE}" "${PROJECT_ROOT}/inference_scripts"
    fi
    if [[ -e "${OLD_RUNTIME}" ]]; then
      mv "${OLD_RUNTIME}" "${PROJECT_ROOT}/runtime"
    fi
    if [[ -e "${OLD_MANIFEST}" ]]; then
      mv "${OLD_MANIFEST}" "${MANIFEST}"
    fi
  else
    rm -rf -- "${OLD_INFERENCE}" "${OLD_RUNTIME}"
    rm -f -- "${OLD_MANIFEST}"
  fi
  if [[ -e "${STAGE_ROOT}" ]]; then
    rm -rf -- "${STAGE_ROOT}"
  fi
  exit "${exit_code}"
}
on_signal() {
  exit "$1"
}
fault_inject() {
  local stage="$1"
  [[ "${LOESS_TEST_SIGNAL_AT:-}" == "${stage}" ]] || return 0
  case "${LOESS_TEST_SIGNAL:-TERM}" in
    INT) kill -INT "$$" ;;
    TERM) kill -TERM "$$" ;;
    HUP) kill -HUP "$$" ;;
    *)
      echo "Invalid LOESS_TEST_SIGNAL: ${LOESS_TEST_SIGNAL}" >&2
      return 2
      ;;
  esac
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

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

SOURCE_INFO_AFTER="$("${PYTHON_BIN}" "${SCRIPT_DIR}/deployment_source.py" "${source_args[@]}")"
[[ "${SOURCE_INFO_AFTER}" == "${SOURCE_INFO}" ]] || {
  echo "Deployable source changed while project staging was in progress" >&2
  exit 1
}

PYTHONPYCACHEPREFIX="${STAGE_ROOT}/compile-cache" "${PYTHON_BIN}" -m compileall -q \
  "${STAGE_ROOT}/inference_scripts" "${STAGE_ROOT}/runtime"
rm -rf -- "${STAGE_ROOT}/compile-cache"

LOESS_STAGE_ROOT="${STAGE_ROOT}" \
LOESS_GIT_SHA="${GIT_SHA}" \
LOESS_SOURCE_INFO="${SOURCE_INFO}" \
LOESS_PLATFORM="${PLATFORM}" \
LOESS_PROJECT_ROOT="${PROJECT_ROOT}" \
LOESS_PROJECT_ID="${PROJECT_ID}" \
LOESS_CREATE_IDENTITY="${CREATE_IDENTITY}" \
LOESS_CONDA_EXE="${CONDA_EXE}" \
LOESS_CONDA_ENV="${ENV_NAME}" \
"${PYTHON_BIN}" -c '
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

stage = Path(os.environ["LOESS_STAGE_ROOT"])
config = (stage / "inference_scripts" / "config.yaml").read_text(encoding="utf-8")
launcher_path = stage / "runtime" / "loess_launcher.sh"
launcher_path.write_text(
    "# Generated by bash/init_project.sh; do not edit.\n"
    + "LOESS_CONFIGURED_PLATFORM="
    + shlex.quote(os.environ["LOESS_PLATFORM"])
    + "\nLOESS_CONFIGURED_CONDA_EXE="
    + shlex.quote(os.environ["LOESS_CONDA_EXE"])
    + "\nLOESS_CONFIGURED_CONDA_ENV="
    + shlex.quote(os.environ["LOESS_CONDA_ENV"])
    + "\n",
    encoding="utf-8",
)
launcher_sha256 = hashlib.sha256(launcher_path.read_bytes()).hexdigest()

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
pending_fusion = None
sam_asset = None
for raw_line in config.splitlines():
    line = raw_line.strip()
    if line == "semantic_models:":
        if pending_fusion:
            assets.append(pending_fusion)
            pending_fusion = None
        section = "models"
        continue
    if line == "fusion_profiles:":
        if pending_fusion:
            assets.append(pending_fusion)
            pending_fusion = None
        section = "fusion"
        continue
    if line == "sam3:":
        if pending_fusion:
            assets.append(pending_fusion)
            pending_fusion = None
        section = "sam3"
        continue
    if line and not raw_line.startswith((" ", "\t")):
        if pending_fusion:
            assets.append(pending_fusion)
            pending_fusion = None
        if sam_asset:
            assets.append(sam_asset)
            sam_asset = None
        section = ""
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
        if pending_fusion:
            assets.append(pending_fusion)
        value = line.split(":", 1)[1].strip().strip("\"'\''")
        pending_fusion = {
            "kind": "fusion_profile",
            "path": "weights/" + Path(value).name,
            "sha256": "",
        }
    elif section == "fusion" and pending_fusion and line.startswith("sha256:"):
        pending_fusion["sha256"] = line.split(":", 1)[1].strip().strip("\"'\''")
    elif section == "sam3" and line.startswith("checkpoint:"):
        value = line.split(":", 1)[1].strip().strip("\"'\''")
        sam_asset = {
            "kind": "sam3_checkpoint",
            "path": "weights/" + Path(value).name,
            "sha256": "",
        }
    elif section == "sam3" and sam_asset and line.startswith("sha256:"):
        sam_asset["sha256"] = line.split(":", 1)[1].strip().strip("\"'\''")

if pending_fusion:
    assets.append(pending_fusion)
if sam_asset:
    assets.append(sam_asset)
for asset in assets:
    if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256") or "")):
        raise SystemExit(
            f"Missing trusted SHA256 for required asset {asset.get('\''path'\'')}"
        )

payload = {
    "schema_version": 2,
    "deployment_kind": "loess_project",
    "project_id": os.environ["LOESS_PROJECT_ID"],
    "git_sha": os.environ["LOESS_GIT_SHA"],
    "source": json.loads(os.environ["LOESS_SOURCE_INFO"]),
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
    "launcher": {
        "path": "runtime/loess_launcher.sh",
        "sha256": launcher_sha256,
        "conda_executable": os.environ["LOESS_CONDA_EXE"],
        "conda_environment": os.environ["LOESS_CONDA_ENV"],
    },
    "inference_files": inference_files,
    "required_assets": assets,
}
(stage / "project_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if os.environ["LOESS_CREATE_IDENTITY"] == "1":
    identity = {
        "schema_version": 1,
        "deployment_kind": "loess_project_identity",
        "project_id": os.environ["LOESS_PROJECT_ID"],
    }
    (stage / ".loess-project-id").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
'

rm -rf -- "${OLD_INFERENCE}" "${OLD_RUNTIME}"
rm -f -- "${OLD_MANIFEST}"
if [[ -e "${PROJECT_ROOT}/inference_scripts" ]]; then
  mv "${PROJECT_ROOT}/inference_scripts" "${OLD_INFERENCE}"
fi
fault_inject previous_inference_moved
if [[ -e "${PROJECT_ROOT}/runtime" ]]; then
  mv "${PROJECT_ROOT}/runtime" "${OLD_RUNTIME}"
fi
fault_inject previous_runtime_moved
if [[ -e "${MANIFEST}" ]]; then
  mv "${MANIFEST}" "${OLD_MANIFEST}"
fi
fault_inject previous_manifest_moved

NEW_INFERENCE=1
mv "${STAGE_ROOT}/inference_scripts" "${PROJECT_ROOT}/inference_scripts"
fault_inject new_inference_moved
NEW_RUNTIME=1
mv "${STAGE_ROOT}/runtime" "${PROJECT_ROOT}/runtime"
fault_inject new_runtime_moved
NEW_MANIFEST=1
mv "${STAGE_ROOT}/project_manifest.json" "${MANIFEST}"
fault_inject new_manifest_moved
if [[ "${CREATE_IDENTITY}" -eq 1 ]]; then
  NEW_IDENTITY=1
  mv "${STAGE_ROOT}/.loess-project-id" "${PROJECT_ROOT}/.loess-project-id"
fi
fault_inject identity_installed

test -f "${PROJECT_ROOT}/inference_scripts/config.yaml"
test -f "${PROJECT_ROOT}/runtime/loess_launcher.sh"
test -f "${MANIFEST}"
test -f "${PROJECT_ROOT}/.loess-project-id"
"${PYTHON_BIN}" "${SCRIPT_DIR}/project_identity.py" \
  --project-root "${PROJECT_ROOT}" >/dev/null
fault_inject installation_verified
DEPLOY_COMMITTED=1
rm -rf -- "${OLD_INFERENCE}" "${OLD_RUNTIME}"
rm -f -- "${OLD_MANIFEST}"

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
echo "  project ID: ${PROJECT_ID}"
echo "  source bundle: $(
  "${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["source_bundle_sha256"])' \
    "${SOURCE_INFO}"
)"
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
