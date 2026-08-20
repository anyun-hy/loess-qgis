#!/usr/bin/env bash
set -euo pipefail

readonly SSH_HOST="Tencent"
readonly CONTROL_PERSIST="${LOESS_SSH_CONTROL_PERSIST:-15m}"
readonly CONTROL_DIR="${LOESS_SSH_CONTROL_DIR:-/tmp/loess-qgis-ssh-${UID}}"
readonly SSH_BIN="${SSH_BIN:-$(command -v ssh 2>/dev/null || true)}"

usage() {
  cat <<'EOF'
Usage: ./bash/ssh_tencent.sh [remote command [arguments...]]

Connect to the project's Tencent Ubuntu host. Consecutive invocations reuse one
SSH connection; the connection closes automatically after 15 idle minutes.

Environment overrides:
  LOESS_SSH_CONTROL_PERSIST  OpenSSH ControlPersist duration. Default: 15m
  LOESS_SSH_CONTROL_DIR      Directory for the user-owned control socket
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

[[ -n "${SSH_BIN}" && -x "${SSH_BIN}" ]] || {
  echo "OpenSSH client not found; set SSH_BIN to an executable" >&2
  exit 1
}

if [[ -L "${CONTROL_DIR}" ]]; then
  echo "Refusing symlink SSH control directory: ${CONTROL_DIR}" >&2
  exit 1
fi
mkdir -p -- "${CONTROL_DIR}"
chmod 700 "${CONTROL_DIR}"

exec "${SSH_BIN}" \
  -o ControlMaster=auto \
  -o "ControlPersist=${CONTROL_PERSIST}" \
  -o "ControlPath=${CONTROL_DIR}/%C" \
  "${SSH_HOST}" "$@"
