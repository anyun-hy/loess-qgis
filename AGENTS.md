# Repository Operating Boundary

## Authoritative source

`/Users/example/Desktop/loess-qgis` is the only current authoritative source for
the merged macOS and Ubuntu implementation. It contains the shared QGIS plugin,
inference runtime, Bash deployment scripts, tests, documentation, and curated
project inputs.

The platform implementation is intentionally unified:

- Ubuntu: QGIS 3.44, Qt5/PyQt5, CUDA/RTX 3090.
- macOS: QGIS 4.2, Qt6/PyQt6, MPS.
- `bash/install_plugin.sh` installs the shared plugin for the selected platform.
- `bash/init_project.sh` initializes or updates the platform deployment project.

## Test environment

Run project tests in the Conda environment named `qgis`. Use
`conda run -n qgis <test command>` or activate `qgis` before running tests; do
not use system Python or the Conda `base` environment for project test results.

## Remote Ubuntu access

`Tencent` is the project's Ubuntu runtime and validation host. When a task
requires Ubuntu, QGIS 3.44, Qt5, CUDA/RTX 3090, remote logs, or live runtime
evidence, invoke `bash/ssh_tencent.sh <command>` directly; do not wait for the
user to start an interactive `ssh Tencent` session.

The project wrapper reuses an SSH connection for a short idle window and then
closes it automatically. The local `loess-qgis` checkout remains the
authoritative source. Remote writes, synchronization, deployment, and changes
to user-controlled data still require authorization from the current task and
must preserve the deployment-project boundary below.

## Prohibited legacy lookup by default

`/Users/example/Desktop/loess-data` is a legacy/original-data workspace that
predates the consolidation into `loess-qgis`. Do not search, inspect, read,
compare, copy from, or modify `loess-data` unless the user explicitly names that
directory and authorizes the specific lookup or operation in the current task.
Do not use facts, scripts, status notes, or historical snapshots from
`loess-data` to override the current `loess-qgis` repository.

## Deployment project

`loess-project` is a generated/runtime project, not the source repository.
Managed plugin and inference files must originate from `loess-qgis` through the
Bash deployment entry points. Preserve user-controlled weights, inputs, QGIS
projects, accepted labels, and outputs unless the user explicitly requests a
specific change.

