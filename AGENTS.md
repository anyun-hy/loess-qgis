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

