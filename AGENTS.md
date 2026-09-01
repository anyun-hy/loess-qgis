# Repository Operating Boundary

## Authoritative source

The repository checkout is the authoritative source for the merged macOS and
Ubuntu implementation. It contains the shared QGIS plugin,
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

## Waiting tasks

When a task is waiting only for time or an external process, do not keep the
active conversation open with repeated polling or unchanged status messages.
Use an available background wait or monitoring mechanism; otherwise return
control to the user with the current state and the exact condition for
resuming, then continue only after notification or a new request.

## Subagent model selection

When subagents are authorized, choose each subagent's model and reasoning
effort from the task's complexity, ambiguity, risk, and verification burden;
do not use one fixed setting for every task.

- Use `gpt-5.6-luna` with low or medium reasoning for bounded, low-risk work
  such as file discovery, evidence collection, mechanical checks, and simple
  test execution.
- Use `gpt-5.6-terra` with medium or high reasoning as the normal baseline for
  scoped implementation, tests, and reviews with clear acceptance criteria.
- Use `gpt-5.6-sol` with high or xhigh reasoning for difficult root-cause
  analysis, cross-platform architecture, concurrency or data-integrity work,
  security-sensitive review, and other high-risk or highly ambiguous tasks.

Use the least expensive tier that can reliably complete the work. Increase
capability or reasoning when evidence shows the initial tier is insufficient;
do not lower it merely to save tokens when correctness risk remains. Record
the selected model, reasoning effort, task boundary, and expected evidence in
the subagent task card. If a named model is unavailable, use the closest
available model in the same capability tier.

## Session handoff documents

Do not create or update session-handoff documents for routine code changes.
Handoff records are allowed only when the user explicitly requests a new
conversation or handoff, work is actually being transferred to another
session, or the current context can no longer safely carry the unfinished
task. Otherwise rely on the live code, `git diff`, `git status`, and test
evidence. Update formal implementation-status documents only when their own
project rules require it, not as a substitute for a session handoff.

## Documentation discipline

`docs/README.md` is the only documentation entrypoint. Keep current contracts
in `docs/ARCHITECTURE.md`, current verified state in `docs/CURRENT_STATUS.md`,
formal operating methods in `docs/operations/`, and durable choices in
`docs/decisions/`. Internal session logs, handoffs, and historical runtime
evidence are not public documentation and must not be added to the repository.

Routine code changes do not require documentation edits. Do not append dated
work logs, test transcripts, or session summaries to current documents. Update
current docs only when an architecture contract, verified capability,
deployment state, blocker, operating method, or durable project decision has
materially changed.

## Remote Ubuntu access

An Ubuntu validation host may be configured through the local SSH alias passed
as `LOESS_SSH_HOST`. When a task requires Ubuntu, QGIS 3.44, Qt5, CUDA/RTX 3090,
remote logs, or live runtime evidence, invoke `bash/ssh_tencent.sh <command>`
directly; do not wait for the user to start an interactive SSH session.

The project wrapper reuses an SSH connection for a short idle window and then
closes it automatically. The local `loess-qgis` checkout remains the
authoritative source. Remote writes, synchronization, deployment, and changes
to user-controlled data still require authorization from the current task and
must preserve the deployment-project boundary below.

## Prohibited legacy lookup by default

A legacy/original-data workspace may exist outside this repository. Do not
search, inspect, compare, copy from, or modify an external legacy workspace
unless the user explicitly names it and authorizes the specific operation.
External history must not override the current repository.

## Deployment project

`loess-project` is a generated/runtime project, not the source repository.
Managed plugin and inference files must originate from `loess-qgis` through the
Bash deployment entry points. Preserve user-controlled weights, inputs, QGIS
projects, accepted labels, and outputs unless the user explicitly requests a
specific change.
