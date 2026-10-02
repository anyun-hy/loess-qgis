"""Try to acquire one Run owner from an independent Python process."""

from __future__ import annotations

import sys

from labeling_tool.shared.state.run_execution_ownership import (
    RunExecutionOwnership,
    RunOwnershipConflictError,
)
from labeling_tool.shared.state.run_state_db import RunStateDB


def main(argv: list[str]) -> int:
    dsn, schema, run_id, run_dir = argv[1:5]
    database = RunStateDB(dsn, postgres_schema=schema)
    try:
        owner = RunExecutionOwnership.acquire(
            database,
            run_id,
            run_dir=run_dir,
            worker_id="independent-owner-probe",
            trigger_type="resume",
        )
    except RunOwnershipConflictError as error:
        print(str(error))
        return 23
    owner.close()
    print("unexpectedly acquired Run owner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
