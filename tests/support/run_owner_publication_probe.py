"""Hold one owner-fenced publication while its parent lock backend is killed."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from labeling_tool.shared.state.run_execution_ownership import RUN_OWNER_ENV
from labeling_tool.shared.state.run_state_db import RunStateDB


def main(argv: list[str]) -> int:
    if len(argv) != 9:
        return 2
    (
        _program,
        dsn,
        schema,
        run_id,
        run_dir,
        owner_json,
        ready_path,
        release_path,
        output_path,
    ) = argv
    os.environ[RUN_OWNER_ENV] = owner_json
    os.environ["LOESS_MONITOR_EXECUTION_ID"] = __import__("json").loads(owner_json)[
        "execution_id"
    ]
    database = RunStateDB(dsn, postgres_schema=schema)
    with database.owner_publication(run_id, run_dir) as publication:
        Path(ready_path).write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 15.0
        while not Path(release_path).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("publication probe release was not signaled")
            time.sleep(0.02)
        Path(output_path).write_text("old-owner-publication", encoding="utf-8")
        publication.run_streams.append_event(
            run_id,
            "old_owner_publication_completed",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
