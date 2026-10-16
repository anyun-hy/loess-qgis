"""PostgreSQL regressions for ordered V3.3 publication input locks."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg2 import Error as DatabaseError
from psycopg2.errors import LockNotAvailable

from labeling_tool.runs.run_control_graph import _fragmentation_v33_units
from labeling_tool.shared.state import run_state_db as run_state_db_module
from labeling_tool.shared.state.run_state_db import RunStateDB, RunStateError
from labeling_tool.shared.state.run_state_session import RunStateSession

RUN_ID = "20261014_000000_v33_locking"
STREAM_ID = "fusion:v33-locking"


class _BoundedRunStateSession(RunStateSession):
    """Bound concurrent regressions even if a lock-order regression returns."""

    def connect(self, *, autocommit: bool = True):
        connection = super().connect(autocommit=autocommit)
        if not autocommit:
            connection.execute("SET LOCAL lock_timeout = '5s'")
            connection.execute("SET LOCAL statement_timeout = '10s'")
        return connection


@dataclass(frozen=True)
class _PublicationState:
    jobs_by_partition: dict[str, dict[str, Any]]
    finalize_job_id: int
    input_ids: tuple[int, ...]
    baseline_ids: dict[str, int]


def _bounded_database(database: RunStateDB) -> RunStateDB:
    session = _BoundedRunStateSession(
        database.session.location,
        schema=database.session.schema,
    )
    return RunStateDB._from_session(session)


def _write(path: Path, payload: bytes) -> tuple[int, str]:
    path.write_bytes(payload)
    return path.stat().st_size, hashlib.sha256(payload).hexdigest()


def _build_adjacent_publication_state(
    database: RunStateDB,
    tmp_path: Path,
) -> _PublicationState:
    database.run_streams.create_run(RUN_ID, "a" * 64, status="running")
    database.run_streams.register_streams(
        RUN_ID,
        [{"stream_id": STREAM_ID, "kind": "fusion", "profile_id": "locking"}],
    )
    packages = [
        {
            "package_id": f"package_{index:05d}",
            "sequence_no": index,
            "status": "ready",
        }
        for index in range(2)
    ]
    database.control_graph.insert_work_packages(RUN_ID, packages)
    partitions = [
        {
            "partition_id": f"partition_00000_{index:05d}",
            "row": 0,
            "col": index,
            "core_window": {
                "x0": index * 512,
                "y0": 0,
                "x1": (index + 1) * 512,
                "y1": 512,
            },
            "halo_window": {
                "x0": max(0, index * 512 - 256),
                "y0": 0,
                "x1": min(1024, (index + 1) * 512 + 256),
                "y1": 512,
            },
            "package_id": packages[index]["package_id"],
            "status": "ready",
        }
        for index in range(2)
    ]
    database.control_graph.insert_partitions(RUN_ID, partitions)
    units = _fragmentation_v33_units(partitions)
    partition_units = [
        unit for unit in units if unit["unit_type"] == "FragmentationV33Partition"
    ]
    assert len(partition_units) == 2
    assert all(
        set(unit["dependency_ids"])
        == {str(partition["partition_id"]) for partition in partitions}
        for unit in partition_units
    )
    database.control_graph.insert_spatial_units(RUN_ID, units)
    database.jobs.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                "unit_id": unit["unit_id"],
                "priority": 50
                if unit["unit_type"] == "FragmentationV33Partition"
                else 40,
                "max_attempts": 3,
            }
            for unit in units
        ],
    )

    input_ids: list[int] = []
    baseline_ids: dict[str, int] = {}
    for index, partition in enumerate(partitions):
        partition_id = str(partition["partition_id"])
        probability = tmp_path / f"{partition_id}_probability.tif"
        context = tmp_path / f"{partition_id}_context.tif"
        baseline = tmp_path / f"{partition_id}_baseline.tif"
        probability_size, probability_digest = _write(
            probability, f"probability-{index}".encode()
        )
        context_size, context_digest = _write(context, f"context-{index}".encode())
        baseline_size, baseline_digest = _write(baseline, f"baseline-{index}".encode())
        input_ids.append(
            database.artifacts.publish_partition_artifact(
                RUN_ID,
                STREAM_ID,
                partition_id,
                probability,
                byte_count=probability_size,
                sha256=probability_digest,
            )
        )
        input_ids.append(
            database.artifacts.publish_fragmentation_v33_context(
                RUN_ID,
                STREAM_ID,
                partition_id,
                context,
                byte_count=context_size,
                sha256=context_digest,
            )
        )
        baseline_id = database.artifacts.publish_fragmentation_v33_baseline_core(
            RUN_ID,
            STREAM_ID,
            partition_id,
            baseline,
            byte_count=baseline_size,
            sha256=baseline_digest,
        )
        input_ids.append(baseline_id)
        baseline_ids[partition_id] = baseline_id

    leases = [
        database.jobs.lease_next_fragmentation_v33(
            RUN_ID,
            f"locking-worker-{index}",
            lease_seconds=120,
            max_running=4,
        )
        for index in range(2)
    ]
    assert all(lease is not None for lease in leases)
    jobs_by_partition: dict[str, dict[str, Any]] = {}
    for lease in leases:
        assert lease is not None
        unit = database.control_graph.get_spatial_unit(RUN_ID, str(lease["unit_id"]))
        assert unit is not None
        assert unit["unit_type"] == "FragmentationV33Partition"
        jobs_by_partition[str(unit["owner_key"])] = lease

    with database.session.connection() as connection:
        finalize_job_id = int(
            connection.execute(
                """SELECT j.job_id FROM jobs j
                   JOIN spatial_units u ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.stream_id=%s
                     AND j.job_type='fragmentation_v33'
                     AND u.unit_type='FragmentationV33Finalize'""",
                (RUN_ID, STREAM_ID),
            ).fetchone()[0]
        )
        dependency_rows = connection.execute(
            """SELECT job_id, artifact_id FROM artifact_dependencies
               WHERE job_id IN (%s, %s) ORDER BY job_id, artifact_id""",
            tuple(int(job["job_id"]) for job in jobs_by_partition.values()),
        ).fetchall()
    assert len(dependency_rows) == 12
    assert len(input_ids) == 6
    assert all(
        int(database.artifacts.get_artifact(artifact_id)["ref_count"]) == 2
        for artifact_id in input_ids
    )
    return _PublicationState(
        jobs_by_partition=jobs_by_partition,
        finalize_job_id=finalize_job_id,
        input_ids=tuple(sorted(input_ids)),
        baseline_ids=baseline_ids,
    )


def _output_arguments(tmp_path: Path, partition_id: str) -> dict[str, Any]:
    mask = tmp_path / f"{partition_id}_staged_mask.tif"
    audit = tmp_path / f"{partition_id}_staged_audit.json"
    mask_size, mask_digest = _write(mask, f"mask:{partition_id}".encode())
    audit_size, audit_digest = _write(audit, f"audit:{partition_id}".encode())
    return {
        "mask_path": mask,
        "mask_byte_count": mask_size,
        "mask_sha256": mask_digest,
        "audit_path": audit,
        "audit_byte_count": audit_size,
        "audit_sha256": audit_digest,
        "production": None,
    }


def _artifact_rows(database: RunStateDB) -> list[tuple[Any, ...]]:
    with database.session.connection() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                """SELECT artifact_id, unit_id, kind, status, ref_count
                   FROM artifacts WHERE run_id=%s ORDER BY artifact_id""",
                (RUN_ID,),
            ).fetchall()
        ]


def _dependency_rows(database: RunStateDB) -> list[tuple[int, int]]:
    with database.session.connection() as connection:
        return [
            (int(row["job_id"]), int(row["artifact_id"]))
            for row in connection.execute(
                """SELECT job_id, artifact_id FROM artifact_dependencies
                   ORDER BY job_id, artifact_id"""
            ).fetchall()
        ]


def _assert_rows_locked(database: RunStateDB, artifact_ids: tuple[int, ...]) -> None:
    for artifact_id in artifact_ids:
        connection = database.session.connect(autocommit=False)
        try:
            with pytest.raises(LockNotAvailable):
                connection.execute(
                    "SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE NOWAIT",
                    (artifact_id,),
                ).fetchone()
        finally:
            connection.rollback()
            connection.close()


def _assert_rows_lockable(database: RunStateDB, artifact_ids: tuple[int, ...]) -> None:
    connection = database.session.connect(autocommit=False)
    try:
        locked = connection.execute(
            """SELECT artifact_id FROM artifacts
               WHERE artifact_id = ANY(%s) ORDER BY artifact_id FOR UPDATE NOWAIT""",
            (list(artifact_ids),),
        ).fetchall()
        assert [int(row["artifact_id"]) for row in locked] == list(artifact_ids)
    finally:
        connection.rollback()
        connection.close()


def _row_is_locked_nowait(database: RunStateDB, table: str, row_id: int) -> bool:
    assert table in {"artifacts", "jobs"}
    id_column = "artifact_id" if table == "artifacts" else "job_id"
    connection = database.session.connect(autocommit=False)
    try:
        connection.execute(
            f"SELECT {id_column} FROM {table} WHERE {id_column}=%s FOR UPDATE NOWAIT",
            (int(row_id),),
        ).fetchone()
        return False
    except LockNotAvailable:
        return True
    finally:
        connection.rollback()
        connection.close()


def test_attempt_scope_locks_every_shared_input_before_yield(
    tmp_path,
    postgres_database,
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    job = next(iter(state.jobs_by_partition.values()))
    with postgres_database.session.connection() as connection:
        input_kinds = {
            str(row["kind"])
            for row in connection.execute(
                """SELECT a.kind FROM artifact_dependencies ad
                   JOIN artifacts a ON a.artifact_id=ad.artifact_id
                   WHERE ad.job_id=%s""",
                (int(job["job_id"]),),
            ).fetchall()
        }
    assert input_kinds == {
        "partition_probability",
        "v3_context_core",
        "v3_baseline_core",
    }
    assert len(state.baseline_ids) == 2
    assert set(state.baseline_ids.values()).issubset(state.input_ids)

    with postgres_database.fragmentation_v33_attempt_commit(
        int(job["job_id"]), str(job["lease_token"])
    ):
        _assert_rows_locked(postgres_database, state.input_ids)

    _assert_rows_lockable(postgres_database, state.input_ids)


def test_attempt_scope_rechecks_lease_after_waiting_for_input_lock(
    tmp_path,
    postgres_database,
    monkeypatch,
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    job = next(iter(state.jobs_by_partition.values()))
    job_id = int(job["job_id"])
    artifacts_before = _artifact_rows(postgres_database)
    dependencies_before = _dependency_rows(postgres_database)
    job_before = postgres_database.jobs.get_job(job_id)
    assert job_before is not None

    blocker = postgres_database.session.connect(autocommit=False)
    blocker.execute(
        "SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE",
        (state.input_ids[0],),
    ).fetchone()
    yielded = threading.Event()

    def enter_attempt_scope() -> None:
        database = _bounded_database(postgres_database)
        with database.fragmentation_v33_attempt_commit(job_id, str(job["lease_token"])):
            yielded.set()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(enter_attempt_scope)
    blocker_released = False
    try:
        deadline = time.monotonic() + 3.0
        while not _row_is_locked_nowait(postgres_database, "jobs", job_id):
            if future.done():
                future.result()
                raise AssertionError(
                    "publication scope returned before input lock wait"
                )
            assert time.monotonic() < deadline, (
                "worker did not acquire the Job row lock"
            )
            threading.Event().wait(0.01)

        expired_time = float(job_before["lease_expires"]) + 1.0
        with monkeypatch.context() as patch:
            patch.setattr(
                run_state_db_module,
                "time",
                SimpleNamespace(time=lambda: expired_time),
            )
            blocker.rollback()
            blocker_released = True
            with pytest.raises(
                RunStateError,
                match="lease expired while waiting for artifact inputs",
            ):
                future.result(timeout=10)
    finally:
        if not blocker_released:
            blocker.rollback()
        blocker.close()
        executor.shutdown(wait=True, cancel_futures=True)

    assert not yielded.is_set()
    assert postgres_database.jobs.get_job(job_id) == job_before
    assert _artifact_rows(postgres_database) == artifacts_before
    assert _dependency_rows(postgres_database) == dependencies_before
    _assert_rows_lockable(postgres_database, state.input_ids)


def test_adjacent_partitions_publish_and_complete_concurrently(
    tmp_path,
    postgres_database,
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    outputs = {
        partition_id: _output_arguments(tmp_path, partition_id)
        for partition_id in state.jobs_by_partition
    }
    start = threading.Event()

    def publish(partition_id: str) -> tuple[int, int]:
        database = _bounded_database(postgres_database)
        job = state.jobs_by_partition[partition_id]
        assert start.wait(timeout=5)
        with database.fragmentation_v33_attempt_commit(
            int(job["job_id"]), str(job["lease_token"])
        ) as publication:
            staged_ids = publication.artifacts.publish_fragmentation_v33_output_pair(
                RUN_ID,
                STREAM_ID,
                partition_id,
                **outputs[partition_id],
            )
            assert publication.jobs.complete_fragmentation_v33_job(
                int(job["job_id"]), str(job["lease_token"])
            )
        return staged_ids

    executor = ThreadPoolExecutor(max_workers=2)
    futures = [executor.submit(publish, partition_id) for partition_id in outputs]
    start.set()
    try:
        done, pending = wait(futures, timeout=20)
        assert not pending, "concurrent V3.3 publication exceeded bounded timeout"
        staged_ids = [artifact_id for future in done for artifact_id in future.result()]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert len(staged_ids) == 4
    assert all(
        postgres_database.jobs.get_job(int(job["job_id"]))["status"] == "ready"
        for job in state.jobs_by_partition.values()
    )
    for artifact_id in state.input_ids:
        artifact = postgres_database.artifacts.get_artifact(artifact_id)
        assert artifact is not None
        expected = 1 if artifact["kind"] == "v3_baseline_core" else 0
        assert artifact["status"] == "ready"
        assert int(artifact["ref_count"]) == expected
    assert all(
        postgres_database.artifacts.get_artifact(artifact_id)["status"] == "ready"
        and int(postgres_database.artifacts.get_artifact(artifact_id)["ref_count"]) == 1
        for artifact_id in staged_ids
    )
    with postgres_database.session.connection() as connection:
        finalize_dependencies = connection.execute(
            """SELECT a.kind, a.unit_id FROM artifact_dependencies ad
               JOIN artifacts a ON a.artifact_id=ad.artifact_id
               WHERE ad.job_id=%s ORDER BY a.artifact_id""",
            (state.finalize_job_id,),
        ).fetchall()
    assert len(finalize_dependencies) == 6
    assert [str(row["kind"]) for row in finalize_dependencies].count(
        "v3_baseline_core"
    ) == 2
    assert {str(row["unit_id"]) for row in finalize_dependencies} == set(outputs)


@pytest.mark.parametrize("failure_point", ["publication", "completion"])
def test_publication_attempt_failure_rolls_back_state_and_releases_locks(
    tmp_path,
    postgres_database,
    failure_point,
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    partition_id, job = next(iter(state.jobs_by_partition.items()))
    outputs = _output_arguments(tmp_path, partition_id)
    artifacts_before = _artifact_rows(postgres_database)
    dependencies_before = _dependency_rows(postgres_database)
    job_before = postgres_database.jobs.get_job(int(job["job_id"]))
    assert job_before is not None

    with postgres_database.session.connection() as connection:
        if failure_point == "publication":
            connection.execute(
                f"""CREATE FUNCTION reject_v33_publication_dependency()
                    RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                      IF NEW.job_id={state.finalize_job_id} THEN
                        RAISE EXCEPTION 'injected V3.3 publication failure';
                      END IF;
                      RETURN NEW;
                    END;
                    $$"""
            )
            connection.execute(
                """CREATE TRIGGER reject_v33_publication_dependency
                   BEFORE INSERT ON artifact_dependencies FOR EACH ROW
                   EXECUTE FUNCTION reject_v33_publication_dependency()"""
            )
        else:
            connection.execute(
                f"""CREATE FUNCTION reject_v33_completion_dependency()
                    RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                      IF OLD.job_id={int(job["job_id"])} THEN
                        RAISE EXCEPTION 'injected V3.3 completion failure';
                      END IF;
                      RETURN OLD;
                    END;
                    $$"""
            )
            connection.execute(
                """CREATE TRIGGER reject_v33_completion_dependency
                   BEFORE DELETE ON artifact_dependencies FOR EACH ROW
                   EXECUTE FUNCTION reject_v33_completion_dependency()"""
            )

    with pytest.raises(DatabaseError, match=f"injected V3.3 {failure_point} failure"):
        with postgres_database.fragmentation_v33_attempt_commit(
            int(job["job_id"]), str(job["lease_token"])
        ) as publication:
            publication.artifacts.publish_fragmentation_v33_output_pair(
                RUN_ID,
                STREAM_ID,
                partition_id,
                **outputs,
            )
            if failure_point == "completion":
                publication.jobs.complete_fragmentation_v33_job(
                    int(job["job_id"]), str(job["lease_token"])
                )

    assert _artifact_rows(postgres_database) == artifacts_before
    assert _dependency_rows(postgres_database) == dependencies_before
    job_after = postgres_database.jobs.get_job(int(job["job_id"]))
    assert job_after is not None
    assert job_after["status"] == job_before["status"] == "running"
    assert job_after["lease_token"] == job_before["lease_token"]
    assert job_after["progress_current"] == job_before["progress_current"]
    assert job_after["progress_total"] == job_before["progress_total"]
    assert not [
        row
        for row in _artifact_rows(postgres_database)
        if row[2] in {"v33_staged_mask", "v33_staged_audit"}
    ]
    assert not [row for row in dependencies_before if row[0] == state.finalize_job_id]
    assert all(
        int(postgres_database.artifacts.get_artifact(artifact_id)["ref_count"]) == 2
        for artifact_id in state.input_ids
    )
    _assert_rows_lockable(postgres_database, state.input_ids)
