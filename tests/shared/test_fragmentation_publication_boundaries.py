"""PostgreSQL concurrency boundaries for current V3.3 publication."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tests.shared.test_fragmentation_publication_locking import (
    _output_arguments,
    _write,
)

from labeling_tool.runs.run_control_graph import _fragmentation_v33_units
from labeling_tool.shared.state.run_state_db import RunStateDB
from labeling_tool.shared.state.run_state_session import RunStateSession

RUN_ID = "20261015_000000_v33_boundaries"
STREAM_ID = "fusion:v33-boundaries"
PRODUCER_GATE = 8_610_021
TRANSFER_GATE = 8_610_022


class _BoundedObservedSession(RunStateSession):
    """Bound every worker transaction and expose its PostgreSQL backend PID."""

    def __init__(
        self,
        *args: Any,
        transaction_started: Callable[[int], None] = lambda _pid: None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transaction_started = transaction_started

    def connect(self, *, autocommit: bool = True):
        connection = super().connect(autocommit=autocommit)
        if not autocommit:
            connection.execute("SET LOCAL lock_timeout = '5s'")
            connection.execute("SET LOCAL statement_timeout = '10s'")
            pid = int(connection.execute("SELECT pg_backend_pid()").fetchone()[0])
            self._transaction_started(pid)
        return connection


@dataclass(frozen=True)
class _FarPublicationState:
    jobs_by_partition: dict[str, dict[str, Any]]
    finalize_job_id: int
    input_ids_by_partition: dict[str, tuple[int, ...]]


def _worker_database(
    database: RunStateDB,
    transaction_started: Callable[[int], None] = lambda _pid: None,
) -> RunStateDB:
    return RunStateDB._from_session(
        _BoundedObservedSession(
            database.session.location,
            schema=database.session.schema,
            transaction_started=transaction_started,
        )
    )


def _build_far_publication_state(
    database: RunStateDB,
    tmp_path: Path,
) -> _FarPublicationState:
    database.run_streams.create_run(RUN_ID, "a" * 64, status="running")
    database.run_streams.register_streams(
        RUN_ID,
        [{"stream_id": STREAM_ID, "kind": "fusion", "profile_id": "boundaries"}],
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
                "x0": index * 2048,
                "y0": 0,
                "x1": index * 2048 + 512,
                "y1": 512,
            },
            "halo_window": {
                "x0": max(0, index * 2048 - 256),
                "y0": 0,
                "x1": index * 2048 + 768,
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
    assert {
        str(unit["owner_key"]): tuple(unit["dependency_ids"])
        for unit in partition_units
    } == {
        str(partition["partition_id"]): (str(partition["partition_id"]),)
        for partition in partitions
    }
    database.control_graph.insert_spatial_units(RUN_ID, units)
    database.jobs.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                "unit_id": unit["unit_id"],
                "priority": (
                    50 if unit["unit_type"] == "FragmentationV33Partition" else 40
                ),
                "max_attempts": 3,
            }
            for unit in units
        ],
    )

    input_ids_by_partition: dict[str, tuple[int, ...]] = {}
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
        input_ids_by_partition[partition_id] = (
            database.artifacts.publish_partition_artifact(
                RUN_ID,
                STREAM_ID,
                partition_id,
                probability,
                byte_count=probability_size,
                sha256=probability_digest,
            ),
            database.artifacts.publish_fragmentation_v33_context(
                RUN_ID,
                STREAM_ID,
                partition_id,
                context,
                byte_count=context_size,
                sha256=context_digest,
            ),
            database.artifacts.publish_fragmentation_v33_baseline_core(
                RUN_ID,
                STREAM_ID,
                partition_id,
                baseline,
                byte_count=baseline_size,
                sha256=baseline_digest,
            ),
        )

    leases = [
        database.jobs.lease_next_fragmentation_v33(
            RUN_ID,
            f"boundary-worker-{index}",
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
        for partition_id, job in jobs_by_partition.items():
            dependency_ids = {
                int(row[0])
                for row in connection.execute(
                    """SELECT artifact_id FROM artifact_dependencies
                       WHERE job_id=%s ORDER BY artifact_id""",
                    (int(job["job_id"]),),
                ).fetchall()
            }
            assert dependency_ids == set(input_ids_by_partition[partition_id])
    first_ids, second_ids = input_ids_by_partition.values()
    assert set(first_ids).isdisjoint(second_ids)
    return _FarPublicationState(
        jobs_by_partition=jobs_by_partition,
        finalize_job_id=finalize_job_id,
        input_ids_by_partition=input_ids_by_partition,
    )


def _publish_and_complete(
    database: RunStateDB,
    partition_id: str,
    job: dict[str, Any],
    output: dict[str, Any],
    *,
    hold_open: threading.Event | None = None,
    transaction_open: threading.Event | None = None,
) -> None:
    with database.fragmentation_v33_attempt_commit(
        int(job["job_id"]), str(job["lease_token"])
    ) as publication:
        publication.artifacts.publish_fragmentation_v33_output_pair(
            RUN_ID, STREAM_ID, partition_id, **output
        )
        assert publication.jobs.complete_fragmentation_v33_job(
            int(job["job_id"]), str(job["lease_token"])
        )
        if transaction_open is not None:
            transaction_open.set()
        if hold_open is not None:
            assert hold_open.wait(timeout=8), "test did not release publication"


def _future_result(future: Future[Any], *, timeout: float = 8) -> Any:
    return future.result(timeout=timeout)


def _wait_for_lock(
    database: RunStateDB,
    *,
    waiting_pid: Callable[[], int | None],
    blocking_pid: int,
    query_fragment: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        pid = waiting_pid()
        if pid is not None:
            with database.session.connection() as connection:
                row = connection.execute(
                    """SELECT wait_event_type, wait_event, query,
                              pg_blocking_pids(pid) AS blocking_pids
                       FROM pg_stat_activity WHERE pid=%s""",
                    (pid,),
                ).fetchone()
            if row is not None:
                last = dict(row)
                if (
                    row["wait_event_type"] == "Lock"
                    and blocking_pid in row["blocking_pids"]
                    and query_fragment in str(row["query"])
                ):
                    return last
        threading.Event().wait(0.01)
    raise AssertionError(
        f"expected PostgreSQL lock wait was not observed; last activity={last!r}"
    )


def _create_insert_gate(database: RunStateDB, artifact_id: int) -> None:
    with database.session.connection() as connection:
        connection.execute(
            f"""CREATE FUNCTION boundary_pause_dependency_insert() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                  IF NEW.artifact_id = {int(artifact_id)} THEN
                    PERFORM pg_advisory_xact_lock(
                      hashtext(current_schema()), {PRODUCER_GATE}
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$"""
        )
        connection.execute(
            """CREATE TRIGGER boundary_pause_dependency_insert
               BEFORE INSERT ON artifact_dependencies FOR EACH ROW
               EXECUTE FUNCTION boundary_pause_dependency_insert()"""
        )


def _create_release_gate(database: RunStateDB, job_id: int, artifact_id: int) -> None:
    with database.session.connection() as connection:
        connection.execute(
            f"""CREATE FUNCTION boundary_pause_dependency_release() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                  IF OLD.job_id = {int(job_id)}
                     AND OLD.artifact_id = {int(artifact_id)} THEN
                    PERFORM pg_advisory_xact_lock(
                      hashtext(current_schema()), {TRANSFER_GATE}
                    );
                  END IF;
                  RETURN OLD;
                END;
                $$"""
        )
        connection.execute(
            """CREATE TRIGGER boundary_pause_dependency_release
               BEFORE DELETE ON artifact_dependencies FOR EACH ROW
               EXECUTE FUNCTION boundary_pause_dependency_release()"""
        )


def _hold_advisory_gate(database: RunStateDB, key: int):
    connection = database.session.connect()
    connection.execute(
        "SELECT pg_advisory_lock(hashtext(current_schema()), %s)", (key,)
    )
    pid = int(connection.execute("SELECT pg_backend_pid()").fetchone()[0])
    return connection, pid


def _release_advisory_gate(connection: Any, key: int) -> None:
    try:
        unlocked = connection.execute(
            "SELECT pg_advisory_unlock(hashtext(current_schema()), %s)", (key,)
        ).fetchone()[0]
        assert unlocked
    finally:
        connection.close()


def _artifact_reference_rows(
    database: RunStateDB, artifact_ids: tuple[int, ...]
) -> dict[int, tuple[str, int, int]]:
    placeholders = ",".join("%s" for _ in artifact_ids)
    with database.session.connection() as connection:
        rows = connection.execute(
            f"""SELECT a.artifact_id, a.status, a.ref_count,
                       COUNT(d.artifact_id) AS dependency_count
                FROM artifacts a LEFT JOIN artifact_dependencies d
                  ON d.artifact_id=a.artifact_id
                WHERE a.artifact_id IN ({placeholders})
                GROUP BY a.artifact_id, a.status, a.ref_count
                ORDER BY a.artifact_id""",
            artifact_ids,
        ).fetchall()
    return {
        int(row["artifact_id"]): (
            str(row["status"]),
            int(row["ref_count"]),
            int(row["dependency_count"]),
        )
        for row in rows
    }


def test_far_partition_publications_complete_independently(
    tmp_path: Path, postgres_database: RunStateDB
) -> None:
    state = _build_far_publication_state(postgres_database, tmp_path)
    first_partition, second_partition = tuple(state.jobs_by_partition)
    outputs = {
        partition_id: _output_arguments(tmp_path, partition_id)
        for partition_id in state.jobs_by_partition
    }
    first_open = threading.Event()
    release_first = threading.Event()
    executor = ThreadPoolExecutor(max_workers=2)
    first = executor.submit(
        _publish_and_complete,
        _worker_database(postgres_database),
        first_partition,
        state.jobs_by_partition[first_partition],
        outputs[first_partition],
        hold_open=release_first,
        transaction_open=first_open,
    )
    second: Future[Any] | None = None
    try:
        assert first_open.wait(timeout=5), "first publication did not reach hold point"
        second = executor.submit(
            _publish_and_complete,
            _worker_database(postgres_database),
            second_partition,
            state.jobs_by_partition[second_partition],
            outputs[second_partition],
        )
        _future_result(second, timeout=5)
        assert not first.done(), "held publication unexpectedly left its transaction"
    finally:
        release_first.set()
        try:
            _future_result(first)
            if second is not None:
                _future_result(second)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    for job in state.jobs_by_partition.values():
        assert postgres_database.jobs.get_job(int(job["job_id"]))["status"] == "ready"
    all_input_ids = tuple(
        artifact_id
        for partition_ids in state.input_ids_by_partition.values()
        for artifact_id in partition_ids
    )
    input_rows = _artifact_reference_rows(postgres_database, all_input_ids)
    assert all(
        (ref_count, dependency_count)
        == ((1, 1) if kind == "v3_baseline_core" else (0, 0))
        for artifact_id, (_status, ref_count, dependency_count) in input_rows.items()
        for kind in [postgres_database.artifacts.get_artifact(artifact_id)["kind"]]
    )
    with postgres_database.session.connection() as connection:
        finalize_kinds = [
            str(row[0])
            for row in connection.execute(
                """SELECT a.kind FROM artifact_dependencies d
                   JOIN artifacts a ON a.artifact_id=d.artifact_id
                   WHERE d.job_id=%s ORDER BY a.artifact_id""",
                (state.finalize_job_id,),
            ).fetchall()
        ]
    assert sorted(finalize_kinds) == sorted(
        ["v3_baseline_core", "v33_staged_mask", "v33_staged_audit"] * 2
    )


def test_baseline_ready_and_candidate_link_are_atomic_against_cleanup(
    tmp_path: Path, postgres_database: RunStateDB
) -> None:
    state = _build_far_publication_state(postgres_database, tmp_path)
    partition_id = next(iter(state.jobs_by_partition))
    baseline_id = state.input_ids_by_partition[partition_id][2]
    candidate_job_id = int(state.jobs_by_partition[partition_id]["job_id"])
    assert postgres_database.artifacts.release_artifact_dependency(
        candidate_job_id, baseline_id
    )
    baseline = postgres_database.artifacts.get_artifact(baseline_id)
    assert baseline["status"] == "ready" and int(baseline["ref_count"]) == 0
    _create_insert_gate(postgres_database, baseline_id)

    gate, gate_pid = _hold_advisory_gate(postgres_database, PRODUCER_GATE)
    publisher_pid: list[int] = []
    cleanup_pid: list[int] = []
    cleanup_result: list[dict[str, Any] | None] = []
    executor = ThreadPoolExecutor(max_workers=2)

    def publish_baseline() -> int:
        database = _worker_database(postgres_database, publisher_pid.append)
        return database.artifacts.publish_fragmentation_v33_baseline_core(
            RUN_ID,
            STREAM_ID,
            partition_id,
            Path(str(baseline["path"])),
            byte_count=int(baseline["byte_count"]),
            sha256=str(baseline["sha256"]),
        )

    def claim_cleanup() -> None:
        database = _worker_database(postgres_database, cleanup_pid.append)
        cleanup_result.append(database.artifacts.claim_artifact_cleanup(baseline_id))

    publisher = executor.submit(publish_baseline)
    cleanup: Future[Any] | None = None
    try:
        _wait_for_lock(
            postgres_database,
            waiting_pid=lambda: publisher_pid[-1] if publisher_pid else None,
            blocking_pid=gate_pid,
            query_fragment="INSERT INTO artifact_dependencies",
        )
        cleanup = executor.submit(claim_cleanup)
        _wait_for_lock(
            postgres_database,
            waiting_pid=lambda: cleanup_pid[-1] if cleanup_pid else None,
            blocking_pid=publisher_pid[-1],
            query_fragment="UPDATE artifacts SET status='cleaning'",
        )
        assert not cleanup.done(), "cleanup did not wait for the publisher row lock"
    finally:
        _release_advisory_gate(gate, PRODUCER_GATE)
        try:
            _future_result(publisher)
            if cleanup is not None:
                _future_result(cleanup)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    assert cleanup_result == [None]
    assert _artifact_reference_rows(postgres_database, (baseline_id,)) == {
        baseline_id: ("ready", 1, 1)
    }
    with postgres_database.session.connection() as connection:
        assert (
            connection.execute(
                """SELECT COUNT(*) FROM artifact_dependencies
                   WHERE job_id=%s AND artifact_id=%s""",
                (candidate_job_id, baseline_id),
            ).fetchone()[0]
            == 1
        )


def test_candidate_retention_transfer_prevents_cleanup_before_commit(
    tmp_path: Path, postgres_database: RunStateDB
) -> None:
    state = _build_far_publication_state(postgres_database, tmp_path)
    partition_id = next(iter(state.jobs_by_partition))
    job = state.jobs_by_partition[partition_id]
    job_id = int(job["job_id"])
    probability_id, context_id, baseline_id = state.input_ids_by_partition[partition_id]
    _create_release_gate(postgres_database, job_id, baseline_id)

    gate, gate_pid = _hold_advisory_gate(postgres_database, TRANSFER_GATE)
    publisher_pid: list[int] = []
    executor = ThreadPoolExecutor(max_workers=1)

    def publish_candidate() -> None:
        _publish_and_complete(
            _worker_database(postgres_database, publisher_pid.append),
            partition_id,
            job,
            _output_arguments(tmp_path, partition_id),
        )

    publisher = executor.submit(publish_candidate)
    try:
        _wait_for_lock(
            postgres_database,
            waiting_pid=lambda: publisher_pid[-1] if publisher_pid else None,
            blocking_pid=gate_pid,
            query_fragment="DELETE FROM artifact_dependencies",
        )
        # The committed candidate dependency remains visible while the publisher
        # transaction has pinned finalize and is deleting candidate inputs. MVCC
        # therefore lets cleanup reject the baseline immediately at ref_count=1;
        # it does not need to wait for the publisher's row lock.
        assert _artifact_reference_rows(postgres_database, (baseline_id,)) == {
            baseline_id: ("ready", 1, 1)
        }
        cleanup_database = _worker_database(postgres_database)
        assert cleanup_database.artifacts.claim_artifact_cleanup(baseline_id) is None
        assert not publisher.done(), "publisher left the controlled transfer gate"
        _wait_for_lock(
            postgres_database,
            waiting_pid=lambda: publisher_pid[-1] if publisher_pid else None,
            blocking_pid=gate_pid,
            query_fragment="DELETE FROM artifact_dependencies",
        )
    finally:
        _release_advisory_gate(gate, TRANSFER_GATE)
        try:
            _future_result(publisher)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    assert postgres_database.jobs.get_job(job_id)["status"] == "ready"
    assert _artifact_reference_rows(
        postgres_database, (probability_id, context_id, baseline_id)
    ) == {
        probability_id: ("ready", 0, 0),
        context_id: ("ready", 0, 0),
        baseline_id: ("ready", 1, 1),
    }
    assert postgres_database.artifacts.claim_artifact_cleanup(baseline_id) is None
    assert (
        postgres_database.artifacts.claim_artifact_cleanup(probability_id) is not None
    )
    assert postgres_database.artifacts.claim_artifact_cleanup(context_id) is not None
    with postgres_database.session.connection() as connection:
        finalize_rows = connection.execute(
            """SELECT a.kind, a.status, a.ref_count
               FROM artifact_dependencies d
               JOIN artifacts a ON a.artifact_id=d.artifact_id
               WHERE d.job_id=%s AND a.unit_id=%s ORDER BY a.artifact_id""",
            (state.finalize_job_id, partition_id),
        ).fetchall()
    assert sorted((str(row[0]), str(row[1]), int(row[2])) for row in finalize_rows) == [
        ("v33_staged_audit", "ready", 1),
        ("v33_staged_mask", "ready", 1),
        ("v3_baseline_core", "ready", 1),
    ]
