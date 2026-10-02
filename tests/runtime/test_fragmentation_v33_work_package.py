from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from labeling_tool.shared.contracts.monitor_contract import MONITOR_EXECUTION_ENV
from labeling_tool.shared.contracts.run_spec import CLASS_ORDER
from labeling_tool.shared.state.run_execution_ownership import (
    RUN_OWNER_ENV,
    RunExecutionOwnership,
)
from labeling_tool.shared.state.run_state_db import RunStateDB
from labeling_tool.shared.state.run_state_session import RunStateError
from loess_runtime.assembly.finalize_partition_rasters import (
    RasterFinalizeError,
    finalize_partition_rasters,
)
from loess_runtime.geometry import fragmentation_v33_work_package as work_package
from loess_runtime.geometry.fragmentation_v33_candidate import (
    executor_snapshot_sha256,
    policy_snapshot_sha256,
)
from loess_runtime.geometry.fragmentation_v33_work_package import (
    _empty_budget_audit,
    _fragmentation_attempt_key,
    run_worker,
)
from loess_runtime.inference.partition_mosaic import write_partition_rasters

RUN_ID = "20260826_v33_work_package_fixture"
STREAM_ID = "fusion:fixture"


@pytest.fixture
def run_execution_owner(monkeypatch):
    owners = []

    def acquire(database, run_id, run_dir):
        ownership = RunExecutionOwnership.acquire(
            database,
            run_id,
            run_dir=run_dir,
            worker_id="runtime-v33-publication-test",
            trigger_type="start",
        )
        monkeypatch.setenv(RUN_OWNER_ENV, ownership.identity.environment_value())
        monkeypatch.setenv(MONITOR_EXECUTION_ENV, ownership.identity.execution_id)
        owners.append(ownership)
        return ownership

    yield acquire

    for ownership in reversed(owners):
        ownership.close()


def _copy_control_plane_state(source: RunStateDB, destination: RunStateDB) -> None:
    """Copy a frozen PostgreSQL test schema into another isolated schema."""
    table_order = (
        "runs",
        "streams",
        "stream_runtime_progress",
        "work_packages",
        "partitions",
        "tiles",
        "spatial_units",
        "unit_dependencies",
        "stream_units",
        "jobs",
        "artifacts",
        "artifact_dependencies",
        "unit_report_summaries",
        "object_links",
        "object_nodes",
        "events",
    )
    source_schema = source.session.schema
    destination_schema = destination.session.schema
    with source.session.connection() as source_connection:
        with destination.session.transaction() as destination_connection:
            for table in table_order:
                rows = source_connection.execute(
                    f'SELECT * FROM "{source_schema}"."{table}"'
                ).fetchall()
                if not rows:
                    continue
                columns = tuple(rows[0].keys())
                quoted_columns = ", ".join(f'"{column}"' for column in columns)
                placeholders = ", ".join("%s" for _ in columns)
                statement = (
                    f'INSERT INTO "{destination_schema}"."{table}" '
                    f"({quoted_columns}) OVERRIDING SYSTEM VALUE VALUES ({placeholders})"
                )
                for row in rows:
                    destination_connection.execute(
                        statement, tuple(row[column] for column in columns)
                    )
            for table, key in (
                ("jobs", "job_id"),
                ("artifacts", "artifact_id"),
                ("events", "event_id"),
            ):
                destination_connection.execute(
                    "SELECT setval(pg_get_serial_sequence(%s, %s), "
                    "COALESCE((SELECT MAX("
                    + key
                    + ') FROM "'
                    + destination_schema
                    + '"."'
                    + table
                    + '"), 1), true)',
                    (f"{destination_schema}.{table}", key),
                )


def test_empty_strict_core_is_an_explicit_noop():
    audit = _empty_budget_audit()

    assert audit["empty_class_budget"] is True
    assert audit["changed_pixel_count"] == 0
    assert audit["gap_pixels"] == 0
    assert audit["overlap_pixels"] == 0
    assert audit["outside_pixels"] == 0


def test_fragmentation_attempt_paths_use_the_complete_lease_identity():
    first = _fragmentation_attempt_key(17, "lease-token-a")
    second = _fragmentation_attempt_key(17, "lease-token-b")

    assert first != second
    assert first == f"job17.{hashlib.sha256(b'lease-token-a').hexdigest()}"
    assert "lease-token-a" not in first


def test_stale_v33_lease_cannot_replace_the_ready_attempt_artifacts(
    tmp_path, postgres_database, monkeypatch
):
    database = postgres_database
    run_id = "stale-v33-attempt"
    stream_id = "fusion:fixture"
    partition_id = "partition_00000_00000"
    unit_id = f"fragmentation_v33_partition:{partition_id}"
    database.run_streams.create_run(run_id, "a" * 64, status="running")
    database.run_streams.register_streams(
        run_id,
        [{"stream_id": stream_id, "kind": "fusion", "profile_id": "fixture"}],
    )
    database.control_graph.insert_work_packages(
        run_id,
        [{"package_id": "package_00000", "sequence_no": 0, "status": "ready"}],
    )
    partition = {
        "partition_id": partition_id,
        "row": 0,
        "col": 0,
        "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
        "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
        "package_id": "package_00000",
        "status": "ready",
    }
    database.control_graph.insert_partitions(run_id, [partition])
    database.control_graph.insert_spatial_units(
        run_id,
        [
            {
                "unit_id": unit_id,
                "unit_type": "FragmentationV33Partition",
                "owner_key": partition_id,
                "pixel_window": partition["core_window"],
                "dependency_ids": [partition_id],
            },
            {
                "unit_id": "fragmentation_v33_finalize",
                "unit_type": "FragmentationV33Finalize",
                "owner_key": "all_partition_owner_cores",
                "pixel_window": partition["core_window"],
                "dependency_ids": [partition_id],
            },
        ],
    )
    database.jobs.insert_jobs(
        run_id,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": unit_id,
                "max_attempts": 2,
            },
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": "fragmentation_v33_finalize",
                "max_attempts": 2,
            },
        ],
    )
    with database.session.connection() as connection:
        partition_job_id = int(
            connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=%s AND unit_id=%s",
                (run_id, unit_id),
            ).fetchone()[0]
        )
    leased = database.jobs.lease_job(
        partition_job_id, "current-worker", lease_seconds=120
    )
    assert leased is not None

    current_root = tmp_path / "current-attempt"
    current_root.mkdir()
    current_mask = current_root / "mask.tif"
    current_audit = current_root / "audit.json"
    current_mask.write_bytes(b"current-mask")
    current_audit.write_bytes(b"current-audit")
    baseline = tmp_path / "baseline.tif"
    baseline.write_bytes(b"frozen-baseline")
    database.artifacts.publish_fragmentation_v33_baseline_core(
        run_id,
        stream_id,
        partition_id,
        baseline,
        byte_count=baseline.stat().st_size,
        sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    with database.fragmentation_v33_attempt_commit(
        leased["job_id"], leased["lease_token"]
    ) as publication:
        publication.artifacts.publish_fragmentation_v33_output_pair(
            run_id,
            stream_id,
            partition_id,
            mask_path=current_mask,
            mask_byte_count=current_mask.stat().st_size,
            mask_sha256=hashlib.sha256(current_mask.read_bytes()).hexdigest(),
            audit_path=current_audit,
            audit_byte_count=current_audit.stat().st_size,
            audit_sha256=hashlib.sha256(current_audit.read_bytes()).hexdigest(),
            production=None,
        )
        assert publication.jobs.complete_fragmentation_v33_job(
            leased["job_id"], leased["lease_token"]
        )

    stale_root = tmp_path / "stale-attempt"
    stale_root.mkdir()
    stale_mask = stale_root / "mask.tif"
    stale_audit = stale_root / "audit.json"
    stale_mask.write_bytes(b"stale-mask")
    stale_audit.write_bytes(b"stale-audit")
    stale_output = {
        "run_id": run_id,
        "stream_id": stream_id,
        "partition_id": partition_id,
        "mask_path": stale_mask,
        "mask_byte_count": stale_mask.stat().st_size,
        "mask_sha256": hashlib.sha256(stale_mask.read_bytes()).hexdigest(),
        "audit_path": stale_audit,
        "audit_byte_count": stale_audit.stat().st_size,
        "audit_sha256": hashlib.sha256(stale_audit.read_bytes()).hexdigest(),
        "production": None,
    }

    class InertHeartbeat:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, _total):
            pass

        def progress(self, _current):
            pass

        def fence(self):
            pass

        def stop_for_publication(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(work_package, "_Heartbeat", InertHeartbeat)
    monkeypatch.setattr(work_package, "ready_artifact_index", lambda *_args: {})
    monkeypatch.setattr(
        work_package,
        "_run_partition",
        lambda *_args, **_kwargs: ({"partition_id": partition_id}, stale_output),
    )

    with pytest.raises(RunStateError, match="no longer owns its lease"):
        work_package._run_durable_partition_job(
            {"run_id": run_id, "run_dir": str(tmp_path)},
            database,
            leased,
            {"buffer_pixels": 1, "policy_sha256": "b" * 64, "executor_sha256": "c" * 64},
            lease_seconds=120,
        )

    mask_artifact = database.artifacts.artifact_for_stream_unit(
        run_id, stream_id, partition_id, "v33_staged_mask"
    )
    audit_artifact = database.artifacts.artifact_for_stream_unit(
        run_id, stream_id, partition_id, "v33_staged_audit"
    )
    assert mask_artifact["path"] == str(current_mask.resolve())
    assert audit_artifact["path"] == str(current_audit.resolve())
    assert current_mask.read_bytes() == b"current-mask"
    assert current_audit.read_bytes() == b"current-audit"


@pytest.mark.parametrize(
    "filename",
    [
        "run_spec.py",
        "partition_mosaic.py",
        "contracts.py",
        "incremental_metrics.py",
        "proposal_identity.py",
        "topology.py",
        "fragmentation_v33_contract.py",
        "fragmentation_v33_artifact_io.py",
        "fragmentation_v33_finalization.py",
    ],
)
def test_executor_snapshot_covers_split_v33_sources(monkeypatch, filename):
    original_read_bytes = Path.read_bytes
    changed_paths = []
    baseline = executor_snapshot_sha256()

    def changed_read_bytes(path):
        payload = original_read_bytes(path)
        if path.name == filename:
            changed_paths.append(path)
            return payload + b"\n# executor identity test\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed_read_bytes)

    assert executor_snapshot_sha256() != baseline
    assert len(changed_paths) == 1


def test_stale_executor_is_rejected_before_state_store_open(tmp_path, monkeypatch):
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "run_id": "stale-v33-executor",
                "fragmentation_regularization": {
                    "enabled": True,
                    "policy_id": "fragmentation_v33_configurable_absorption_v1",
                    "publication": "authoritative_fusion_core",
                    "policy_sha256": policy_snapshot_sha256(),
                    "executor_sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    opened = []

    def unexpected_open(_spec):
        opened.append(True)
        raise AssertionError("state store must not open for a stale executor")

    monkeypatch.setattr(work_package, "run_state_from_spec", unexpected_open)

    with pytest.raises(
        work_package.FragmentationV33WorkPackageError,
        match="executor differs",
    ):
        run_worker(spec_path, worker_id="stale-executor-test")

    assert opened == []


def _probabilities(labels: np.ndarray) -> np.ndarray:
    values = np.full(
        (len(CLASS_ORDER), *labels.shape),
        0.01 / (len(CLASS_ORDER) - 1),
        dtype=np.float32,
    )
    for index in range(len(CLASS_ORDER)):
        values[index, labels == index] = 0.99
    return values


def _ready_artifact(
    database: RunStateDB,
    path: Path,
    *,
    kind: str,
    partition_id: str,
) -> int:
    artifact_id = database.artifacts.register_artifact(
        RUN_ID,
        kind,
        path,
        stream_id=STREAM_ID,
        unit_id=partition_id,
    )
    assert database.artifacts.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return artifact_id


@pytest.mark.parametrize("cleanup_between_stages", [False, True])
def test_partition_workers_use_neighbor_context_then_finalize_authoritatively(
    tmp_path,
    postgres_database,
    postgres_database_factory,
    run_execution_owner,
    cleanup_between_stages,
):
    database = postgres_database
    database.run_streams.create_run(RUN_ID, "a" * 64, status="running")
    database.run_streams.register_streams(
        RUN_ID,
        [{"stream_id": STREAM_ID, "kind": "fusion", "profile_id": "fixture"}],
    )
    packages = [
        {
            "package_id": f"package_{index}",
            "sequence_no": index,
            "partition_ids": [f"partition_00000_0000{index}"],
            "status": "ready",
        }
        for index in range(2)
    ]
    database.control_graph.insert_work_packages(RUN_ID, packages)
    partitions = [
        {
            "partition_id": "partition_00000_00000",
            "row": 0,
            "col": 0,
            "core_window": {"x0": 0, "y0": 0, "x1": 16, "y1": 20},
            "halo_window": {"x0": 0, "y0": 0, "x1": 24, "y1": 20},
            "package_id": "package_0",
            "status": "ready",
        },
        {
            "partition_id": "partition_00000_00001",
            "row": 0,
            "col": 1,
            "core_window": {"x0": 16, "y0": 0, "x1": 40, "y1": 20},
            "halo_window": {"x0": 12, "y0": 0, "x1": 40, "y1": 20},
            "package_id": "package_1",
            "status": "ready",
        },
    ]
    database.control_graph.insert_partitions(RUN_ID, partitions)
    database.control_graph.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": f"fragmentation_v33_partition:{partition['partition_id']}",
                "unit_type": "FragmentationV33Partition",
                "owner_key": partition["partition_id"],
                "pixel_window": partition["core_window"],
                "dependency_ids": [item["partition_id"] for item in partitions],
            }
            for partition in partitions
        ]
        + [
            {
                "unit_id": "fragmentation_v33_finalize",
                "unit_type": "FragmentationV33Finalize",
                "owner_key": "all_partition_owner_cores",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 40, "y1": 20},
                "dependency_ids": [item["partition_id"] for item in partitions],
            }
        ],
    )
    database.jobs.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                "unit_id": f"fragmentation_v33_partition:{partition['partition_id']}",
                "max_attempts": 2,
            }
            for partition in partitions
        ]
        + [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                "unit_id": "fragmentation_v33_finalize",
                "max_attempts": 2,
            }
        ],
    )

    background = CLASS_ORDER.index(52)
    source = CLASS_ORDER.index(13)
    global_labels = np.full((20, 40), background, dtype=np.int16)
    global_labels[2:10, 2:10] = source
    # This source lies on the Core boundary. It is enclosed only when the
    # second owner's V3 context is stitched into the first target window.
    global_labels[15, 15] = source
    transform = Affine(1, 0, 0, 0, -1, 20)
    v3_hashes: dict[str, str] = {}

    for partition in partitions:
        core = partition["core_window"]
        halo = partition["halo_window"]
        core_labels = global_labels[core["y0"] : core["y1"], core["x0"] : core["x1"]]
        halo_labels = global_labels[halo["y0"] : halo["y1"], halo["x0"] : halo["x1"]]
        root = tmp_path / partition["partition_id"]
        paths = write_partition_rasters(
            {
                "halo_probabilities": _probabilities(halo_labels),
                "core_mask": core_labels,
                "core_confidence": np.full(core_labels.shape, 0.99, dtype=np.float32),
                "v3_context_core": core_labels,
            },
            partition,
            global_transform=transform,
            crs="EPSG:3857",
            output_probability=root / "probability.tif",
            output_mask=root / "v3_mask.tif",
            output_confidence=root / "confidence.tif",
            output_v3_context=root / "v3_context.tif",
        )
        baseline_path = Path(paths["mask"])
        database.artifacts.publish_fragmentation_v33_baseline_core(
            RUN_ID,
            STREAM_ID,
            partition["partition_id"],
            baseline_path,
            byte_count=baseline_path.stat().st_size,
            sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        )
        context_path = Path(paths["v3_context"])
        database.artifacts.publish_fragmentation_v33_context(
            RUN_ID,
            STREAM_ID,
            partition["partition_id"],
            context_path,
            byte_count=context_path.stat().st_size,
            sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
        )
        probability_path = Path(paths["probability"])
        database.artifacts.publish_partition_artifact(
            RUN_ID,
            STREAM_ID,
            partition["partition_id"],
            probability_path,
            byte_count=probability_path.stat().st_size,
            sha256=hashlib.sha256(probability_path.read_bytes()).hexdigest(),
        )
        v3_hashes[partition["partition_id"]] = hashlib.sha256(
            Path(paths["mask"]).read_bytes()
        ).hexdigest()

    # Probability publication and candidate linkage share one transaction, so
    # cleanup cannot claim either probability before the candidate consumes it.
    assert not database.artifacts.cleanup_candidates(
        RUN_ID, kinds=("partition_probability",)
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": RUN_ID,
                "run_dir": str(run_dir),
                "state_backend": "postgresql",
                "state_db": database.session.location,
                "state_schema": database.session.schema,
                "raster": {
                    "path": str(tmp_path / "source.tif"),
                    "crs": "EPSG:3857",
                    "transform": list(transform)[:6],
                },
                "fragmentation_regularization": {
                    "enabled": True,
                    "policy_id": "fragmentation_v33_configurable_absorption_v1",
                    "publication": "authoritative_fusion_core",
                    "buffer_pixels": 256,
                    "policy_sha256": policy_snapshot_sha256(),
                    "executor_sha256": executor_snapshot_sha256(),
                },
            }
        ),
        encoding="utf-8",
    )

    # Keep two frozen control-plane snapshots. They retain the same immutable
    # V3/probability inputs, but publish into independent PostgreSQL schemas.
    snapshots = {}
    for worker_limit in () if cleanup_between_stages else (1, 2):
        root = tmp_path / f"workers-{worker_limit}"
        root.mkdir()
        snapshot_database = postgres_database_factory()
        _copy_control_plane_state(database, snapshot_database)
        run_copy = root / "run"
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        payload["state_db"] = snapshot_database.session.location
        payload["state_schema"] = snapshot_database.session.schema
        payload["run_dir"] = str(run_copy)
        run_copy.mkdir()
        copied_spec = run_copy / "run_spec.json"
        copied_spec.write_text(json.dumps(payload), encoding="utf-8")
        snapshots[worker_limit] = (snapshot_database, copied_spec)

    cleaned_kinds = set()

    def cleanup_released_inputs(active_database):
        for artifact in active_database.artifacts.cleanup_candidates(
            RUN_ID,
            kinds=("partition_probability", "v3_context_core", "v3_baseline_core"),
        ):
            claimed = active_database.artifacts.claim_artifact_cleanup(
                artifact["artifact_id"]
            )
            assert claimed is not None
            path = Path(claimed["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == claimed["sha256"]
            path.unlink()
            assert active_database.artifacts.finish_artifact_cleanup(
                claimed["artifact_id"], success=True
            )
            cleaned_kinds.add(claimed["kind"])

    def execute_v33_graph(active_database, active_spec, worker_limit):
        partition_reports = []
        while True:
            leases = []
            for index in range(worker_limit):
                leased = active_database.jobs.lease_next_fragmentation_v33(
                    RUN_ID,
                    f"test-v33-{worker_limit}-{index}",
                    lease_seconds=60,
                    max_running=worker_limit,
                )
                if leased is not None:
                    leases.append(leased)
            if not leases:
                break
            for leased in leases:
                report = run_worker(
                    active_spec,
                    worker_id=f"test-v33-{worker_limit}",
                    lease_seconds=60,
                    job_id=leased["job_id"],
                    lease_token=leased["lease_token"],
                )
                if report.get("stage") == "partition":
                    partition_reports.append(report)
                    if cleanup_between_stages:
                        # Exercise the runner's cleanup lifecycle before the
                        # global audit consumes its baseline, not just after.
                        cleanup_released_inputs(active_database)
                else:
                    return partition_reports, report
        raise AssertionError("V3.3 graph did not reach its finalize barrier")

    with pytest.raises(RasterFinalizeError, match="V3.3 authoritative raster"):
        finalize_partition_rasters(spec_path)

    # The durable graph has one staged owner job per partition, then one
    # global barrier job. External leases mirror runner dispatch and prove that
    # the worker cannot silently choose the old serial/replay route.
    run_execution_owner(database, RUN_ID, run_dir)
    partition_reports, report = execute_v33_graph(database, spec_path, 4)
    assert {item["stage"] for item in partition_reports} == {"partition"}

    assert report["status"] == "ready"
    assert report["partition_count"] == 2
    assert database.jobs.job_counts(RUN_ID, job_type="fragmentation_v33") == {
        "ready": 3
    }
    for partition in partitions:
        mask_path = tmp_path / partition["partition_id"] / "v3_mask.tif"
        assert (
            hashlib.sha256(mask_path.read_bytes()).hexdigest()
            == v3_hashes[partition["partition_id"]]
        )
    candidate = (
        run_dir
        / "fusion"
        / "fixture"
        / "raster_parts"
        / "partition_00000_00000_mask.tif"
    )
    with rasterio.open(candidate) as source_raster:
        result = source_raster.read(1)
        assert source_raster.tags()["production_replacement"] == "true"
        assert source_raster.tags()["classification_authority"] == (
            "fragmentation_v33_authoritative_fusion_core_v1"
        )
    assert result[15, 15] == background
    assert report["validation_status"] == "passed"
    assert report["production_replacement"] is True
    assert report["acceptance"]["gap_pixels"] == 0
    assert database.artifacts.artifact_for_stream_unit(
        RUN_ID, STREAM_ID, "partition_00000_00000", "core_mask"
    )["path"] == str(candidate.resolve())
    assert database.artifacts.cleanup_candidates(
        RUN_ID,
        kinds=("partition_probability", "v3_context_core", "v3_baseline_core"),
    )

    if cleanup_between_stages:
        assert cleaned_kinds == {"partition_probability", "v3_context_core"}
        baselines = database.artifacts.artifacts_for_stream(
            RUN_ID, STREAM_ID, kind="v3_baseline_core"
        )
        assert len(baselines) == len(partitions)
        assert all(
            row["status"] == "ready" and row["ref_count"] == 0 for row in baselines
        )
        cleanup_released_inputs(database)
        assert all(not Path(row["path"]).exists() for row in baselines)
        assert all(
            database.artifacts.get_artifact(row["artifact_id"])["status"] == "cleaned"
            for row in baselines
        )
        return

    mask_hashes = {
        4: {
            item["unit_id"]: item["sha256"]
            for item in database.artifacts.artifacts_for_stream(
                RUN_ID, STREAM_ID, kind="core_mask"
            )
        }
    }
    for worker_limit, (snapshot_database, snapshot_spec) in snapshots.items():
        run_execution_owner(
            snapshot_database,
            RUN_ID,
            Path(snapshot_spec).parent,
        )
        staged, snapshot_report = execute_v33_graph(
            snapshot_database, snapshot_spec, worker_limit
        )
        assert len(staged) == len(partitions)
        assert snapshot_report["global_connectivity_4_connected"]["hard_gate"]["passed"]
        mask_hashes[worker_limit] = {
            item["unit_id"]: item["sha256"]
            for item in snapshot_database.artifacts.artifacts_for_stream(
                RUN_ID, STREAM_ID, kind="core_mask"
            )
        }
    assert mask_hashes[1] == mask_hashes[2] == mask_hashes[4]
