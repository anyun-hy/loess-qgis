"""Real two-partition V3.3 Run fixture for publication recovery tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from labeling_tool.shared.contracts.run_spec import CLASS_ORDER
from labeling_tool.shared.state.run_state_db import RunStateDB
from labeling_tool.shared.state.run_state_session import RunStateSession
from loess_runtime.geometry.fragmentation_v33_candidate import (
    executor_snapshot_sha256,
    policy_snapshot_sha256,
)
from loess_runtime.inference.partition_mosaic import write_partition_rasters

RUN_ID = "20261015_v33_recovery_fixture"
STREAM_ID = "fusion:recovery"


class TimedRunStateSession(RunStateSession):
    """Apply bounded waits only to isolated test transactions."""

    def __init__(self, *args, lock_timeout: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_lock_timeout = str(lock_timeout)

    def connect(self, *, autocommit: bool = True):
        connection = super().connect(autocommit=autocommit)
        if not autocommit:
            connection.execute(
                "SET LOCAL lock_timeout = %s",
                (self._test_lock_timeout,),
            )
            connection.execute("SET LOCAL statement_timeout = '20s'")
        return connection


@dataclass(frozen=True)
class RecoveryRun:
    database: RunStateDB
    spec: dict[str, Any]
    spec_path: Path
    run_dir: Path
    partitions: tuple[dict[str, Any], ...]
    input_ids: tuple[int, ...]


def worker_database(database: RunStateDB, *, lock_timeout: str) -> RunStateDB:
    return RunStateDB._from_session(
        TimedRunStateSession(
            database.session.location,
            schema=database.session.schema,
            lock_timeout=lock_timeout,
        )
    )


def _probabilities(labels: np.ndarray) -> np.ndarray:
    values = np.full(
        (len(CLASS_ORDER), *labels.shape),
        0.01 / (len(CLASS_ORDER) - 1),
        dtype=np.float32,
    )
    for index in range(len(CLASS_ORDER)):
        values[index, labels == index] = 0.99
    return values


def build_recovery_run(database: RunStateDB, tmp_path: Path) -> RecoveryRun:
    database.run_streams.create_run(RUN_ID, "a" * 64, status="running")
    database.run_streams.register_streams(
        RUN_ID,
        [{"stream_id": STREAM_ID, "kind": "fusion", "profile_id": "recovery"}],
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
    partitions = (
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
    )
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
                "priority": 60,
                "max_attempts": 3,
            }
            for partition in partitions
        ]
        + [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                "unit_id": "fragmentation_v33_finalize",
                "priority": 50,
                "max_attempts": 3,
            }
        ],
    )

    background = CLASS_ORDER.index(52)
    source = CLASS_ORDER.index(13)
    global_labels = np.full((20, 40), background, dtype=np.int16)
    global_labels[2:10, 2:10] = source
    global_labels[15, 15] = source
    transform = Affine(1, 0, 0, 0, -1, 20)
    input_ids: list[int] = []
    for partition in partitions:
        core = partition["core_window"]
        halo = partition["halo_window"]
        core_labels = global_labels[core["y0"] : core["y1"], core["x0"] : core["x1"]]
        halo_labels = global_labels[halo["y0"] : halo["y1"], halo["x0"] : halo["x1"]]
        root = tmp_path / "inputs" / partition["partition_id"]
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
        baseline = Path(paths["mask"])
        context = Path(paths["v3_context"])
        probability = Path(paths["probability"])
        input_ids.append(
            database.artifacts.publish_fragmentation_v33_baseline_core(
                RUN_ID,
                STREAM_ID,
                partition["partition_id"],
                baseline,
                byte_count=baseline.stat().st_size,
                sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
            )
        )
        input_ids.append(
            database.artifacts.publish_fragmentation_v33_context(
                RUN_ID,
                STREAM_ID,
                partition["partition_id"],
                context,
                byte_count=context.stat().st_size,
                sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
            )
        )
        input_ids.append(
            database.artifacts.publish_partition_artifact(
                RUN_ID,
                STREAM_ID,
                partition["partition_id"],
                probability,
                byte_count=probability.stat().st_size,
                sha256=hashlib.sha256(probability.read_bytes()).hexdigest(),
            )
        )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {
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
        "scaling": {
            "max_cpu_partition_workers": 2,
            "max_cpu_partition_workers_with_package": 2,
        },
        "resource_tuning": {
            "resolved": {
                "fragmentation_v33_process_threads": 2,
                "package_process_threads": 2,
                "unit_process_threads": 1,
            }
        },
        "fragmentation_regularization": {
            "enabled": True,
            "policy_id": "fragmentation_v33_configurable_absorption_v1",
            "publication": "authoritative_fusion_core",
            "buffer_pixels": 256,
            "max_workers": 1,
            "policy_sha256": policy_snapshot_sha256(),
            "executor_sha256": executor_snapshot_sha256(),
        },
    }
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return RecoveryRun(
        database=database,
        spec=spec,
        spec_path=spec_path,
        run_dir=run_dir,
        partitions=partitions,
        input_ids=tuple(sorted(input_ids)),
    )


def artifact_reference_rows(database: RunStateDB) -> list[tuple[int, int, int]]:
    with database.session.connection() as connection:
        return [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in connection.execute(
                """SELECT a.artifact_id, a.ref_count, COUNT(d.artifact_id)
                   FROM artifacts a LEFT JOIN artifact_dependencies d
                     ON d.artifact_id=a.artifact_id
                   WHERE a.run_id=%s
                   GROUP BY a.artifact_id, a.ref_count ORDER BY a.artifact_id""",
                (RUN_ID,),
            ).fetchall()
        ]
