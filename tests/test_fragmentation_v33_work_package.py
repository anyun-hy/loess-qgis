from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from deployment_config import CLASS_ORDER
from finalize_partition_rasters import RasterFinalizeError, finalize_partition_rasters
from fragmentation_v33_work_package import _empty_budget_audit, run_worker
from fragmentation_v33_candidate import (
    executor_snapshot_sha256,
    policy_snapshot_sha256,
)
from labeling_tool.core.run_state_db import RunStateDB
from partition_mosaic import write_partition_rasters


RUN_ID = "20260826_v33_work_package_fixture"
STREAM_ID = "fusion:fixture"


def test_empty_strict_core_is_an_explicit_noop():
    audit = _empty_budget_audit()

    assert audit["empty_class_budget"] is True
    assert audit["changed_pixel_count"] == 0
    assert audit["gap_pixels"] == 0
    assert audit["overlap_pixels"] == 0
    assert audit["outside_pixels"] == 0


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
    artifact_id = database.register_artifact(
        RUN_ID,
        kind,
        path,
        stream_id=STREAM_ID,
        unit_id=partition_id,
    )
    assert database.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return artifact_id


def test_second_stage_uses_neighbor_context_and_publishes_authoritative_v33(tmp_path):
    database = RunStateDB(tmp_path / "state.sqlite")
    database.initialize()
    database.create_run(RUN_ID, "a" * 64, status="running")
    database.register_streams(
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
    database.insert_work_packages(RUN_ID, packages)
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
    database.insert_partitions(RUN_ID, partitions)
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                    "unit_id": "fragmentation_v33",
                    "unit_type": "FragmentationV33",
                "owner_key": "all_partition_owner_cores",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 40, "y1": 20},
                "dependency_ids": [item["partition_id"] for item in partitions],
            }
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": STREAM_ID,
                    "unit_id": "fragmentation_v33",
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
        core_labels = global_labels[
            core["y0"] : core["y1"], core["x0"] : core["x1"]
        ]
        halo_labels = global_labels[
            halo["y0"] : halo["y1"], halo["x0"] : halo["x1"]
        ]
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
        database.publish_fragmentation_v33_baseline_core(
            RUN_ID,
            STREAM_ID,
            partition["partition_id"],
            baseline_path,
            byte_count=baseline_path.stat().st_size,
            sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        )
        context_path = Path(paths["v3_context"])
        database.publish_fragmentation_v33_context(
            RUN_ID,
            STREAM_ID,
            partition["partition_id"],
            context_path,
            byte_count=context_path.stat().st_size,
            sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
        )
        probability_path = Path(paths["probability"])
        database.publish_partition_artifact(
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
    assert not database.cleanup_candidates(
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
                "state_db": str(database.path),
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

    with pytest.raises(RasterFinalizeError, match="V3.3 authoritative raster"):
        finalize_partition_rasters(spec_path)

    report = run_worker(spec_path, worker_id="test-v33", lease_seconds=60)

    assert report["status"] == "ready"
    assert report["partition_count"] == 2
    assert database.job_counts(RUN_ID, job_type="fragmentation_v33") == {"ready": 1}
    for partition in partitions:
        mask_path = tmp_path / partition["partition_id"] / "v3_mask.tif"
        assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == v3_hashes[
            partition["partition_id"]
        ]
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
    assert database.artifact_for_stream_unit(
        RUN_ID, STREAM_ID, "partition_00000_00000", "core_mask"
    )["path"] == str(candidate.resolve())
    assert database.cleanup_candidates(
        RUN_ID,
        kinds=("partition_probability", "v3_context_core", "v3_baseline_core"),
    )
