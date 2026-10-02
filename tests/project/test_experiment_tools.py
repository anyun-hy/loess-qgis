import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tools.experiments import evaluate_fragmentation_v33_replay as replay
from tools.experiments import subpixel_vectorize_experiment as subpixel


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "tools" / "experiments"
EXPERIMENT_SCRIPTS = (
    "fragmentation_ab_experiment.py",
    "subpixel_vectorize_experiment.py",
    "evaluate_fragmentation_v33_replay.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, partitions: list[dict]) -> None:
    payload = {
        "status": "complete",
        "completed_partition_count": replay.PARTITION_COUNT,
        "partitions": partitions,
    }
    payload["manifest_sha256"] = replay._canonical_sha(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_experiment_clis_locate_source_dependencies_from_an_isolated_cwd(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    for script_name in EXPERIMENT_SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(EXPERIMENTS / script_name), "--help"],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()

    assert replay.REPOSITORY_ROOT == ROOT
    assert subpixel.REPOSITORY_ROOT == ROOT
    assert not (ROOT / "inference_scripts" / "fragmentation_ab_experiment.py").exists()
    assert not (ROOT / "inference_scripts" / "subpixel_vectorize_experiment.py").exists()
    assert not (
        ROOT / "inference_scripts" / "evaluate_fragmentation_v33_replay.py"
    ).exists()


def test_subpixel_experiment_writes_small_probability_outputs(tmp_path):
    height = width = 8
    probabilities = np.full((14, height, width), 1e-5, dtype=np.float32)
    probabilities[0, :, :4] = 0.9
    probabilities[1, :, 4:] = 0.9
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    scores_path = tmp_path / "scores.npz"
    np.savez(scores_path, probabilities=probabilities)

    raster_path = tmp_path / "reference.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, height, 1, 1),
    ) as destination:
        destination.write(np.zeros((height, width), dtype=np.uint8), 1)

    output_dir = tmp_path / "subpixel-output"
    report = subpixel.run(
        scores_path,
        raster_path,
        output_dir,
        sigmas=(0.0,),
        post_tolerance=0.2,
    )

    assert set(report["variants"]) == {"a_coverage_simplify", "b_subpixel"}
    assert (output_dir / "subpixel_ab_report.json").is_file()
    for variant in report["variants"].values():
        assert Path(variant["output"]).is_file()
        assert variant["coverage"]["coverage_valid"]


def test_replay_evaluation_preserves_manifest_hash_and_relative_paths(tmp_path):
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    run_dir = tmp_path / "deployment-run"
    candidate_dir = run_dir / "candidates" / "fragmentation_v33" / "raster_parts"
    candidate_dir.mkdir(parents=True)
    values = np.asarray([[12, 13], [-1, 12]], dtype=np.int16)
    indices = np.asarray([[0, 1], [-1, 0]], dtype=np.int16)
    v3_partitions = []
    historical_partitions = []
    lineage_records = []
    for number in range(replay.PARTITION_COUNT):
        partition_id = f"partition_{number:03d}"
        array_path = arrays_dir / f"{partition_id}.npy"
        np.save(array_path, values)
        digest = _sha256(array_path)
        v3_partitions.append(
            {
                "partition_id": partition_id,
                "outputs": {
                    "v3": {
                        "path": str(Path("..") / "arrays" / array_path.name),
                        "sha256": digest,
                        "shape": [2, 2],
                    }
                },
            }
        )
        historical_partitions.append(
            {
                "partition_id": partition_id,
                "outputs": {
                    "v33": {
                        "path": str(array_path.resolve()),
                        "sha256": digest,
                        "shape": [2, 2],
                    },
                    "v3": {"sha256": digest},
                },
            }
        )
        lineage_records.append(
            {
                "partition_id": partition_id,
                "v3_core": {
                    "path": str(array_path.resolve()),
                    "source_path": str(array_path.resolve()),
                },
            }
        )
        with rasterio.open(
            candidate_dir / f"{partition_id}_mask.tif",
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="int16",
            transform=from_origin(0, 2, 1, 1),
        ) as destination:
            destination.write(indices, 1)
            destination.update_tags(production_replacement="false")

    manifests_dir = tmp_path / "manifests"
    v3_manifest_path = manifests_dir / "v3.json"
    historical_manifest_path = manifests_dir / "historical-v33.json"
    _write_manifest(v3_manifest_path, v3_partitions)
    _write_manifest(historical_manifest_path, historical_partitions)
    (run_dir / "replay_lineage.json").write_text(
        json.dumps({"input_records": lineage_records}), encoding="utf-8"
    )
    run_spec_path = tmp_path / "run_spec.json"
    run_spec_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "replay": {
                    "isolated": True,
                    "production_replacement": False,
                    "v3_manifest_sha256": _sha256(v3_manifest_path),
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "reports" / "replay-evaluation.json"
    report = replay.evaluate(
        run_spec_path,
        v3_manifest_path,
        historical_manifest_path,
        output_path=output_path,
    )

    assert report["status"] == "exact_match"
    assert report["historical_difference_pixels"] == 0
    assert report["partition_count"] == replay.PARTITION_COUNT
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8"))["report_sha256"] == report[
        "report_sha256"
    ]

    np.save(arrays_dir / "partition_000.npy", values + 1)
    rejected_output = tmp_path / "reports" / "rejected-replay-evaluation.json"
    with pytest.raises(replay.ReplayEvaluationError, match="input hash changed"):
        replay.evaluate(
            run_spec_path,
            v3_manifest_path,
            historical_manifest_path,
            output_path=rejected_output,
        )
    assert not rejected_output.exists()
