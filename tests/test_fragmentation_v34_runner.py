from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "_test_v34_runner",
    "scratch/v34_full_140_20260826/run_v34_from_v33.py",
)


def test_v34_four_core_runner_and_resume_are_stable(tmp_path):
    parent = RUNNER._B._write_self_test_parent(tmp_path)
    v33_root = tmp_path / "v33"
    RUNNER.V33_RUNNER.run(parent, v33_root, workers=2, resume=False, self_test=True)
    output = tmp_path / "v34"
    first = RUNNER.run(
        v33_root / "run_manifest.json", output,
        workers=2, resume=False, self_test=True,
    )
    second = RUNNER.run(
        v33_root / "run_manifest.json", output,
        workers=2, resume=True, self_test=True,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert second["status"] == "complete"
    assert second["completed_partition_count"] == 4
    assert second["parent_v33_manifest_sha256"] == first["parent_v33_manifest_sha256"]
    assert all(Path(item["outputs"]["v34"]["path"]).is_file() for item in second["partitions"])


def test_v34_stitch_treats_touching_owner_cores_as_one_context(tmp_path):
    parent = RUNNER._B._write_self_test_parent(tmp_path)
    v33_root = tmp_path / "v33"
    manifest = RUNNER.V33_RUNNER.run(
        parent, v33_root, workers=2, resume=False, self_test=True,
    )
    data = RUNNER._load_v33(v33_root / "run_manifest.json", self_test=True)
    entry = data["source"]["entries"][0]
    stitched, expanded = RUNNER._stitch_v33(
        entry, data["source"]["entries"], data["parts"], data["source"]["global_window"],
    )
    component_map, components = RUNNER.V33_RUNNER.apply_v33_candidate.__globals__["_b"]._component_index(
        stitched,
        stitched >= 0,
        RUNNER.CLASS_ORDER,
    )
    assert expanded == RUNNER._B._expand(entry["core_window"], data["source"]["global_window"])
    assert len(components) == 1
    assert component_map.max() == 1
    assert manifest["completed_partition_count"] == 4
