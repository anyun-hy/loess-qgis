#!/usr/bin/env python3
"""Run the required 500,000-Tile v5 control-plane pressure acceptance."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_spec import atomic_write_json, sha256_file
from labeling_tool.core.run_state_db import MAX_TILE_PAGE_SIZE, RunStateDB


TILE_ROWS = 1000
TILE_COLS = 500
TILE_COUNT = TILE_ROWS * TILE_COLS
STREAM_COUNT = 4
MIB = 1024**2


class MemorySampler:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.baseline = int(self._process.memory_info().rss)
        self.peak = self.baseline

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, int(self._process.memory_info().rss))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.peak = max(self.peak, int(self._process.memory_info().rss))
        self._stop.set()
        self._thread.join(timeout=2.0)


def _tiles():
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            yield {
                "tile_id": f"{row}_{col}",
                "row": row,
                "col": col,
                "width": 512,
                "height": 512,
                "status": "queued",
            }


def _table_counts(database_path: Path) -> tuple[dict[str, int], str]:
    tables = (
        "tiles",
        "partitions",
        "spatial_units",
        "unit_dependencies",
        "streams",
        "stream_units",
        "work_packages",
        "jobs",
    )
    with sqlite3.connect(database_path) as connection:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    return counts, journal_mode


def run_acceptance(output_root: Path, run_id: str | None = None) -> dict:
    identifier = run_id or (
        dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_c500k"
    )
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sampler = MemorySampler()
    sampler.start()
    started = time.monotonic()
    models = [
        {
            "model_id": model_id,
            "version": "control-plane-fixture-v1",
            "path": f"/fixture/{model_id}.torchscript.pt",
        }
        for model_id in ("fixture_a", "fixture_b", "fixture_c")
    ]
    scaling = {
        "partition_tile_rows": 8,
        "partition_tile_cols": 8,
        "partition_halo_px": 192,
        "seam_band_px": 64,
        "max_job_retries": 2,
    }
    spec, spec_path, database_path = create_v5_run(
        output_root=output_root,
        run_id=identifier,
        raster={
            "path": "/fixture/control-plane-source.tif",
            "crs": "EPSG:4490",
            "transform": [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
        tiles=_tiles(),
        models=models,
        effective_device="fixture",
        keep_score_cache=False,
        overlap=192,
        scaling=scaling,
        boundary_fitting={
            "enabled": True,
            "mode": "divider_cubic_bspline_adaptive_v2",
            "smoothing_factor": 1.0,
            "curve_sampling_spacing_px": 0.5,
            "max_chord_error_px": 0.25,
            "max_segment_arc_length_px": 8.0,
            "diagnostic_level": "changed_and_failed",
        },
        storage_report={"package_tile_limit": 256, "working_bytes_per_tile": 1},
        fusion={
            "profile_id": "fixture_fusion",
            "version": "control-plane-fixture-v1",
            "profile": {
                "profile_id": "fixture_fusion",
                "version": "control-plane-fixture-v1",
                "status": "approved",
                "approval": {"passed": True},
                "model_ids": [item["model_id"] for item in models],
            },
        },
        skip_accepted=False,
        config_fingerprint="control-plane-fixture",
    )
    database = RunStateDB(database_path)
    page_started = time.monotonic()
    oversized_page = database.page_tiles(identifier, limit=10_000, offset=0)
    last_page = database.page_tiles(identifier, limit=10_000, offset=TILE_COUNT - 50)
    for offset in range(0, TILE_COUNT, 5_000):
        page = database.page_tiles(identifier, limit=500, offset=offset)
        if not page:
            raise RuntimeError(f"empty Tile page at offset {offset}")
    first_stream = spec["streams"][0]["stream_id"]
    unit_page = database.page_stream_units(identifier, first_stream, limit=10_000)
    page_elapsed = time.monotonic() - page_started
    counts, journal_mode = _table_counts(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    sampler.stop()

    run_spec_bytes = spec_path.stat().st_size
    database_bytes = database_path.stat().st_size
    peak_increase = max(0, sampler.peak - sampler.baseline)
    ui_source = (
        ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "inference_monitor.py"
    ).read_text(encoding="utf-8")
    gates = {
        "exactly_500000_tiles": counts["tiles"] == TILE_COUNT,
        "four_streams": counts["streams"] == STREAM_COUNT,
        "all_stream_units_materialized": (
            counts["stream_units"] == counts["spatial_units"] * STREAM_COUNT
        ),
        "wal_enabled": journal_mode.lower() == "wal",
        "tile_page_hard_cap_500": len(oversized_page) == MAX_TILE_PAGE_SIZE,
        "last_tile_page_50": len(last_page) == 50,
        "unit_page_hard_cap_500": len(unit_page) == MAX_TILE_PAGE_SIZE,
        "run_spec_has_no_tile_details": "tiles" not in spec,
        "run_spec_below_2_mib": run_spec_bytes <= 2 * MIB,
        "peak_rss_increase_below_512_mib": peak_increase <= 512 * MIB,
        "ui_uses_database_tile_paging": (
            "self._database.page_tiles(" in ui_source
            and "self._page_size = max(1, min(int(page_size), 500))" in ui_source
            and "self._tiles.setRowCount(len(values))" in ui_source
        ),
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "scope": "500000-Tile control-plane fixture only; not L3 model inference",
        "run_id": identifier,
        "run_spec": str(spec_path),
        "state_db": str(database_path),
        "counts": counts,
        "paging": {
            "requested_limit": 10_000,
            "effective_limit": MAX_TILE_PAGE_SIZE,
            "first_page_rows": len(oversized_page),
            "last_page_rows": len(last_page),
            "unit_page_rows": len(unit_page),
            "sampled_page_query_seconds": page_elapsed,
        },
        "memory": {
            "baseline_rss_bytes": sampler.baseline,
            "peak_rss_bytes": sampler.peak,
            "peak_rss_increase_bytes": peak_increase,
            "limit_bytes": 512 * MIB,
        },
        "artifacts": {
            "run_spec_bytes": run_spec_bytes,
            "run_spec_sha256": sha256_file(spec_path),
            "state_db_bytes": database_bytes,
            "state_db_sha256": sha256_file(database_path),
        },
        "elapsed_seconds": time.monotonic() - started,
        "gates": gates,
    }
    report_path = Path(spec["run_dir"]) / "logs" / "scale_acceptance_report.json"
    atomic_write_json(report_path, report)
    report["report"] = str(report_path)
    if report["status"] != "passed":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("control-plane pressure gates failed: " + ", ".join(failed))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "v5_control_plane"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    report = run_acceptance(Path(args.output_root), args.run_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
