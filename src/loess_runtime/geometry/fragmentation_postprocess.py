"""Resumable V3 fragmentation repair for historical completed Fusion runs.

The stage consumes committed partition mask/confidence rasters and writes only
below ``run_dir/postprocess``.  Original inference, fitted vectors, confidence
rasters, and class workspaces are never overwritten.  Once every derived
artifact passes validation, ``--activate-review`` may atomically point the Run
manifest at the derived polygon layer. New v5 Runs apply V3 to probability
Halos before vectorization and never invoke this compatibility tool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import sqlite3
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import fiona
import rasterio

from labeling_tool.shared.contracts.run_spec import (
    CLASS_ORDER,
    load_json,
    sha256_file,
)
from loess_runtime.geometry.fragmentation_postprocess_partitions import (
    FORMAL_SCHEMA,
    LAYER_NAME,
    FragmentationPostprocessError,
    confidence_path,
    file_signature,
    json_fingerprint,
    partition_id,
    partition_key,
    process_mask_partition,
    process_vector_partition,
    write_atomic_json,
)
from loess_runtime.geometry.fragmentation_v3 import (
    DEFAULT_BUFFER_PIXELS,
    DEFAULT_MAX_WORKERS,
    POLICY_ID,
    POLICY_VERSION,
    PROTECTED_SOURCE_CLASS_CODES,
    policy_snapshot,
)
from loess_runtime.inference.partition_mosaic import build_vrt

MANIFEST_NAME = "fragmentation_v3_manifest.json"
REPORT_NAME = "fragmentation_v3_report.json"


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _source_inventory_fingerprint(mask_paths: Sequence[Path]) -> str:
    return json_fingerprint(
        {
            "policy_version": POLICY_VERSION,
            "parts": [
                {
                    "mask": file_signature(path),
                    "confidence": file_signature(confidence_path(path)),
                }
                for path in mask_paths
            ],
        }
    )


def _stream(spec: Mapping[str, Any], stream_id: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in spec.get("streams") or []
        if str(item.get("stream_id") or "") == stream_id
    ]
    if len(matches) != 1:
        raise FragmentationPostprocessError(
            f"run_spec must contain exactly one stream {stream_id!r}"
        )
    stream = matches[0]
    if stream.get("kind") != "fusion":
        raise FragmentationPostprocessError(
            "V3 production repair requires a Fusion stream"
        )
    return stream


def derived_root(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    return (
        Path(str(spec["run_dir"])).resolve()
        / "postprocess"
        / POLICY_ID
        / "fusion"
        / str(stream["profile_id"])
    )


def derived_manifest_path(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    return derived_root(spec, stream) / MANIFEST_NAME


def validated_derived_review(
    run_spec: Mapping[str, Any], stream: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a passed derived manifest only when its final GPKG is unchanged."""

    path = derived_manifest_path(run_spec, stream)
    try:
        manifest = load_json(path)
        output = Path(str(manifest["semantic_polygons"]))
        if (
            manifest.get("status") != "passed"
            or manifest.get("run_id") != run_spec.get("run_id")
            or manifest.get("stream_id") != stream.get("stream_id")
            or manifest.get("policy_version") != POLICY_VERSION
            or not output.is_file()
            or sha256_file(output) != manifest.get("semantic_polygons_sha256")
        ):
            return None
    except (KeyError, OSError, ValueError, TypeError):
        return None
    return dict(manifest)


def _run_bounded_processes(
    payloads: Sequence[Mapping[str, Any]],
    worker,
    *,
    workers: int,
    event_prefix: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(payloads)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        iterator = iter(payloads)
        active = set()
        for _ in range(min(max(1, int(workers)), total)):
            try:
                active.add(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while active:
            completed, active = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                results.append(result)
                emit(
                    f"{event_prefix}_progress",
                    current=len(results),
                    total=total,
                    partition_id=result.get("partition_id", ""),
                    resumed=bool(result.get("resumed")),
                )
                try:
                    active.add(executor.submit(worker, next(iterator)))
                except StopIteration:
                    pass
    return results


def _assemble_vectors(
    output_path: Path,
    part_paths: Sequence[Path],
    *,
    crs_wkt: str,
) -> dict[str, Any]:
    temporary = output_path.with_name(f".{output_path.stem}.building.gpkg")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with fiona.open(
            temporary,
            "w",
            driver="GPKG",
            layer=LAYER_NAME,
            schema=FORMAL_SCHEMA,
            crs_wkt=crs_wkt,
        ) as destination:
            total = len(part_paths)
            for position, part_path in enumerate(part_paths, start=1):
                with fiona.open(part_path, layer=LAYER_NAME) as source:
                    batch = []
                    for feature in source:
                        batch.append(feature)
                        if len(batch) >= 4096:
                            destination.writerecords(batch)
                            count += len(batch)
                            batch = []
                    if batch:
                        destination.writerecords(batch)
                        count += len(batch)
                if position % 25 == 0 or position == total:
                    emit(
                        "fragmentation_vector_assembly_progress",
                        current=position,
                        total=total,
                        feature_count=count,
                    )
        with sqlite3.connect(temporary) as connection:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_polygons_class_code "
                "ON semantic_polygons(class_code)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_polygons_object_id "
                "ON semantic_polygons(object_id)"
            )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            stored_count = int(
                connection.execute("SELECT COUNT(*) FROM semantic_polygons").fetchone()[
                    0
                ]
            )
        if integrity != "ok" or stored_count != count:
            raise FragmentationPostprocessError(
                f"assembled GPKG validation failed: integrity={integrity}, "
                f"features={stored_count}/{count}"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "feature_count": count,
        "integrity_check": "ok",
        "sha256": sha256_file(output_path),
    }


def _aggregate_mask_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    before = Counter()
    after = Counter()
    changed_pixels = 0
    changed_components = 0
    kept = Counter()
    pairs = Counter()
    for report in reports:
        before.update(
            {
                int(code): int(value)
                for code, value in report["class_pixel_count_before"].items()
            }
        )
        after.update(
            {
                int(code): int(value)
                for code, value in report["class_pixel_count_after"].items()
            }
        )
        changed_pixels += int(report["changed_pixel_count"])
        regularization = report.get("regularization") or {}
        changed_components += int(regularization.get("changed_component_count") or 0)
        kept.update(regularization.get("kept_reason_counts") or {})
        pairs.update(regularization.get("changed_pair_counts") or {})
    protected_unchanged = all(
        before[code] == after[code] for code in PROTECTED_SOURCE_CLASS_CODES
    )
    disappeared = [
        code for code in CLASS_ORDER if before[code] > 0 and after[code] <= 0
    ]
    return {
        "partition_count": len(reports),
        "changed_pixel_count": changed_pixels,
        "changed_component_count": changed_components,
        "class_pixel_count_before": {str(code): before[code] for code in CLASS_ORDER},
        "class_pixel_count_after": {str(code): after[code] for code in CLASS_ORDER},
        "protected_classes_unchanged": protected_unchanged,
        "disappeared_class_codes": disappeared,
        "kept_reason_counts": dict(sorted(kept.items())),
        "changed_pair_counts": dict(sorted(pairs.items())),
        "passed": protected_unchanged and not disappeared,
    }


def _activate_review(
    run_dir: Path,
    *,
    stream_id: str,
    manifest: Mapping[str, Any],
) -> Path:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FragmentationPostprocessError(
            f"run manifest is missing; cannot activate review source: {run_manifest_path}"
        )
    run_manifest = dict(load_json(run_manifest_path))
    updated = 0
    for key in ("ready_streams", "streams"):
        collection = run_manifest.get(key)
        if not isinstance(collection, list):
            continue
        for stream in collection:
            if str(stream.get("stream_id") or "") != stream_id:
                continue
            stream["review_polygons"] = str(manifest["semantic_polygons"])
            stream["review_layer_name"] = LAYER_NAME
            checksums = dict(stream.get("output_sha256") or {})
            checksums["review_polygons"] = str(manifest["semantic_polygons_sha256"])
            stream["output_sha256"] = checksums
            stream["fragmentation_postprocess"] = {
                "policy_id": POLICY_ID,
                "policy_version": POLICY_VERSION,
                "manifest": str(
                    Path(str(manifest["report_path"])).parent / MANIFEST_NAME
                ),
                "report": str(manifest["report_path"]),
            }
            updated += 1
    if updated < 1:
        raise FragmentationPostprocessError(
            f"stream {stream_id!r} is not present in run_manifest"
        )
    write_atomic_json(run_manifest_path, run_manifest)
    return run_manifest_path


def run_postprocess(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.run_spec).expanduser().resolve()
    spec = load_json(spec_path)
    if spec.get("schema_version") != 2:
        raise FragmentationPostprocessError("V3 postprocess requires run_spec schema 2")
    stream_id = str(args.stream_id)
    stream = _stream(spec, stream_id)
    run_dir = Path(str(spec["run_dir"])).resolve()
    source_root = run_dir / "fusion" / str(stream["profile_id"])
    mask_dir = source_root / "raster_parts"
    mask_paths = sorted(mask_dir.glob("partition_*_mask.tif"))
    if not mask_paths:
        raise FragmentationPostprocessError(f"no Fusion partition masks: {mask_dir}")
    expected = int(
        (spec.get("spatial_plan_summary") or {}).get("partition_count")
        or len(mask_paths)
    )
    if len(mask_paths) != expected:
        raise FragmentationPostprocessError(
            f"incomplete mask set: {len(mask_paths)}/{expected}"
        )
    for path in mask_paths:
        confidence = confidence_path(path)
        if not confidence.is_file() or confidence.stat().st_size <= 0:
            raise FragmentationPostprocessError(
                f"missing confidence partition: {confidence}"
            )
    path_map = {partition_key(path): path for path in mask_paths}
    source_inventory_fingerprint = _source_inventory_fingerprint(mask_paths)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir or "").strip()
        else derived_root(spec, stream)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".fragmentation_v3.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FragmentationPostprocessError(
                f"another V3 postprocess owns the output lock: {lock_path}"
            ) from error
        workers = max(1, int(args.workers))
        resume = not bool(args.restart)
        manifest_path = output_root / MANIFEST_NAME
        if resume and args.stage == "all" and manifest_path.is_file():
            try:
                completed = dict(load_json(manifest_path))
                completed_output = Path(str(completed["semantic_polygons"]))
                completed_report = Path(str(completed["report_path"]))
                if (
                    completed.get("status") == "passed"
                    and completed.get("run_id") == spec.get("run_id")
                    and completed.get("stream_id") == stream_id
                    and completed.get("policy_version") == POLICY_VERSION
                    and completed.get("source_inventory_fingerprint")
                    == source_inventory_fingerprint
                    and completed.get("source_run_spec_sha256")
                    == sha256_file(spec_path)
                    and completed_output.is_file()
                    and completed.get("semantic_polygons_sha256")
                    == sha256_file(completed_output)
                    and completed_report.is_file()
                    and completed.get("report_sha256") == sha256_file(completed_report)
                ):
                    if bool(args.activate_review):
                        _activate_review(
                            run_dir, stream_id=stream_id, manifest=completed
                        )
                    emit(
                        "fragmentation_postprocess_resumed",
                        run_id=spec["run_id"],
                        stream_id=stream_id,
                        output_dir=str(output_root),
                        activated=bool(args.activate_review),
                    )
                    return completed
            except (KeyError, OSError, TypeError, ValueError):
                pass
        mask_outputs = output_root / "regularized_raster_parts"
        mask_reports_root = output_root / "partition_reports" / "masks"
        vector_outputs = output_root / "polygon_parts"
        vector_reports_root = output_root / "partition_reports" / "vectors"
        mask_reports: list[dict[str, Any]] = []
        if args.stage in {"all", "masks"}:
            emit(
                "fragmentation_mask_stage_started",
                total=len(mask_paths),
                workers=workers,
                policy_version=POLICY_VERSION,
            )
            payloads = []
            for path in mask_paths:
                row, col = partition_key(path)
                neighbor_map = {
                    f"{row + dr},{col + dc}": str(path_map[(row + dr, col + dc)])
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (row + dr, col + dc) in path_map
                }
                payloads.append(
                    {
                        "center_path": str(path),
                        "output_path": str(mask_outputs / path.name),
                        "report_path": str(
                            mask_reports_root / f"{partition_id(path)}.json"
                        ),
                        "path_map": neighbor_map,
                        "buffer_pixels": int(args.buffer_pixels),
                        "resume": resume,
                    }
                )
            mask_reports = _run_bounded_processes(
                payloads,
                process_mask_partition,
                workers=workers,
                event_prefix="fragmentation_mask",
            )
        else:
            for path in mask_paths:
                report_path = mask_reports_root / f"{partition_id(path)}.json"
                output_path = mask_outputs / path.name
                if not report_path.is_file() or not output_path.is_file():
                    raise FragmentationPostprocessError(
                        f"mask stage is incomplete: {partition_id(path)}"
                    )
                report = dict(load_json(report_path))
                if report.get("output_sha256") != sha256_file(output_path):
                    raise FragmentationPostprocessError(
                        f"regularized mask changed: {output_path}"
                    )
                mask_reports.append(report)
        mask_summary = _aggregate_mask_reports(mask_reports)
        if not mask_summary["passed"]:
            raise FragmentationPostprocessError(
                "V3 semantic safety failed: "
                + json.dumps(mask_summary, ensure_ascii=False, separators=(",", ":"))
            )
        regularized_paths = [mask_outputs / path.name for path in mask_paths]
        mask_vrt = output_root / "mask_mosaic.vrt"
        if args.stage in {"all", "masks"}:
            build_vrt(mask_vrt, regularized_paths)

        if args.stage == "masks":
            result = {
                "schema_version": 1,
                "status": "masks_ready",
                "run_id": spec["run_id"],
                "stream_id": stream_id,
                "policy": policy_snapshot(),
                "mask_mosaic": str(mask_vrt),
                "mask_summary": mask_summary,
            }
            write_atomic_json(output_root / REPORT_NAME, result)
            emit("fragmentation_mask_stage_finished", **result)
            return result

        emit(
            "fragmentation_vector_stage_started",
            total=len(mask_paths),
            workers=workers,
        )
        vector_payloads = [
            {
                "run_id": spec["run_id"],
                "stream_id": stream_id,
                "profile_id": stream["profile_id"],
                "model_version": stream.get("version", ""),
                "partition_id": partition_id(source),
                "mask_path": str(mask_outputs / source.name),
                "confidence_path": str(confidence_path(source)),
                "output_path": str(vector_outputs / f"{partition_id(source)}.gpkg"),
                "report_path": str(
                    vector_reports_root / f"{partition_id(source)}.json"
                ),
                "resume": resume,
            }
            for source in mask_paths
        ]
        vector_reports = _run_bounded_processes(
            vector_payloads,
            process_vector_partition,
            workers=workers,
            event_prefix="fragmentation_vector",
        )
        vector_reports.sort(key=lambda item: str(item["partition_id"]))
        part_paths = [
            vector_outputs / f"{partition_id(source)}.gpkg" for source in mask_paths
        ]
        with rasterio.open(regularized_paths[0]) as reference:
            crs_wkt = reference.crs.to_wkt()
        semantic_path = output_root / "semantic_polygons.gpkg"
        assembly = _assemble_vectors(
            semantic_path,
            part_paths,
            crs_wkt=crs_wkt,
        )
        expected_features = sum(int(item["feature_count"]) for item in vector_reports)
        if int(assembly["feature_count"]) != expected_features:
            raise FragmentationPostprocessError(
                f"final feature count mismatch: {assembly['feature_count']}/{expected_features}"
            )
        report = {
            "schema_version": 1,
            "status": "passed",
            "run_id": spec["run_id"],
            "stream_id": stream_id,
            "source_run_spec": str(spec_path),
            "source_run_spec_sha256": sha256_file(spec_path),
            "source_inventory_fingerprint": source_inventory_fingerprint,
            "source_mask_mosaic": str(source_root / "mask_mosaic.vrt"),
            "source_confidence_mosaic": str(source_root / "confidence_mosaic.vrt"),
            "policy": policy_snapshot(),
            "buffer_pixels": int(args.buffer_pixels),
            "workers": workers,
            "mask_mosaic": str(mask_vrt),
            "semantic_polygons": str(semantic_path),
            "semantic_polygons_sha256": assembly["sha256"],
            "semantic_polygon_feature_count": assembly["feature_count"],
            "partition_count": len(mask_paths),
            "mask_summary": mask_summary,
            "vector_part_feature_count": expected_features,
            "validation": {
                "passed": True,
                "gpkg_integrity_check": assembly["integrity_check"],
                "protected_classes_unchanged": mask_summary[
                    "protected_classes_unchanged"
                ],
                "disappeared_class_codes": mask_summary["disappeared_class_codes"],
            },
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        report_path = output_root / REPORT_NAME
        write_atomic_json(report_path, report)
        manifest = {
            "schema_version": 1,
            "status": "passed",
            "run_id": spec["run_id"],
            "stream_id": stream_id,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "source_run_spec_sha256": sha256_file(spec_path),
            "source_inventory_fingerprint": source_inventory_fingerprint,
            "semantic_polygons": str(semantic_path.resolve()),
            "semantic_polygons_layer": LAYER_NAME,
            "semantic_polygons_sha256": assembly["sha256"],
            "semantic_polygon_feature_count": assembly["feature_count"],
            "mask_mosaic": str(mask_vrt.resolve()),
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
        }
        write_atomic_json(manifest_path, manifest)
        if bool(args.activate_review):
            activated = _activate_review(
                run_dir, stream_id=stream_id, manifest=manifest
            )
            manifest["activated_run_manifest"] = str(activated)
        emit(
            "fragmentation_postprocess_finished",
            run_id=spec["run_id"],
            stream_id=stream_id,
            output_dir=str(output_root),
            feature_count=assembly["feature_count"],
            activated=bool(args.activate_review),
        )
        return manifest
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume V3 fragmentation repair from committed Fusion rasters"
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stage", choices=("all", "masks", "vectors"), default="all")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--buffer-pixels", type=int, default=DEFAULT_BUFFER_PIXELS)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--activate-review", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.buffer_pixels < 1:
        raise SystemExit("--workers and --buffer-pixels must be positive")
    try:
        run_postprocess(args)
        return 0
    except Exception as error:
        emit("fragmentation_postprocess_failed", error=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
