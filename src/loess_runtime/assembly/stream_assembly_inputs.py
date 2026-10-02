"""Validated inputs and ready-Stream reuse for assembly."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fiona
from fiona.crs import CRS

from labeling_tool.shared.contracts.run_spec import load_json, sha256_file
from loess_runtime.assembly.assembly_errors import StreamAssemblyError
from loess_runtime.assembly.stream_coverage_validation import (
    AppendCoverageEvent,
    publish_coverage_validation,
    validate_exact_range_coverage,
)
from loess_runtime.geometry.vector_data_plane import read_geoparquet

if TYPE_CHECKING:
    from labeling_tool.shared.state.artifact_repository import ArtifactRepository


ASSEMBLY_VALIDATION_MAX_IN_FLIGHT = 32


@contextmanager
def readonly_geopackage(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def file_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise StreamAssemblyError(f"file changed while it was being checked: {path}")
    return {
        "byte_count": int(after.st_size),
        "sha256": str(digest),
        "mtime_ns": int(after.st_mtime_ns),
    }


def validate_existing_gpkg(
    path: Path,
    *,
    layer: str,
    schema: Mapping[str, Any],
    crs: Any,
    identity: Mapping[str, str],
    expected_feature_count: int | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise StreamAssemblyError(f"resume input is missing: {path}")
    with readonly_geopackage(path) as connection:
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
    if integrity != ["ok"]:
        raise StreamAssemblyError(
            f"resume input GeoPackage integrity check failed: {path}; {integrity[:3]}"
        )
    if layer not in fiona.listlayers(path):
        raise StreamAssemblyError(f"resume input layer is missing: {path}::{layer}")
    with fiona.open(path, layer=layer) as source:
        actual_properties = source.schema.get("properties") or {}
        expected_properties = schema.get("properties") or {}
        actual_fields = set(actual_properties)
        expected_fields = set(expected_properties)
        if actual_fields != expected_fields:
            raise StreamAssemblyError(
                f"resume input fields changed: {path}::{layer}; "
                f"expected={sorted(expected_fields)}, actual={sorted(actual_fields)}"
            )
        type_aliases = {
            "int32": "int",
            "int64": "int",
            "float32": "float",
            "float64": "float",
        }
        mismatched_types = []
        for key in sorted(expected_fields):
            expected_type = str(expected_properties[key]).split(":", 1)[0].lower()
            actual_type = str(actual_properties[key]).split(":", 1)[0].lower()
            expected_type = type_aliases.get(expected_type, expected_type)
            actual_type = type_aliases.get(actual_type, actual_type)
            if actual_type != expected_type:
                mismatched_types.append(
                    f"{key}:{actual_properties[key]}!={expected_properties[key]}"
                )
        if mismatched_types:
            raise StreamAssemblyError(
                f"resume input field types changed: {path}::{layer}; {mismatched_types}"
            )
        actual_geometry = str(source.schema.get("geometry") or "")
        expected_geometry = str(schema.get("geometry") or "")
        if actual_geometry != expected_geometry:
            raise StreamAssemblyError(
                f"resume input geometry type changed: {path}::{layer}; "
                f"expected={expected_geometry}, actual={actual_geometry}"
            )
        actual_crs = CRS.from_user_input(source.crs_wkt or source.crs)
        expected_crs = CRS.from_user_input(crs)
        if actual_crs != expected_crs:
            raise StreamAssemblyError(
                f"resume input CRS changed: {path}::{layer}; "
                f"expected={expected_crs}, actual={actual_crs}"
            )
        feature_count = len(source)
        if expected_feature_count is not None and feature_count != int(
            expected_feature_count
        ):
            raise StreamAssemblyError(
                f"resume input feature count changed: {path}::{layer}; "
                f"expected={expected_feature_count}, actual={feature_count}"
            )
        if next(iter(source), None) is None:
            raise StreamAssemblyError(f"resume input layer is empty: {path}::{layer}")
    if not layer.replace("_", "").isalnum():
        raise StreamAssemblyError(f"unsafe GeoPackage layer name: {layer}")
    clauses = [f"COALESCE(CAST(\"{key}\" AS TEXT), '') != ?" for key in identity]
    with readonly_geopackage(path) as connection:
        mismatched = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{layer}" WHERE {" OR ".join(clauses)}',
                tuple(str(value) for value in identity.values()),
            ).fetchone()[0]
        )
    if mismatched:
        raise StreamAssemblyError(
            f"resume input contains {mismatched} rows for another run or stream: "
            f"{path}::{layer}"
        )
    return {
        "path": str(path),
        "layer": layer,
        "feature_count": int(feature_count),
        **file_fingerprint(path),
    }


def assert_fingerprint_unchanged(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = file_fingerprint(path)
    if int(actual["byte_count"]) != int(expected["byte_count"]) or str(
        actual["sha256"]
    ) != str(expected["sha256"]):
        raise StreamAssemblyError(
            f"resume input changed during report assembly: {path}"
        )


def assembly_validation_workers(
    spec: Mapping[str, Any],
    item_count: int,
) -> int:
    configured = int(
        (spec.get("scaling") or {}).get("assembly_validation_workers")
        or min(8, max(2, (os.cpu_count() or 2) - 2))
    )
    return max(1, min(configured, ASSEMBLY_VALIDATION_MAX_IN_FLIGHT, item_count))


def parallel_validate_summary_artifacts(
    items: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    edge_schema: Mapping[str, Any],
    crs: Any,
    run_id: str,
    stream_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Validate immutable report/edge shards with a bounded future window."""
    if not items:
        return {
            "workers": 0,
            "peak_in_flight": 0,
            "artifact_count": 0,
            "elapsed_sec": 0.0,
        }
    started_at = time.monotonic()
    peak_in_flight = 0

    def validate(item: Mapping[str, Any]) -> None:
        artifact = item["artifact"]
        path = Path(str(artifact["path"]))
        if str(item["kind"]) == "edge":
            manifest, table = read_geoparquet(path)
            if int(manifest["feature_count"]) != int(item["expected_feature_count"]):
                raise StreamAssemblyError(f"unit fitted-edge count changed: {path}")
            if not set(edge_schema["properties"]).issubset(set(table.column_names)):
                raise StreamAssemblyError(f"unit fitted-edge fields changed: {path}")
            fingerprint = file_fingerprint(path)
        else:
            if not path.is_file():
                raise StreamAssemblyError(f"unit report Artifact is missing: {path}")
            fingerprint = file_fingerprint(path)
        if int(artifact["byte_count"]) != int(fingerprint["byte_count"]) or str(
            artifact["sha256"]
        ) != str(fingerprint["sha256"]):
            raise StreamAssemblyError(f"unit {item['kind']} Artifact changed: {path}")

    iterator = iter(items)
    pending: set[Future[None]] = set()
    completed_count = 0
    with ThreadPoolExecutor(
        max_workers=max(1, int(workers)),
        thread_name_prefix="assembly-validator",
    ) as executor:
        while len(pending) < ASSEMBLY_VALIDATION_MAX_IN_FLIGHT:
            try:
                pending.add(executor.submit(validate, next(iterator)))
            except StopIteration:
                break
        peak_in_flight = max(peak_in_flight, len(pending))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, len(items))
            while len(pending) < ASSEMBLY_VALIDATION_MAX_IN_FLIGHT:
                try:
                    pending.add(executor.submit(validate, next(iterator)))
                except StopIteration:
                    break
            peak_in_flight = max(peak_in_flight, len(pending))
    return {
        "workers": int(workers),
        "peak_in_flight": int(peak_in_flight),
        "artifact_count": len(items),
        "elapsed_sec": round(time.monotonic() - started_at, 3),
    }


def validated_summary_inputs(
    spec: Mapping[str, Any],
    *,
    run_id: str,
    stream_id: str,
    expected_units: int,
    report_artifacts: list[Mapping[str, Any]],
    report_summaries: list[Mapping[str, Any]],
    artifacts: ArtifactRepository,
    edge_schema: Mapping[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = report_summaries
    if len(summaries) != int(expected_units):
        raise StreamAssemblyError(
            "unit report summaries are incomplete; rerun unit fitting with the "
            f"current Ubuntu runtime: {len(summaries)}/{expected_units}"
        )
    reports_by_unit = {
        str(artifact["unit_id"]): dict(artifact) for artifact in report_artifacts
    }
    summaries_by_unit = {
        str(summary["unit_id"]): dict(summary) for summary in summaries
    }
    if set(reports_by_unit) != set(summaries_by_unit):
        raise StreamAssemblyError(
            "unit report summaries do not match ready report Artifacts"
        )
    validation_items: list[dict[str, Any]] = []
    for unit_id in sorted(summaries_by_unit):
        summary = summaries_by_unit[unit_id]
        artifact = reports_by_unit[unit_id]
        if (
            Path(str(summary["report_path"])).resolve()
            != Path(str(artifact["path"])).resolve()
            or int(summary["report_byte_count"]) != int(artifact["byte_count"])
            or str(summary["report_sha256"]) != str(artifact["sha256"])
        ):
            raise StreamAssemblyError(
                f"unit report summary fingerprint changed: {unit_id}"
            )
        validation_items.append({"kind": "report", "artifact": artifact})

    edge_artifacts = artifacts.artifacts_for_stream(
        run_id,
        stream_id,
        kind="unit_fitted_edges_geoparquet",
    )
    edges_by_unit = {
        str(artifact["unit_id"]): dict(artifact) for artifact in edge_artifacts
    }
    expected_edge_units = {
        unit_id
        for unit_id, summary in summaries_by_unit.items()
        if int(summary["fitted_edge_count"]) > 0
    }
    if set(edges_by_unit) != expected_edge_units:
        missing = sorted(expected_edge_units - set(edges_by_unit))
        unexpected = sorted(set(edges_by_unit) - expected_edge_units)
        raise StreamAssemblyError(
            "unit fitted-edge Artifacts do not match database summaries; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for unit_id in sorted(edges_by_unit):
        validation_items.append(
            {
                "kind": "edge",
                "artifact": edges_by_unit[unit_id],
                "expected_feature_count": int(
                    summaries_by_unit[unit_id]["fitted_edge_count"]
                ),
            }
        )
    workers = assembly_validation_workers(spec, len(validation_items))
    validation = parallel_validate_summary_artifacts(
        validation_items,
        workers=workers,
        edge_schema=edge_schema,
        crs=spec["raster"]["crs"],
        run_id=run_id,
        stream_id=stream_id,
        progress_callback=progress_callback,
    )
    return (
        [edges_by_unit[unit_id] for unit_id in sorted(edges_by_unit)],
        validation,
    )


def reuse_ready_assembly(
    *,
    spec: Mapping[str, Any],
    run_id: str,
    stream_id: str,
    stream_status: str,
    root: Path,
    artifacts: ArtifactRepository,
    append_event: AppendCoverageEvent,
) -> dict[str, Any] | None:
    if stream_status not in {"ready", "failed"}:
        return None
    paths = {
        "semantic_polygons_raw": root / "semantic_polygons_raw.gpkg",
        "semantic_polygons": root / "semantic_polygons.gpkg",
        "boundary_fitting_report": root / "boundary_fitting_report.json",
        "fitted_edges": root / "fitted_edges.gpkg",
    }
    for kind, path in paths.items():
        artifact = artifacts.artifact_for_stream_unit(
            run_id,
            stream_id,
            "assembled",
            kind,
        )
        if artifact is None or artifact["status"] != "ready" or not path.is_file():
            # A failed Stream may legitimately contain only the raw/formal
            # pair from a previously completed assembly attempt.  In that
            # state the caller must continue through the normal full or
            # report-resume validation path; treating it as a corrupt ready
            # Stream prevents the strict unit-report checks from running.
            # A Stream still marked ready, however, must remain an immutable
            # completed set and missing members are a hard integrity error.
            if stream_status == "failed":
                return None
            raise StreamAssemblyError(
                f"ready stream is missing assembled Artifact: {stream_id}/{kind}"
            )
        if int(artifact["byte_count"]) != path.stat().st_size or str(
            artifact["sha256"]
        ) != sha256_file(path):
            raise StreamAssemblyError(
                f"ready assembled Artifact changed on disk: {path}"
            )
    report = dict(load_json(paths["boundary_fitting_report"]))
    if (
        report.get("status") != "passed"
        or (report.get("validation") or {}).get("passed") is not True
    ):
        raise StreamAssemblyError(
            f"ready stream has a failed boundary report: {stream_id}"
        )
    coverage = dict(report.get("coverage_validation") or {})
    if not coverage:
        coverage = validate_exact_range_coverage(
            paths["semantic_polygons"],
            layer="semantic_polygons",
            spec=spec,
        )
        publish_coverage_validation(
            append_event,
            run_id,
            stream_id,
            coverage,
        )
        if coverage["status"] == "failed":
            raise StreamAssemblyError(
                "ready stream failed exact coverage validation: "
                f"gap={coverage['gap_area_m2']:.6g} m2, "
                f"overlap={coverage['overlap_area_m2']:.6g} m2, "
                f"outside={coverage['outside_area_m2']:.6g} m2"
            )
        report["coverage_validation"] = coverage
    report["assembly_mode"] = "reused"
    report.setdefault(
        "report_processed_count",
        int(report.get("unit_count") or 0),
    )
    report.setdefault(
        "report_queue_capacity",
        ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
    )
    report.setdefault("report_peak_loaded_count", 0)
    report.setdefault("report_summary_source", "run_state_database")
    report.setdefault("report_json_parse_count", 0)
    report.setdefault("object_link_count", 0)
    report["recovered_stream_status"] = stream_status
    print(
        json.dumps(
            {"event": "stream_assembly_reused", **report},
            separators=(",", ":"),
        )
    )
    return report
