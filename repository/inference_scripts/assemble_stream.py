"""Stream unit outputs into final GPKGs and assign disk-backed object IDs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

import fiona
from fiona.crs import CRS
from shapely.geometry import shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.ownership_neighbors import ownership_neighbors
from labeling_tool.core.run_state_db import RunStateDB
from labeling_tool.core.run_spec import sha256_file

from deployment_config import load_json
from difference_runtime import apply_accepted_difference
from semantic_batch import _atomic_json
from work_package_runtime import _commit_artifact


class StreamAssemblyError(RuntimeError):
    pass


ASSEMBLY_VALIDATION_MAX_IN_FLIGHT = 32
OBJECT_ID_BATCH_SIZE = 512


def _stream_root(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    run_dir = Path(spec["run_dir"])
    if stream["kind"] == "model":
        return run_dir / "models" / str(stream["model_id"])
    return run_dir / "fusion" / str(stream["profile_id"])


def _read_features(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with fiona.open(path, layer="polygons") as source:
        for feature in source:
            result.append(
                {
                    "geometry": shape(feature["geometry"]),
                    "properties": dict(feature["properties"]),
                }
            )
    return result


def _atomic_gpkg(path: Path, layer: str, schema, crs, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.gpkg"
    temporary.unlink(missing_ok=True)
    try:
        with fiona.open(
            temporary,
            "w",
            driver="GPKG",
            layer=layer,
            schema=schema,
            crs=CRS.from_user_input(crs),
        ) as destination:
            writer(destination)
        with sqlite3.connect(temporary) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise StreamAssemblyError(f"GeoPackage integrity check failed: {temporary}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _file_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise StreamAssemblyError(f"file changed while it was being checked: {path}")
    return {
        "byte_count": int(after.st_size),
        "sha256": str(digest),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _validate_existing_gpkg(
    path: Path,
    *,
    layer: str,
    schema: Mapping[str, Any],
    crs: Any,
    identity: Mapping[str, str],
    expected_feature_count: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise StreamAssemblyError(f"resume input is missing: {path}")
    with _readonly_sqlite(path) as connection:
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
    if integrity != ["ok"]:
        raise StreamAssemblyError(
            f"resume input GeoPackage integrity check failed: {path}; "
            f"{integrity[:3]}"
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
                f"resume input field types changed: {path}::{layer}; "
                f"{mismatched_types}"
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
        if feature_count != int(expected_feature_count):
            raise StreamAssemblyError(
                f"resume input feature count changed: {path}::{layer}; "
                f"expected={expected_feature_count}, actual={feature_count}"
            )
        if next(iter(source), None) is None:
            raise StreamAssemblyError(f"resume input layer is empty: {path}::{layer}")
    if not layer.replace("_", "").isalnum():
        raise StreamAssemblyError(f"unsafe GeoPackage layer name: {layer}")
    clauses = [
        f"COALESCE(CAST(\"{key}\" AS TEXT), '') != ?"
        for key in identity
    ]
    with _readonly_sqlite(path) as connection:
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
        **_file_fingerprint(path),
    }


def _resolved_object_state(
    state_db: str | Path,
    run_id: str,
    stream_id: str,
) -> tuple[int, int]:
    with _readonly_sqlite(Path(state_db)) as connection:
        row = connection.execute(
            """SELECT COUNT(*) AS part_count,
                      COALESCE(SUM(CASE WHEN object_id IS NULL OR object_id=''
                                        THEN 1 ELSE 0 END), 0) AS unresolved_count,
                      COALESCE(SUM(CASE WHEN parent_id=part_id THEN 1 ELSE 0 END), 0)
                        AS object_count
               FROM object_nodes WHERE run_id=? AND stream_id=?""",
            (str(run_id), str(stream_id)),
        ).fetchone()
    part_count = int(row[0])
    unresolved_count = int(row[1])
    object_count = int(row[2])
    if part_count < 1 or object_count < 1 or unresolved_count:
        raise StreamAssemblyError(
            "resume requires existing resolved object IDs; "
            f"parts={part_count}, objects={object_count}, unresolved={unresolved_count}"
        )
    return part_count, object_count


def _assert_fingerprint_unchanged(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = _file_fingerprint(path)
    if (
        int(actual["byte_count"]) != int(expected["byte_count"])
        or str(actual["sha256"]) != str(expected["sha256"])
    ):
        raise StreamAssemblyError(f"resume input changed during report assembly: {path}")


def _feature_batches(path: str | Path, *, size: int = OBJECT_ID_BATCH_SIZE):
    batch = []
    with fiona.open(path, layer="polygons") as source:
        for feature in source:
            batch.append(feature)
            if len(batch) >= int(size):
                yield batch
                batch = []
    if batch:
        yield batch


def _assembly_validation_workers(
    spec: Mapping[str, Any],
    item_count: int,
) -> int:
    configured = int(
        (spec.get("scaling") or {}).get("assembly_validation_workers")
        or min(8, max(2, (os.cpu_count() or 2) - 2))
    )
    return max(1, min(configured, ASSEMBLY_VALIDATION_MAX_IN_FLIGHT, item_count))


def _parallel_validate_summary_artifacts(
    items: list[Mapping[str, Any]],
    *,
    workers: int,
    edge_schema: Mapping[str, Any],
    crs: Any,
    run_id: str,
    stream_id: str,
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
            fingerprint = _validate_existing_gpkg(
                path,
                layer="fitted_edges",
                schema=edge_schema,
                crs=crs,
                identity={
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "unit_id": str(artifact["unit_id"]),
                },
                expected_feature_count=int(item["expected_feature_count"]),
            )
        else:
            if not path.is_file():
                raise StreamAssemblyError(f"unit report Artifact is missing: {path}")
            fingerprint = _file_fingerprint(path)
        if (
            int(artifact["byte_count"]) != int(fingerprint["byte_count"])
            or str(artifact["sha256"]) != str(fingerprint["sha256"])
        ):
            raise StreamAssemblyError(
                f"unit {item['kind']} Artifact changed: {path}"
            )

    iterator = iter(items)
    pending = set()
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


def _validated_summary_inputs(
    spec: Mapping[str, Any],
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    expected_units: int,
    report_artifacts: list[Mapping[str, Any]],
    edge_schema: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = database.unit_report_summaries(run_id, stream_id)
    if len(summaries) != int(expected_units):
        raise StreamAssemblyError(
            "unit report summaries are incomplete; rerun unit fitting with the "
            f"current Ubuntu runtime: {len(summaries)}/{expected_units}"
        )
    reports_by_unit = {
        str(artifact["unit_id"]): dict(artifact)
        for artifact in report_artifacts
    }
    summaries_by_unit = {
        str(summary["unit_id"]): dict(summary)
        for summary in summaries
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
        validation_items.append(
            {"kind": "report", "artifact": artifact}
        )

    edge_artifacts = database.artifacts_for_stream(
        run_id,
        stream_id,
        kind="unit_fitted_edges",
    )
    edges_by_unit = {
        str(artifact["unit_id"]): dict(artifact)
        for artifact in edge_artifacts
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
            "unit fitted-edge Artifacts do not match SQLite summaries; "
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
    workers = _assembly_validation_workers(spec, len(validation_items))
    validation = _parallel_validate_summary_artifacts(
        validation_items,
        workers=workers,
        edge_schema=edge_schema,
        crs=spec["raster"]["crs"],
        run_id=run_id,
        stream_id=stream_id,
    )
    return (
        [edges_by_unit[unit_id] for unit_id in sorted(edges_by_unit)],
        validation,
    )


def _reuse_ready_assembly(
    spec: Mapping[str, Any],
    stream: Mapping[str, Any],
    database: RunStateDB,
) -> dict[str, Any] | None:
    run_id = str(spec["run_id"])
    stream_id = str(stream["stream_id"])
    rows = {
        str(row["stream_id"]): row for row in database.stream_rows(run_id)
    }
    if str((rows.get(stream_id) or {}).get("status") or "") != "ready":
        return None
    root = _stream_root(spec, stream)
    paths = {
        "semantic_polygons_raw": root / "semantic_polygons_raw.gpkg",
        "semantic_polygons": root / "semantic_polygons.gpkg",
        "boundary_fitting_report": root / "boundary_fitting_report.json",
        "fitted_edges": root / "fitted_edges.gpkg",
    }
    for kind, path in paths.items():
        artifact = database.artifact_for_stream_unit(
            run_id,
            stream_id,
            "assembled",
            kind,
        )
        if artifact is None or artifact["status"] != "ready" or not path.is_file():
            raise StreamAssemblyError(
                f"ready stream is missing assembled Artifact: {stream_id}/{kind}"
            )
        if (
            int(artifact["byte_count"]) != path.stat().st_size
            or str(artifact["sha256"]) != sha256_file(path)
        ):
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
    report.setdefault("report_summary_source", "sqlite")
    report.setdefault("report_json_parse_count", 0)
    report["object_link_count"] = database.object_link_count(run_id, stream_id)
    print(
        json.dumps(
            {"event": "stream_assembly_reused", **report},
            separators=(",", ":"),
        )
    )
    return report


def _link_neighbor_parts(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    formal_by_unit: Mapping[str, str],
    units: list[Mapping[str, Any]],
    tolerance: float,
) -> int:
    linked = 0
    for left_unit, right_unit in ownership_neighbors(units):
        left_features = _read_features(formal_by_unit[left_unit])
        right_features = _read_features(formal_by_unit[right_unit])
        right_geometries = [item["geometry"] for item in right_features]
        tree = STRtree(right_geometries)
        for left in left_features:
            left_class = int(left["properties"]["class_code"])
            for index in tree.query(left["geometry"]):
                right = right_features[int(index)]
                if int(right["properties"]["class_code"]) != left_class:
                    continue
                shared = left["geometry"].boundary.intersection(right["geometry"].boundary)
                if shared.length <= tolerance:
                    continue
                if database.add_object_link(
                    run_id,
                    stream_id,
                    str(left["properties"]["polygon_id"]),
                    str(right["properties"]["polygon_id"]),
                    left_class,
                ):
                    linked += 1
    return linked


def _assemble_stream_impl(
    run_spec_path: str | Path,
    stream_id: str,
    *,
    resume_from_reports: bool = False,
) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 2:
        raise StreamAssemblyError("stream assembly requires run_spec schema 2")
    run_id = str(spec["run_id"])
    database = RunStateDB(spec["state_db"])
    streams = [item for item in spec["streams"] if item["stream_id"] == stream_id]
    if len(streams) != 1:
        raise StreamAssemblyError(f"unknown result stream: {stream_id}")
    stream = streams[0]
    boundary = spec.get("boundary_fitting") or {}
    fit_mode = str(boundary.get("mode") or "")
    if fit_mode != "divider_cubic_bspline_adaptive_v2":
        raise StreamAssemblyError(
            "only divider_cubic_bspline_adaptive_v2 is supported "
            "by the current runtime"
        )
    smoothing_enabled = bool(boundary.get("enabled", True))
    fit_version = (
        "divider_cubic_bspline_adaptive_v2"
        if smoothing_enabled else "raw_polygonize_v1"
    )
    reused = _reuse_ready_assembly(spec, stream, database)
    if reused is not None:
        return reused
    database.set_stream_status(
        run_id,
        stream_id,
        "assembling",
        error="",
    )
    counts = database.stream_unit_counts(run_id, stream_id)
    expected_units = sum(counts.values())
    if expected_units < 1 or counts != {"ready": expected_units}:
        raise StreamAssemblyError(f"stream units are not all ready: {counts}")
    units = database.spatial_units(run_id)
    formal_artifacts = database.artifacts_for_stream(
        run_id, stream_id, kind="unit_formal"
    )
    raw_artifacts = database.artifacts_for_stream(run_id, stream_id, kind="unit_raw")
    report_artifacts = database.artifacts_for_stream(
        run_id, stream_id, kind="unit_boundary_report"
    )
    if not (
        len(formal_artifacts) == len(raw_artifacts) == len(report_artifacts) == expected_units
    ):
        raise StreamAssemblyError("unit Artifact count does not match ready unit count")
    formal_by_unit = {str(item["unit_id"]): str(item["path"]) for item in formal_artifacts}
    raw_by_unit = {str(item["unit_id"]): str(item["path"]) for item in raw_artifacts}

    transform = spec["raster"]["transform"]
    root = _stream_root(spec, stream)
    raw_path = root / "semantic_polygons_raw.gpkg"
    formal_path = root / "semantic_polygons.gpkg"
    report_path = root / "boundary_fitting_report.json"
    fitted_edges_path = root / "fitted_edges.gpkg"
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    raw_schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
            "unit_id": "str:96",
            "polygon_id": "str:96",
            "class_code": "int",
        },
    }

    formal_schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "result_stream_id": "str:96",
            "result_kind": "str:16",
            "model_id": "str:64",
            "fusion_profile_id": "str:64",
            "object_id": "str:64",
            "part_id": "str:96",
            "class_code": "int",
            "class_name": "str:64",
            "confidence_mean": "float",
            "confidence_std": "float",
            "model_version": "str:64",
            "source": "str:32",
            "fit_changed": "int",
            "fit_methods": "str:64",
            "fit_version": "str:40",
            "fit_status": "str:24",
            "origin_unit_ids": "str:254",
            "vertex_count_before": "int",
            "vertex_count_after": "int",
            "max_shift_px": "float",
            "mean_shift_px": "float",
            "area_change_ratio": "float",
            "created_at": "str:40",
        },
    }
    edge_schema = {
        "geometry": "LineString",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
            "unit_id": "str:96",
            "chain_id": "str:96",
            "method": "str:24",
            "status": "str:32",
            "max_shift": "float",
            "dense_vtx": "int",
            "sparse_vtx": "int",
            "chord_err": "float",
            "arc_len": "float",
        },
    }
    edge_artifacts, summary_validation = _validated_summary_inputs(
        spec,
        database,
        run_id,
        stream_id,
        expected_units,
        report_artifacts,
        edge_schema,
    )
    summary_aggregate = database.unit_report_summary_aggregate(
        run_id,
        stream_id,
    )
    if int(summary_aggregate["unit_count"]) != expected_units:
        raise StreamAssemblyError(
            "SQLite report aggregate does not cover every ready unit"
        )
    model_id = str(stream.get("model_id") or "")
    profile_id = str(stream.get("profile_id") or "")
    version = str(stream.get("version") or "")
    ownership_validation = {
        "passed": True,
        "scope": "all_output_polygons",
        "invalid_count": 0,
    }
    resume_inputs: dict[str, dict[str, Any]] = {}

    if resume_from_reports:
        part_count, object_count = _resolved_object_state(
            spec["state_db"],
            run_id,
            stream_id,
        )
        link_count = database.object_link_count(run_id, stream_id)
        resume_inputs["raw"] = _validate_existing_gpkg(
            raw_path,
            layer="semantic_polygons_raw",
            schema=raw_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "stream_id": stream_id},
            expected_feature_count=part_count,
        )
        resume_inputs["formal"] = _validate_existing_gpkg(
            formal_path,
            layer="semantic_polygons",
            schema=formal_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "result_stream_id": stream_id},
            expected_feature_count=part_count,
        )
        print(
            json.dumps(
                {
                    "event": "stream_report_resume_inputs_validated",
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "feature_count": part_count,
                    "object_count": object_count,
                    "raw_sha256": resume_inputs["raw"]["sha256"],
                    "formal_sha256": resume_inputs["formal"]["sha256"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    else:
        for artifact in formal_artifacts:
            unit_id = str(artifact["unit_id"])
            for features in _feature_batches(artifact["path"]):
                database.register_object_parts(
                    run_id,
                    stream_id,
                    (
                        {
                            "part_id": str(feature["properties"]["polygon_id"]),
                            "class_code": int(feature["properties"]["class_code"]),
                            "unit_id": unit_id,
                        }
                        for feature in features
                    ),
                )
        pixel_tolerance = (
            max(abs(float(transform[0])), abs(float(transform[4]))) * 1e-6
        )
        _link_neighbor_parts(
            database,
            run_id,
            stream_id,
            formal_by_unit,
            units,
            pixel_tolerance,
        )
        link_count = database.object_link_count(run_id, stream_id)
        object_count = database.resolve_object_components(run_id, stream_id)
        class_snapshot = load_json(Path(spec["class_mapping_snapshot"]))
        class_names = class_snapshot["class_mapping"]

        def write_raw(destination):
            def records():
                for unit in units:
                    unit_id = str(unit["unit_id"])
                    with fiona.open(
                        raw_by_unit[unit_id],
                        layer="polygons",
                    ) as source:
                        for feature in source:
                            yield {
                                "geometry": feature["geometry"],
                                "properties": {
                                    "run_id": run_id,
                                    "stream_id": stream_id,
                                    "unit_id": unit_id,
                                    "polygon_id": str(
                                        feature["properties"]["polygon_id"]
                                    ),
                                    "class_code": int(
                                        feature["properties"]["class_code"]
                                    ),
                                },
                            }

            destination.writerecords(records())

        _atomic_gpkg(
            raw_path,
            "semantic_polygons_raw",
            raw_schema,
            spec["raster"]["crs"],
            write_raw,
        )

        def write_formal(destination):
            def records():
                for unit in units:
                    unit_id = str(unit["unit_id"])
                    for features in _feature_batches(formal_by_unit[unit_id]):
                        part_ids = [
                            str(feature["properties"]["polygon_id"])
                            for feature in features
                        ]
                        object_ids = database.object_ids_for_parts(
                            run_id,
                            stream_id,
                            part_ids,
                        )
                        for feature in features:
                            geometry = shape(feature["geometry"])
                            if (
                                geometry.is_empty
                                or not geometry.is_valid
                                or geometry.area <= 0
                            ):
                                raise StreamAssemblyError(
                                    "formal output contains invalid geometry: "
                                    f"{unit_id}"
                                )
                            properties = feature["properties"]
                            part_id = str(properties["polygon_id"])
                            class_code = int(properties["class_code"])
                            yield {
                                "geometry": feature["geometry"],
                                "properties": {
                                    "run_id": run_id,
                                    "result_stream_id": stream_id,
                                    "result_kind": str(stream["kind"]),
                                    "model_id": model_id,
                                    "fusion_profile_id": profile_id,
                                    "object_id": object_ids[part_id],
                                    "part_id": part_id,
                                    "class_code": class_code,
                                    "class_name": str(
                                        class_names[str(class_code)]
                                    ),
                                    "confidence_mean": float(
                                        properties.get("conf_mean", 0.0)
                                    ),
                                    "confidence_std": float(
                                        properties.get("conf_std", 0.0)
                                    ),
                                    "model_version": version,
                                    "source": (
                                        "semantic_model"
                                        if stream["kind"] == "model"
                                        else "semantic_fusion"
                                    ),
                                    "fit_changed": int(
                                        str(properties.get("fit_status"))
                                        == "changed"
                                    ),
                                    "fit_methods": str(
                                        properties.get("fit_method")
                                        or "unchanged"
                                    ),
                                    "fit_version": str(
                                        properties.get("fit_version")
                                        or fit_version
                                    ),
                                    "fit_status": str(
                                        properties.get("fit_status")
                                        or "unchanged"
                                    ),
                                    "origin_unit_ids": unit_id,
                                    "vertex_count_before": int(
                                        properties.get("vtx_before", 0)
                                    ),
                                    "vertex_count_after": int(
                                        properties.get("vtx_after", 0)
                                    ),
                                    "max_shift_px": float(
                                        properties.get("max_shift", 0.0)
                                    ),
                                    "mean_shift_px": float(
                                        properties.get("mean_shift", 0.0)
                                    ),
                                    "area_change_ratio": float(
                                        properties.get("area_ratio", 0.0)
                                    ),
                                    "created_at": now,
                                },
                            }

            destination.writerecords(records())

        _atomic_gpkg(
            formal_path,
            "semantic_polygons",
            formal_schema,
            spec["raster"]["crs"],
            write_formal,
        )

    aggregate = {
        "schema_version": 1,
        "run_id": run_id,
        "stream_id": stream_id,
        "assembly_mode": "report_resume" if resume_from_reports else "full",
        "report_queue_capacity": ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
        "report_summary_source": "sqlite",
        "report_processed_count": expected_units,
        "report_peak_loaded_count": 0,
        "report_json_parse_count": 0,
        "summary_validation_workers": summary_validation["workers"],
        "summary_validation_peak_in_flight": summary_validation[
            "peak_in_flight"
        ],
        "summary_validation_artifact_count": summary_validation[
            "artifact_count"
        ],
        "summary_validation_elapsed_sec": summary_validation["elapsed_sec"],
        "gpkg_write_mode": "gdal_batch_writerecords",
        "object_id_lookup_batch_size": OBJECT_ID_BATCH_SIZE,
        "fitted_edge_shard_count": len(edge_artifacts),
        "status": "passed",
        "smoothing_enabled": smoothing_enabled,
        "unit_count": expected_units,
        "object_count": object_count,
        "object_link_count": link_count,
        "fit_version": fit_version,
        "curve_sampling_spacing_px": float(
            boundary.get("curve_sampling_spacing_px", 0.5)
        ),
        "max_chord_error_limit_px": float(
            boundary.get("max_chord_error_px", 0.25)
        ),
        "max_segment_arc_length_limit_px": float(
            boundary.get("max_segment_arc_length_px", 8.0)
        ),
        "chain_count": int(summary_aggregate["chain_count"]),
        "shared_chain_count": int(summary_aggregate["shared_chain_count"]),
        "spline_count": int(summary_aggregate["spline_count"]),
        "unchanged_count": int(summary_aggregate["unchanged_count"]),
        "skipped_invalid_count": int(
            summary_aggregate["skipped_invalid_count"]
        ),
        "failed_unit_count": int(summary_aggregate["failed_unit_count"]),
        "max_displacement_px": float(
            summary_aggregate["max_displacement_px"]
        ),
        "diagnostic_count": int(summary_aggregate["diagnostic_count"]),
        "fitted_edge_count": int(summary_aggregate["fitted_edge_count"]),
        "validation": ownership_validation,
        "topology_checks_performed": False,
    }

    edge_metrics = {
        "dense_curve_point_count": 0,
        "sparse_curve_point_count": 0,
        "max_chord_error_px": 0.0,
        "max_segment_arc_length_px": 0.0,
    }

    def write_edges(destination):
        def records():
            for artifact in edge_artifacts:
                with fiona.open(
                    artifact["path"],
                    layer="fitted_edges",
                ) as source:
                    for feature in source:
                        properties = dict(feature["properties"])
                        edge_metrics["dense_curve_point_count"] += int(
                            properties.get("dense_vtx") or 0
                        )
                        edge_metrics["sparse_curve_point_count"] += int(
                            properties.get("sparse_vtx") or 0
                        )
                        edge_metrics["max_chord_error_px"] = max(
                            edge_metrics["max_chord_error_px"],
                            float(properties.get("chord_err") or 0.0),
                        )
                        edge_metrics["max_segment_arc_length_px"] = max(
                            edge_metrics["max_segment_arc_length_px"],
                            float(properties.get("arc_len") or 0.0),
                        )
                        yield {
                            "geometry": feature["geometry"],
                            "properties": properties,
                        }

        destination.writerecords(records())

    candidate_path = root / "semantic_candidates.gpkg"
    staged_edges_path = root / f".fitted_edges.{os.getpid()}.stage.gpkg"
    staged_report_path = root / f".boundary_fitting_report.{os.getpid()}.stage.json"
    staged_candidate_path = root / f".semantic_candidates.{os.getpid()}.stage.gpkg"
    staged_paths = (
        staged_edges_path,
        staged_report_path,
        staged_candidate_path,
    )
    for path in staged_paths:
        path.unlink(missing_ok=True)

    def build_report_outputs() -> bool:
        try:
            print(
                json.dumps(
                    {
                        "event": "report_assembly_started",
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "assembly_mode": aggregate["assembly_mode"],
                        "total": expected_units,
                        "report_queue_capacity": aggregate[
                            "report_queue_capacity"
                        ],
                        "report_summary_source": "sqlite",
                        "report_json_parse_count": 0,
                        "summary_validation_workers": aggregate[
                            "summary_validation_workers"
                        ],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            _atomic_gpkg(
                staged_edges_path,
                "fitted_edges",
                edge_schema,
                spec["raster"]["crs"],
                write_edges,
            )
            aggregate.update(edge_metrics)
            dense_points = int(aggregate["dense_curve_point_count"])
            sparse_points = int(aggregate["sparse_curve_point_count"])
            aggregate["adaptive_point_reduction"] = (
                1.0 - sparse_points / dense_points
                if dense_points
                else 0.0
            )
            chord_limit = float(aggregate["max_chord_error_limit_px"])
            arc_limit = float(
                aggregate["max_segment_arc_length_limit_px"]
            )
            tolerance = 1e-9
            if aggregate["max_chord_error_px"] > chord_limit + tolerance:
                raise StreamAssemblyError(
                    "adaptive curve chord error exceeds configured limit: "
                    f"{aggregate['max_chord_error_px']} > {chord_limit}"
                )
            if (
                aggregate["max_segment_arc_length_px"]
                > arc_limit + tolerance
            ):
                raise StreamAssemblyError(
                    "adaptive curve arc length exceeds configured limit: "
                    f"{aggregate['max_segment_arc_length_px']} > {arc_limit}"
                )
            print(
                json.dumps(
                    {
                        "event": "report_assembly_completed",
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "assembly_mode": aggregate["assembly_mode"],
                        "current": aggregate["report_processed_count"],
                        "total": expected_units,
                        "report_processed_count": aggregate[
                            "report_processed_count"
                        ],
                        "report_queue_capacity": aggregate[
                            "report_queue_capacity"
                        ],
                        "report_peak_loaded_count": aggregate[
                            "report_peak_loaded_count"
                        ],
                        "report_summary_source": aggregate[
                            "report_summary_source"
                        ],
                        "report_json_parse_count": aggregate[
                            "report_json_parse_count"
                        ],
                        "summary_validation_peak_in_flight": aggregate[
                            "summary_validation_peak_in_flight"
                        ],
                        "failed_unit_count": aggregate["failed_unit_count"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            fitting_passed = aggregate["failed_unit_count"] == 0
            aggregate["status"] = "passed" if fitting_passed else "failed"
            aggregate["validation"]["passed"] = bool(
                aggregate["validation"].get("passed") and fitting_passed
            )
            if not fitting_passed:
                raise StreamAssemblyError("boundary fitting contains failed units")
            if resume_from_reports:
                aggregate["input_sha256"] = resume_inputs["raw"]["sha256"]
                aggregate["output_sha256"] = resume_inputs["formal"]["sha256"]
            else:
                aggregate["input_sha256"] = sha256_file(raw_path)
                aggregate["output_sha256"] = sha256_file(formal_path)
            accepted_value = str(spec.get("accepted_gpkg") or "")
            accepted_sha = str(spec.get("accepted_gpkg_sha256") or "")
            if accepted_value and accepted_sha:
                accepted_path = Path(accepted_value)
                if (
                    not accepted_path.is_file()
                    or sha256_file(accepted_path) != accepted_sha
                ):
                    raise StreamAssemblyError(
                        "accepted_labels changed after run creation"
                    )
            difference = apply_accepted_difference(
                formal_path,
                accepted_value,
                staged_candidate_path,
            )
            candidate_written = staged_candidate_path.is_file()
            if candidate_written and difference.get("output"):
                difference = dict(difference)
                difference["output"] = str(candidate_path)
            aggregate["difference"] = difference
            _atomic_json(staged_report_path, aggregate)
            if resume_from_reports:
                _assert_fingerprint_unchanged(raw_path, resume_inputs["raw"])
                _assert_fingerprint_unchanged(formal_path, resume_inputs["formal"])
            if candidate_written:
                os.replace(staged_candidate_path, candidate_path)
            os.replace(staged_edges_path, fitted_edges_path)
            os.replace(staged_report_path, report_path)
            return candidate_written
        finally:
            for path in staged_paths:
                path.unlink(missing_ok=True)

    candidate_written = build_report_outputs()
    for kind, path in (
        ("semantic_polygons_raw", raw_path),
        ("semantic_polygons", formal_path),
        ("boundary_fitting_report", report_path),
        ("fitted_edges", fitted_edges_path),
    ):
        _commit_artifact(
            database,
            run_id,
            path=path,
            kind=kind,
            stream_id=stream_id,
            unit_id="assembled",
        )
    if candidate_written:
        _commit_artifact(
            database,
            run_id,
            path=candidate_path,
            kind="semantic_candidates",
            stream_id=stream_id,
            unit_id="assembled",
        )
    database.set_stream_status(
        run_id,
        stream_id,
        "ready",
        error="",
    )
    print(json.dumps({"event": "stream_assembled", **aggregate}, separators=(",", ":")))
    return aggregate


def assemble_stream(
    run_spec_path: str | Path,
    stream_id: str,
    *,
    resume_from_reports: bool = False,
) -> dict[str, Any]:
    try:
        return _assemble_stream_impl(
            run_spec_path,
            stream_id,
            resume_from_reports=resume_from_reports,
        )
    except Exception as error:
        try:
            spec = load_json(Path(run_spec_path).resolve())
            if spec.get("schema_version") == 2:
                RunStateDB(spec["state_db"]).set_stream_status(
                    str(spec["run_id"]),
                    str(stream_id),
                    "failed",
                    error=str(error),
                )
        except Exception:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble one completed result stream")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument(
        "--resume-from-reports",
        action="store_true",
        help="reuse validated raw/formal outputs and continue from boundary reports",
    )
    args = parser.parse_args(argv)
    try:
        report = assemble_stream(
            args.run_spec,
            args.stream_id,
            resume_from_reports=args.resume_from_reports,
        )
        if report.get("status") != "passed":
            print(
                json.dumps(
                    {
                        "event": "stream_assembly_failed",
                        "assembly_mode": report.get("assembly_mode")
                        or (
                            "report_resume"
                            if args.resume_from_reports
                            else "full"
                        ),
                        "error": "boundary fitting contains failed units",
                    }
                )
            )
            return 2
        return 0
    except Exception as error:
        failure = {
            "event": "stream_assembly_failed",
            "assembly_mode": (
                "report_resume" if args.resume_from_reports else "full"
            ),
            "error": str(error),
        }
        if args.resume_from_reports:
            failure["safe_retry"] = "rerun_without_resume_from_reports"
        print(json.dumps(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
