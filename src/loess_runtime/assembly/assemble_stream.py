"""Stream unit outputs into final GPKGs and assign disk-backed object IDs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import fiona

import loess_runtime.assembly.stream_unit_cleanup as stream_unit_cleanup
from labeling_tool.shared.contracts.monitor_contract import (
    ASSEMBLY_PHASES as MONITOR_ASSEMBLY_PHASES,
)
from labeling_tool.shared.contracts.run_spec import load_json, sha256_file
from labeling_tool.shared.planning.ownership_neighbors import ownership_neighbors
from labeling_tool.shared.state.run_state_db import RunStateDB, run_state_from_spec
from loess_runtime.assembly.assembly_errors import StreamAssemblyError
from loess_runtime.assembly.difference_runtime import apply_accepted_difference
from loess_runtime.assembly.range_clip_runtime import apply_adaptive_range_clip
from loess_runtime.assembly.stream_assembly_inputs import (
    ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
    assert_fingerprint_unchanged,
    reuse_ready_assembly,
    validate_existing_gpkg,
    validated_summary_inputs,
)
from loess_runtime.assembly.stream_assembly_publication import (
    StagedReportOutputs,
)
from loess_runtime.assembly.stream_coverage_validation import (
    assert_gpkg_within_exact_range,
    publish_coverage_validation,
    validate_exact_range_coverage,
)
from loess_runtime.assembly.stream_vector_outputs import (
    FITTED_EDGE_SCHEMA,
    FORMAL_STREAM_SCHEMA,
    RAW_STREAM_SCHEMA,
    FormalOutputMetadata,
    estimate_source_gpkg_bytes,
    read_stream_vector_features,
    write_fitted_edges_output,
    write_formal_stream_output,
    write_raw_stream_output,
)
from loess_runtime.geometry.vector_data_plane import (
    read_boundary_signatures,
    read_geoparquet,
    signature_links,
)
from loess_runtime.inference.semantic_batch import write_atomic_json
from loess_runtime.system.artifact_publication import publish_artifact
from loess_runtime.system.concurrent_storage_reservation import (
    concurrent_storage_reservation,
)
from loess_runtime.system.storage_guard import (
    StorageGuard,
    create_run_storage_guard,
)

ASSEMBLY_PHASES = tuple((phase, name) for phase, name, _unit in MONITOR_ASSEMBLY_PHASES)
ASSEMBLY_PHASE_INDEX = {
    phase: index for index, (phase, _name) in enumerate(ASSEMBLY_PHASES, start=1)
}
ASSEMBLY_PHASE_NAMES = dict(ASSEMBLY_PHASES)
ASSEMBLY_PROGRESS_INTERVAL_SEC = 0.75


class _AssemblyProgress:
    """Emit and persist throttled, restart-visible Stream assembly progress."""

    def __init__(self, database: RunStateDB, run_id: str, stream_id: str):
        self.database = database
        self.run_id = str(run_id)
        self.stream_id = str(stream_id)
        self.started_at = time.monotonic()
        self._last_emit_at = 0.0
        self._last_phase = ""

    def emit(
        self,
        phase: str,
        *,
        current: int = 0,
        total: int = 0,
        feature_count: int = 0,
        status: str = "running",
        message: str = "",
        force: bool = False,
    ) -> None:
        phase_value = str(phase)
        if phase_value not in ASSEMBLY_PHASE_INDEX:
            raise StreamAssemblyError(f"unknown assembly progress phase: {phase_value}")
        now = time.monotonic()
        total_value = max(0, int(total))
        current_value = max(0, int(current))
        if total_value:
            current_value = min(current_value, total_value)
        should_emit = (
            force
            or phase_value != self._last_phase
            or str(status) != "running"
            or (total_value > 0 and current_value >= total_value)
            or now - self._last_emit_at >= ASSEMBLY_PROGRESS_INTERVAL_SEC
        )
        if not should_emit:
            return
        event = {
            "event": "assembly_progress",
            "run_id": self.run_id,
            "stream_id": self.stream_id,
            "stage": "assembly",
            "phase": phase_value,
            "phase_name": ASSEMBLY_PHASE_NAMES[phase_value],
            "phase_index": ASSEMBLY_PHASE_INDEX[phase_value],
            "phase_total": len(ASSEMBLY_PHASES),
            "current": current_value,
            "total": total_value,
            "feature_count": max(0, int(feature_count)),
            "status": str(status),
            "message": str(message),
            "elapsed_sec": round(now - self.started_at, 3),
        }
        try:
            self.database.run_streams.upsert_stream_runtime_progress(
                self.run_id,
                self.stream_id,
                stage="assembly",
                phase=phase_value,
                phase_name=event["phase_name"],
                phase_index=event["phase_index"],
                phase_total=event["phase_total"],
                current=current_value,
                total=total_value,
                feature_count=event["feature_count"],
                status=event["status"],
                message=event["message"],
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "assembly_progress_persistence_warning",
                        "run_id": self.run_id,
                        "stream_id": self.stream_id,
                        "phase": phase_value,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        self._last_phase = phase_value
        self._last_emit_at = now


JSON_ATOMIC_OVERHEAD_BYTES = 64 * 1024


def _estimate_json_bytes(payload: Mapping[str, Any]) -> int:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return len(encoded) + JSON_ATOMIC_OVERHEAD_BYTES


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    operation: str = "stream_report",
) -> None:
    with concurrent_storage_reservation(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_json_bytes(payload),
    ):
        write_atomic_json(path, payload)


def _accepted_layer_has_geometry(
    accepted_path: str | Path,
    *,
    accepted_layer: str = "accepted_labels",
) -> bool:
    path = Path(accepted_path).resolve()
    if not path.is_file() or accepted_layer not in fiona.listlayers(path):
        return False
    with fiona.open(path, layer=accepted_layer) as source:
        return any(feature.get("geometry") for feature in source)


def _guarded_accepted_difference(
    source_path: Path,
    accepted_path: str | Path,
    output_path: Path,
    *,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    operation: str,
) -> dict[str, Any]:
    if not _accepted_layer_has_geometry(accepted_path):
        return apply_accepted_difference(source_path, accepted_path, output_path)
    with concurrent_storage_reservation(
        storage_guard,
        storage_lock_path,
        operation,
        estimate_source_gpkg_bytes((source_path,), multiplier=2),
    ):
        return apply_accepted_difference(source_path, accepted_path, output_path)


def _guarded_range_clip(
    source_path: Path,
    spec: Mapping[str, Any],
    *,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    operation: str,
) -> dict[str, Any]:
    with concurrent_storage_reservation(
        storage_guard,
        storage_lock_path,
        operation,
        estimate_source_gpkg_bytes((source_path,), multiplier=2),
    ):
        return apply_adaptive_range_clip(source_path, spec)


def _stream_root(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    run_dir = Path(spec["run_dir"])
    if stream["kind"] == "model":
        return run_dir / "models" / str(stream["model_id"])
    return run_dir / "fusion" / str(stream["profile_id"])


def _signature_object_ids(
    signature_by_unit: Mapping[str, str],
    formal_by_unit: Mapping[str, str],
    units: list[Mapping[str, Any]],
    run_id: str,
    stream_id: str,
    tolerance: float,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[dict[str, str], int]:
    """Resolve the existing deterministic object IDs from signature links."""

    parents: dict[str, str] = {}
    ranks: dict[str, int] = {}
    for path in formal_by_unit.values():
        for feature in read_stream_vector_features(path):
            part_id = str(feature["properties"]["part_id"])
            parents[part_id] = part_id
            ranks[part_id] = 0

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        left_rank, right_rank = ranks[left_root], ranks[right_root]
        if left_rank < right_rank or (
            left_rank == right_rank and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        parents[right_root] = left_root
        if left_rank == right_rank:
            ranks[left_root] = left_rank + 1

    links: set[tuple[str, str, int]] = set()
    neighbors = ownership_neighbors(units)
    for pair_index, (left_unit, right_unit) in enumerate(neighbors, start=1):
        left = read_boundary_signatures(
            signature_by_unit[left_unit], stream_id=str(stream_id), unit_id=left_unit
        )
        right = read_boundary_signatures(
            signature_by_unit[right_unit], stream_id=str(stream_id), unit_id=right_unit
        )
        for link in signature_links(left, right, tolerance=tolerance):
            links.add(link)
        if progress_callback is not None:
            progress_callback(pair_index, len(neighbors), len(links))
    for left_part_id, right_part_id, _class_code in sorted(links):
        union(left_part_id, right_part_id)
    object_ids = {
        part_id: "obj_"
        + hashlib.sha1(
            f"{run_id}|{stream_id}|{find(part_id)}".encode("utf-8")
        ).hexdigest()[:24]
        for part_id in sorted(parents)
    }
    return object_ids, len(links)


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
    database = run_state_from_spec(spec)
    streams = [item for item in spec["streams"] if item["stream_id"] == stream_id]
    if len(streams) != 1:
        raise StreamAssemblyError(f"unknown result stream: {stream_id}")
    stream = streams[0]
    boundary = spec.get("boundary_fitting") or {}
    fit_mode = str(boundary.get("mode") or "")
    if fit_mode != "divider_cubic_bspline_adaptive_v2":
        raise StreamAssemblyError(
            "only divider_cubic_bspline_adaptive_v2 is supported by the current runtime"
        )
    smoothing_enabled = bool(boundary.get("enabled", True))
    fit_version = (
        "divider_cubic_bspline_adaptive_v2"
        if smoothing_enabled
        else "raw_polygonize_v1"
    )
    counts = database.control_graph.stream_unit_counts(run_id, stream_id)
    expected_units = sum(counts.values())
    progress = _AssemblyProgress(database, run_id, stream_id)
    stream_rows = {
        str(row["stream_id"]): row for row in database.run_streams.stream_rows(run_id)
    }
    reused = reuse_ready_assembly(
        spec=spec,
        run_id=run_id,
        stream_id=stream_id,
        stream_status=str((stream_rows.get(stream_id) or {}).get("status") or ""),
        root=_stream_root(spec, stream),
        artifacts=database.artifacts,
        append_event=database.run_streams.append_event,
    )
    if reused is not None:
        progress.emit(
            "publish_cleanup",
            current=0,
            total=1,
            message="校验已组装产物并清理残留中间文件",
            force=True,
        )
        reused["unit_artifact_cleanup"] = (
            stream_unit_cleanup.cleanup_stream_unit_artifacts(
                run_id=run_id,
                run_dir=Path(spec["run_dir"]),
                stream_id=stream_id,
                artifacts=database.artifacts,
                append_event=database.run_streams.append_event,
            )
        )
        with database.owner_publication(run_id, Path(spec["run_dir"])) as publication:
            if not publication.run_streams.set_stream_status(
                run_id, stream_id, "ready", error=""
            ):
                raise StreamAssemblyError(
                    f"cannot mark reused Stream ready: {stream_id}"
                )
        progress.emit(
            "publish_cleanup",
            current=1,
            total=1,
            status="reused",
            message="已复用完整组装产物",
            force=True,
        )
        return reused
    database.run_streams.set_stream_status(
        run_id,
        stream_id,
        "assembling",
        error="",
    )
    if expected_units < 1 or counts != {"ready": expected_units}:
        raise StreamAssemblyError(f"stream units are not all ready: {counts}")
    units = database.control_graph.spatial_units_for_stream(run_id, stream_id)
    formal_artifacts = database.artifacts.artifacts_for_stream(
        run_id, stream_id, kind="unit_formal_geoparquet"
    )
    raw_artifacts = database.artifacts.artifacts_for_stream(
        run_id, stream_id, kind="unit_raw_geoparquet"
    )
    signature_artifacts = database.artifacts.artifacts_for_stream(
        run_id, stream_id, kind="unit_boundary_signatures"
    )
    report_artifacts = database.artifacts.artifacts_for_stream(
        run_id, stream_id, kind="unit_boundary_report"
    )
    if not (
        len(formal_artifacts)
        == len(raw_artifacts)
        == len(report_artifacts)
        == len(signature_artifacts)
        == expected_units
    ):
        raise StreamAssemblyError("unit Artifact count does not match ready unit count")
    formal_by_unit = {
        str(item["unit_id"]): str(item["path"]) for item in formal_artifacts
    }
    raw_by_unit = {str(item["unit_id"]): str(item["path"]) for item in raw_artifacts}
    signature_by_unit = {
        str(item["unit_id"]): str(item["path"]) for item in signature_artifacts
    }

    root = _stream_root(spec, stream)
    storage_guard = create_run_storage_guard(spec, database)
    storage_lock_path = Path(spec["run_dir"]) / "tmp" / ".vector-storage-reserve.lock"
    canonical_raw_path = root / "semantic_polygons_raw.gpkg"
    canonical_formal_path = root / "semantic_polygons.gpkg"
    report_path = root / "boundary_fitting_report.json"
    fitted_edges_path = root / "fitted_edges.gpkg"
    candidate_path = root / "semantic_candidates.gpkg"
    staged_outputs = StagedReportOutputs.for_stream(root)
    staged_outputs.discard()
    raw_path = canonical_raw_path if resume_from_reports else staged_outputs.raw
    formal_path = canonical_formal_path if resume_from_reports else staged_outputs.formal
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    raw_schema = RAW_STREAM_SCHEMA
    formal_schema = FORMAL_STREAM_SCHEMA
    edge_schema = FITTED_EDGE_SCHEMA
    progress.emit(
        "validate_inputs",
        current=0,
        total=expected_units,
        message="校验单元报告和拟合边界分片",
        force=True,
    )
    edge_artifacts, summary_validation = validated_summary_inputs(
        spec,
        run_id=run_id,
        stream_id=stream_id,
        expected_units=expected_units,
        report_artifacts=report_artifacts,
        report_summaries=database.unit_reports.unit_report_summaries(run_id, stream_id),
        artifacts=database.artifacts,
        edge_schema=edge_schema,
        progress_callback=lambda current, total: progress.emit(
            "validate_inputs",
            current=current,
            total=total,
            message="校验单元报告和拟合边界分片",
        ),
    )
    summary_aggregate = database.unit_reports.unit_report_summary_aggregate(
        run_id,
        stream_id,
    )
    if int(summary_aggregate["unit_count"]) != expected_units:
        raise StreamAssemblyError(
            "run-state report aggregate does not cover every ready unit"
        )
    progress.emit(
        "validate_inputs",
        current=int(summary_validation["artifact_count"]),
        total=int(summary_validation["artifact_count"]),
        status="completed",
        message="单元产物校验完成",
        force=True,
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
    formal_feature_count = 0

    if resume_from_reports:
        progress.emit(
            "register_objects",
            current=0,
            total=1,
            message="校验已有对象身份",
            force=True,
        )
        object_ids, link_count = _signature_object_ids(
            signature_by_unit, formal_by_unit, units, run_id, stream_id, 1e-9
        )
        part_count = len(object_ids)
        object_count = len(set(object_ids.values()))
        resume_inputs["raw"] = validate_existing_gpkg(
            raw_path,
            layer="semantic_polygons_raw",
            schema=raw_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "stream_id": stream_id},
            expected_feature_count=part_count,
        )
        resume_inputs["formal"] = validate_existing_gpkg(
            formal_path,
            layer="semantic_polygons",
            schema=formal_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "result_stream_id": stream_id},
            # Exact clipping may discard a fully outside part or split one at
            # the boundary, so formal feature count is intentionally not tied
            # to the pre-clip columnar parts. Raw remains count-locked above.
            expected_feature_count=None,
        )
        assert_gpkg_within_exact_range(
            formal_path,
            layer="semantic_polygons",
            spec=spec,
        )
        formal_feature_count = int(resume_inputs["formal"].get("feature_count") or 0)
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
        for phase, message in (
            ("register_objects", "已有对象身份校验完成"),
            ("link_objects", "复用已有跨单元对象连接"),
            ("write_raw", "复用已有 Raw GPKG"),
            ("write_formal", "复用并校验已有正式 GPKG"),
        ):
            progress.emit(
                phase,
                current=1,
                total=1,
                status="reused",
                message=message,
                force=True,
            )
    else:
        registered_part_count = sum(
            int(read_geoparquet(artifact["path"])[0]["feature_count"])
            for artifact in formal_artifacts
        )
        progress.emit(
            "register_objects",
            current=0,
            total=len(formal_artifacts),
            message="载入列式多边形部件身份",
            force=True,
        )
        for artifact_index, artifact in enumerate(formal_artifacts, start=1):
            progress.emit(
                "register_objects",
                current=artifact_index,
                total=len(formal_artifacts),
                feature_count=registered_part_count,
                message=f"已读取 {registered_part_count} 个列式多边形部件",
            )
        progress.emit(
            "register_objects",
            current=len(formal_artifacts),
            total=len(formal_artifacts),
            feature_count=registered_part_count,
            status="completed",
            message="列式多边形部件身份读取完成",
            force=True,
        )
        pixel_tolerance = 1e-9
        progress.emit(
            "link_objects",
            current=0,
            total=0,
            feature_count=registered_part_count,
            message="扫描相邻空间单元公共边界",
            force=True,
        )
        object_ids, link_count = _signature_object_ids(
            signature_by_unit,
            formal_by_unit,
            units,
            run_id,
            stream_id,
            pixel_tolerance,
            progress_callback=lambda current, total, linked: progress.emit(
                "link_objects",
                current=current,
                total=total,
                feature_count=registered_part_count,
                message=f"已建立 {linked} 条跨单元对象连接",
            ),
        )
        progress.emit(
            "link_objects",
            current=0,
            total=0,
            feature_count=registered_part_count,
            message=f"解析对象连接，当前连接 {link_count} 条",
            force=True,
        )
        object_count = len(set(object_ids.values()))
        progress.emit(
            "link_objects",
            current=1,
            total=1,
            feature_count=registered_part_count,
            message=f"对象连接完成，共 {object_count} 个对象",
            status="completed",
            force=True,
        )
        class_snapshot = load_json(Path(spec["class_mapping_snapshot"]))
        class_names = class_snapshot["class_mapping"]

        progress.emit(
            "write_raw",
            current=0,
            total=len(units),
            feature_count=0,
            message="创建 Raw GPKG",
            force=True,
        )
        raw_feature_count = write_raw_stream_output(
            raw_by_unit=raw_by_unit,
            units=units,
            output_path=raw_path,
            crs=spec["raster"]["crs"],
            run_id=run_id,
            stream_id=stream_id,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            progress=lambda current, total, feature_count: progress.emit(
                "write_raw",
                current=current,
                total=total,
                feature_count=feature_count,
                message=f"已写入 {feature_count} 个 Raw 面",
            ),
        )
        progress.emit(
            "write_raw",
            current=len(units),
            total=len(units),
            feature_count=raw_feature_count,
            status="completed",
            message="Raw GPKG 已成功写入并提交",
            force=True,
        )

        progress.emit(
            "write_formal",
            current=0,
            total=len(units),
            feature_count=0,
            message="创建正式 GPKG",
            force=True,
        )
        formal_feature_count = write_formal_stream_output(
            formal_by_unit=formal_by_unit,
            units=units,
            output_path=formal_path,
            crs=spec["raster"]["crs"],
            metadata=FormalOutputMetadata(
                run_id=run_id,
                stream_id=stream_id,
                result_kind=str(stream["kind"]),
                model_id=model_id,
                fusion_profile_id=profile_id,
                model_version=version,
                class_names=class_names,
                fit_version=fit_version,
                created_at=now,
            ),
            object_ids=object_ids,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            progress=lambda current, total, feature_count: progress.emit(
                "write_formal",
                current=current,
                total=total,
                feature_count=feature_count,
                message=f"已写入 {feature_count} 个正式面",
            ),
        )
        progress.emit(
            "write_formal",
            current=len(units),
            total=len(units),
            feature_count=formal_feature_count,
            status="completed",
            message="正式 GPKG 已成功写入并提交",
            force=True,
        )

    aggregate = {
        "schema_version": 1,
        "run_id": run_id,
        "stream_id": stream_id,
        "assembly_mode": "report_resume" if resume_from_reports else "full",
        "report_queue_capacity": ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
        "report_summary_source": "run_state_database",
        "report_processed_count": expected_units,
        "report_peak_loaded_count": 0,
        "report_json_parse_count": 0,
        "summary_validation_workers": summary_validation["workers"],
        "summary_validation_peak_in_flight": summary_validation["peak_in_flight"],
        "summary_validation_artifact_count": summary_validation["artifact_count"],
        "summary_validation_elapsed_sec": summary_validation["elapsed_sec"],
        "gpkg_write_mode": "pyogrio_arrow_single_publish",
        "object_id_resolution": "boundary_signature_components_v1",
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
        "max_chord_error_limit_px": float(boundary.get("max_chord_error_px", 0.25)),
        "max_segment_arc_length_limit_px": float(
            boundary.get("max_segment_arc_length_px", 8.0)
        ),
        "chain_count": int(summary_aggregate["chain_count"]),
        "shared_chain_count": int(summary_aggregate["shared_chain_count"]),
        "spline_count": int(summary_aggregate["spline_count"]),
        "unchanged_count": int(summary_aggregate["unchanged_count"]),
        "skipped_invalid_count": int(summary_aggregate["skipped_invalid_count"]),
        "failed_unit_count": int(summary_aggregate["failed_unit_count"]),
        "max_displacement_px": float(summary_aggregate["max_displacement_px"]),
        "diagnostic_count": int(summary_aggregate["diagnostic_count"]),
        "fitted_edge_count": int(summary_aggregate["fitted_edge_count"]),
        "validation": ownership_validation,
        "topology_checks_performed": False,
    }

    staged_edges_path = staged_outputs.fitted_edges
    staged_report_path = staged_outputs.report
    staged_candidate_path = staged_outputs.candidate

    def build_report_outputs() -> bool:
        nonlocal formal_feature_count
        try:
            progress.emit(
                "aggregate_reports",
                current=0,
                total=max(1, len(edge_artifacts)),
                feature_count=0,
                message="汇总拟合报告与公共边界",
                force=True,
            )
            print(
                json.dumps(
                    {
                        "event": "report_assembly_started",
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "assembly_mode": aggregate["assembly_mode"],
                        "total": expected_units,
                        "report_queue_capacity": aggregate["report_queue_capacity"],
                        "report_summary_source": "run_state_database",
                        "report_json_parse_count": 0,
                        "summary_validation_workers": aggregate[
                            "summary_validation_workers"
                        ],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            edge_output = write_fitted_edges_output(
                edge_artifacts=edge_artifacts,
                output_path=staged_edges_path,
                crs=spec["raster"]["crs"],
                stream_id=stream_id,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                progress=lambda current, total, feature_count: progress.emit(
                    "aggregate_reports",
                    current=current,
                    total=total,
                    feature_count=feature_count,
                    message=f"已汇总 {feature_count} 条拟合边界",
                ),
            )
            edge_feature_count = edge_output.feature_count
            progress.emit(
                "aggregate_reports",
                current=max(1, len(edge_artifacts)),
                total=max(1, len(edge_artifacts)),
                feature_count=edge_feature_count,
                message="拟合报告与公共边界汇总完成",
                force=True,
            )
            aggregate.update(
                {
                    "dense_curve_point_count": edge_output.dense_curve_point_count,
                    "sparse_curve_point_count": edge_output.sparse_curve_point_count,
                    "max_chord_error_px": edge_output.max_chord_error_px,
                    "max_segment_arc_length_px": edge_output.max_segment_arc_length_px,
                }
            )
            dense_points = int(aggregate["dense_curve_point_count"])
            sparse_points = int(aggregate["sparse_curve_point_count"])
            aggregate["adaptive_point_reduction"] = (
                1.0 - sparse_points / dense_points if dense_points else 0.0
            )
            chord_limit = float(aggregate["max_chord_error_limit_px"])
            arc_limit = float(aggregate["max_segment_arc_length_limit_px"])
            tolerance = 1e-9
            if aggregate["max_chord_error_px"] > chord_limit + tolerance:
                raise StreamAssemblyError(
                    "adaptive curve chord error exceeds configured limit: "
                    f"{aggregate['max_chord_error_px']} > {chord_limit}"
                )
            if aggregate["max_segment_arc_length_px"] > arc_limit + tolerance:
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
                        "report_processed_count": aggregate["report_processed_count"],
                        "report_queue_capacity": aggregate["report_queue_capacity"],
                        "report_peak_loaded_count": aggregate[
                            "report_peak_loaded_count"
                        ],
                        "report_summary_source": aggregate["report_summary_source"],
                        "report_json_parse_count": aggregate["report_json_parse_count"],
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
            progress.emit(
                "aggregate_reports",
                current=max(1, len(edge_artifacts)),
                total=max(1, len(edge_artifacts)),
                feature_count=edge_feature_count,
                status="completed",
                message="拟合报告与公共边界汇总校验完成",
                force=True,
            )
            progress.emit(
                "range_clip",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message=(
                    "校验已裁剪正式 GPKG"
                    if resume_from_reports
                    else "按冻结研究范围精确裁剪正式 GPKG"
                ),
                force=True,
            )
            if resume_from_reports:
                range_clip = {
                    "status": "already_clipped",
                    "reason": "resume formal output was range-validated before assembly",
                }
            else:
                range_clip = _guarded_range_clip(
                    formal_path,
                    spec,
                    storage_guard=storage_guard,
                    storage_lock_path=storage_lock_path,
                    operation=f"stream_range_clip:{stream_id}",
                )
            aggregate["range_clip"] = range_clip
            if "output_feature_count" in range_clip:
                formal_feature_count = int(range_clip["output_feature_count"])
            elif "source_feature_count" in range_clip:
                formal_feature_count = int(range_clip["source_feature_count"])
            progress.emit(
                "range_clip",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message="研究范围裁剪完成",
                status="reused" if resume_from_reports else "completed",
                force=True,
            )
            progress.emit(
                "coverage_validation",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message="核验研究范围内空白、重叠和范围外面积",
                force=True,
            )
            coverage = validate_exact_range_coverage(
                formal_path,
                layer="semantic_polygons",
                spec=spec,
            )
            aggregate["coverage_validation"] = coverage
            publish_coverage_validation(
                database.run_streams.append_event,
                run_id,
                stream_id,
                coverage,
            )
            if coverage["status"] == "failed":
                progress.emit(
                    "coverage_validation",
                    current=1,
                    total=1,
                    feature_count=formal_feature_count,
                    status="failed",
                    message=(
                        f"空白 {coverage['gap_area_m2']:.6g} m²；"
                        f"重叠 {coverage['overlap_area_m2']:.6g} m²；"
                        f"范围外 {coverage['outside_area_m2']:.6g} m²"
                    ),
                    force=True,
                )
                raise StreamAssemblyError(
                    "exact range coverage validation failed: "
                    f"gap={coverage['gap_area_m2']:.6g} m2, "
                    f"overlap={coverage['overlap_area_m2']:.6g} m2, "
                    f"outside={coverage['outside_area_m2']:.6g} m2"
                )
            if coverage["status"] == "passed":
                coverage_message = "空白 0；重叠 0；范围外 0"
            else:
                coverage_message = "历史 Run 缺少冻结范围，覆盖验收未执行"
            progress.emit(
                "coverage_validation",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message=coverage_message,
                status="completed" if coverage["status"] == "passed" else "skipped",
                force=True,
            )
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
            progress.emit(
                "accepted_difference",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message="计算 Accepted 标签差分",
                force=True,
            )
            difference = _guarded_accepted_difference(
                formal_path,
                accepted_value,
                staged_candidate_path,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                operation=f"stream_candidate_stage:{stream_id}",
            )
            candidate_written = staged_candidate_path.is_file()
            if candidate_written and difference.get("output"):
                difference = dict(difference)
                difference["output"] = str(candidate_path)
            aggregate["difference"] = difference
            progress.emit(
                "accepted_difference",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message="Accepted 标签差分完成",
                status="skipped"
                if difference.get("status") == "skipped"
                else "completed",
                force=True,
            )
            _write_json(
                staged_report_path,
                aggregate,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                operation=f"stream_report_stage:{stream_id}",
            )
            if resume_from_reports:
                assert_fingerprint_unchanged(raw_path, resume_inputs["raw"])
                assert_fingerprint_unchanged(formal_path, resume_inputs["formal"])
            return candidate_written
        except Exception:
            staged_outputs.discard()
            raise

    candidate_written = build_report_outputs()
    assembled_artifacts = [
        ("semantic_polygons_raw", canonical_raw_path),
        ("semantic_polygons", canonical_formal_path),
        ("boundary_fitting_report", report_path),
        ("fitted_edges", fitted_edges_path),
    ]
    if candidate_written:
        assembled_artifacts.append(("semantic_candidates", candidate_path))
    publish_total = len(assembled_artifacts) + 1
    progress.emit(
        "publish_cleanup",
        current=0,
        total=publish_total,
        feature_count=formal_feature_count,
        message="提交正式组装产物",
        force=True,
    )
    try:
        with database.owner_publication(
            run_id, Path(spec["run_dir"])
        ) as publication:
            staged_outputs.publish(
                raw=canonical_raw_path,
                formal=canonical_formal_path,
                fitted_edges=fitted_edges_path,
                report=report_path,
                candidate=candidate_path,
                publish_core_outputs=not resume_from_reports,
                candidate_written=candidate_written,
            )
            for kind, path in assembled_artifacts:
                publish_artifact(
                    publication.artifacts,
                    run_id,
                    path=path,
                    kind=kind,
                    stream_id=stream_id,
                    unit_id="assembled",
                )
            if not publication.run_streams.set_stream_status(
                run_id,
                stream_id,
                "ready",
                error="",
            ):
                raise StreamAssemblyError(f"cannot mark Stream ready: {stream_id}")
    finally:
        staged_outputs.discard()
    progress.emit(
        "publish_cleanup",
        current=len(assembled_artifacts),
        total=publish_total,
        feature_count=formal_feature_count,
        message=f"已提交 {len(assembled_artifacts)} 个正式产物",
    )
    aggregate["unit_artifact_cleanup"] = (
        stream_unit_cleanup.cleanup_stream_unit_artifacts(
            run_id=run_id,
            run_dir=Path(spec["run_dir"]),
            stream_id=stream_id,
            artifacts=database.artifacts,
            append_event=database.run_streams.append_event,
        )
    )
    progress.emit(
        "publish_cleanup",
        current=publish_total,
        total=publish_total,
        feature_count=formal_feature_count,
        status="completed",
        message="正式产物已提交，中间文件清理完成",
        force=True,
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
                database = run_state_from_spec(spec)
                database.run_streams.set_stream_status(
                    str(spec["run_id"]),
                    str(stream_id),
                    "failed",
                    error=str(error),
                )
                database.run_streams.fail_stream_runtime_progress(
                    str(spec["run_id"]),
                    str(stream_id),
                    str(error),
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
                        or ("report_resume" if args.resume_from_reports else "full"),
                        "error": "boundary fitting contains failed units",
                    }
                )
            )
            return 2
        return 0
    except Exception as error:
        failure = {
            "event": "stream_assembly_failed",
            "assembly_mode": ("report_resume" if args.resume_from_reports else "full"),
            "error": str(error),
        }
        if args.resume_from_reports:
            failure["safe_retry"] = "rerun_without_resume_from_reports"
        print(json.dumps(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
