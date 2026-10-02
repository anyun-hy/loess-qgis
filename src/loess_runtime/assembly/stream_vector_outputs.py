"""Final Raw, Formal, and fitted-edge GeoPackage output ownership.

The final writer retains the established single Arrow-table publish behavior:
it materialises rows for one Stream and invokes ``pyogrio.write_arrow`` once.
It does not claim bounded-memory streaming.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from shapely.geometry import (  # type: ignore[import-untyped]
    MultiPolygon,
    mapping,
    shape,
)
from shapely.wkb import loads as load_wkb  # type: ignore[import-untyped]

from loess_runtime.assembly.assembly_errors import StreamAssemblyError
from loess_runtime.geometry.vector_data_plane import read_geoparquet
from loess_runtime.system.concurrent_storage_reservation import (
    concurrent_storage_reservation,
)
from loess_runtime.system.storage_guard import StorageGuard

GPKG_ATOMIC_OVERHEAD_BYTES = 4 * 1024**2

RAW_STREAM_SCHEMA: dict[str, Any] = {
    "geometry": "MultiPolygon",
    "properties": {
        "run_id": "str:48",
        "stream_id": "str:96",
        "unit_id": "str:96",
        "polygon_id": "str:96",
        "class_code": "int",
    },
}

FORMAL_STREAM_SCHEMA: dict[str, Any] = {
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

FITTED_EDGE_SCHEMA: dict[str, Any] = {
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

ProgressSink = Callable[[int, int, int], None]


class _RecordDestination(Protocol):
    def writerecords(self, records: Iterable[Mapping[str, Any]]) -> None: ...


_VectorWriter = Callable[[_RecordDestination], None]


@dataclass(frozen=True)
class FormalOutputMetadata:
    """Frozen Stream metadata written to every formal polygon."""

    run_id: str
    stream_id: str
    result_kind: str
    model_id: str
    fusion_profile_id: str
    model_version: str
    class_names: Mapping[str, str]
    fit_version: str
    created_at: str


@dataclass(frozen=True)
class EdgeOutputMetrics:
    """Counts and maxima observed while assembling fitted-edge shards."""

    feature_count: int
    dense_curve_point_count: int
    sparse_curve_point_count: int
    max_chord_error_px: float
    max_segment_arc_length_px: float


def estimate_source_gpkg_bytes(
    paths: Sequence[str | Path], *, multiplier: int = 2
) -> int:
    """Reserve enough space for an atomic GPKG generated from source shards."""

    source_bytes = sum(Path(path).stat().st_size for path in paths)
    return max(
        GPKG_ATOMIC_OVERHEAD_BYTES,
        max(1, int(multiplier)) * source_bytes + GPKG_ATOMIC_OVERHEAD_BYTES,
    )


def read_stream_vector_features(path: str | Path) -> list[dict[str, Any]]:
    """Materialise one columnar shard for final output or object-ID assembly."""

    _manifest, table = read_geoparquet(path)
    result = []
    for row in table.to_pylist():
        properties = dict(row)
        geometry = load_wkb(bytes(properties.pop("geometry")))
        properties.pop("source_sha256", None)
        result.append({"geometry": geometry, "properties": properties})
    return result


def _stream_vector_feature_batches(
    path: str | Path, *, size: int = 512
) -> Iterator[list[dict[str, Any]]]:
    """Convert formal shards in the established 512-record geometry batches."""

    batch = []
    _manifest, table = read_geoparquet(path)
    for row in table.to_pylist():
        properties = dict(row)
        geometry = load_wkb(bytes(properties.pop("geometry")))
        properties.pop("source_sha256", None)
        batch.append({"geometry": mapping(geometry), "properties": properties})
        if len(batch) >= int(size):
            yield batch
            batch = []
    if batch:
        yield batch


def _write_atomic_gpkg(
    path: Path,
    layer: str,
    schema: Mapping[str, Any],
    crs: str,
    writer: _VectorWriter,
    *,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    estimated_write_bytes: int = GPKG_ATOMIC_OVERHEAD_BYTES,
    operation: str = "stream_gpkg",
) -> None:
    """Write a complete GPKG to a temporary file, integrity-check, then replace.

    ``writer`` receives a Fiona-shaped destination whose rows are accumulated
    before one Arrow publish.  This preserves the pre-existing output order.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.gpkg"
    with concurrent_storage_reservation(
        storage_guard, storage_lock_path, operation, estimated_write_bytes
    ):
        temporary.unlink(missing_ok=True)
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
            import pyogrio  # type: ignore[import-untyped]

            class _ArrowDestination:
                def __init__(self) -> None:
                    self.rows: list[dict[str, Any]] = []

                def writerecords(self, records: Iterable[Mapping[str, Any]]) -> None:
                    for record in records:
                        values = dict(record["properties"])
                        geometry = record.get("geometry")
                        if geometry is None:
                            raise StreamAssemblyError(
                                "final GPKG record has no geometry"
                            )
                        normalized = shape(geometry)
                        if (
                            str(schema.get("geometry")) == "MultiPolygon"
                            and normalized.geom_type == "Polygon"
                        ):
                            normalized = MultiPolygon([normalized])
                        if normalized.geom_type != str(schema.get("geometry")):
                            raise StreamAssemblyError(
                                "final GPKG geometry differs from its declared schema: "
                                f"expected={schema.get('geometry')}, "
                                f"actual={normalized.geom_type}"
                            )
                        values["geometry"] = bytes(normalized.wkb)
                        self.rows.append(values)

            destination = _ArrowDestination()
            writer(destination)
            fields = schema.get("properties") or {}
            columns = {}
            for name, declared in fields.items():
                values = [row.get(name) for row in destination.rows]
                kind = str(declared).split(":", 1)[0].lower()
                if kind == "int":
                    columns[name] = pa.array(values, type=pa.int64())
                elif kind == "float":
                    columns[name] = pa.array(values, type=pa.float64())
                else:
                    columns[name] = pa.array(values, type=pa.string())
            columns["geometry"] = pa.array(
                [row["geometry"] for row in destination.rows], type=pa.binary()
            )
            table = pa.table(columns)
            metadata = dict(table.schema.metadata or {})
            metadata[b"geo"] = json.dumps(
                {
                    "version": "1.1.0",
                    "primary_column": "geometry",
                    "columns": {"geometry": {"encoding": "WKB", "crs": str(crs)}},
                }
            ).encode("utf-8")
            pyogrio.write_arrow(
                table.replace_schema_metadata(metadata),
                temporary,
                layer=layer,
                driver="GPKG",
                geometry_name="geometry",
                geometry_type=str(schema["geometry"]),
                crs=str(crs),
            )
            with sqlite3.connect(temporary) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise StreamAssemblyError(
                        f"GeoPackage integrity check failed: {temporary}"
                    )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_raw_stream_output(
    *,
    raw_by_unit: Mapping[str, str],
    units: Sequence[Mapping[str, Any]],
    output_path: Path,
    crs: str,
    run_id: str,
    stream_id: str,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    progress: ProgressSink | None = None,
) -> int:
    """Write Raw unit shards in unit order and return their feature count."""

    feature_count = 0

    def writer(destination: _RecordDestination) -> None:
        def records() -> Iterator[Mapping[str, Any]]:
            nonlocal feature_count
            for unit_index, unit in enumerate(units, start=1):
                unit_id = str(unit["unit_id"])
                for feature in read_stream_vector_features(raw_by_unit[unit_id]):
                    feature_count += 1
                    yield {
                        "geometry": mapping(feature["geometry"]),
                        "properties": {
                            "run_id": run_id,
                            "stream_id": stream_id,
                            "unit_id": unit_id,
                            "polygon_id": str(feature["properties"]["part_id"]),
                            "class_code": int(feature["properties"]["class_code"]),
                        },
                    }
                if progress is not None:
                    progress(unit_index, len(units), feature_count)

        destination.writerecords(records())

    _write_atomic_gpkg(
        output_path,
        "semantic_polygons_raw",
        RAW_STREAM_SCHEMA,
        crs,
        writer,
        storage_guard=storage_guard,
        storage_lock_path=storage_lock_path,
        estimated_write_bytes=estimate_source_gpkg_bytes(tuple(raw_by_unit.values())),
        operation=f"stream_raw:{stream_id}",
    )
    return feature_count


def write_formal_stream_output(
    *,
    formal_by_unit: Mapping[str, str],
    units: Sequence[Mapping[str, Any]],
    output_path: Path,
    crs: str,
    metadata: FormalOutputMetadata,
    object_ids: Mapping[str, str],
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    progress: ProgressSink | None = None,
) -> int:
    """Write formal fitted shards with frozen Stream metadata and object IDs."""

    feature_count = 0

    def writer(destination: _RecordDestination) -> None:
        def records() -> Iterator[Mapping[str, Any]]:
            nonlocal feature_count
            for unit_index, unit in enumerate(units, start=1):
                unit_id = str(unit["unit_id"])
                for features in _stream_vector_feature_batches(formal_by_unit[unit_id]):
                    for feature in features:
                        geometry = shape(feature["geometry"])
                        if (
                            geometry.is_empty
                            or not geometry.is_valid
                            or geometry.area <= 0
                        ):
                            raise StreamAssemblyError(
                                f"formal output contains invalid geometry: {unit_id}"
                            )
                        properties = feature["properties"]
                        part_id = str(properties["part_id"])
                        class_code = int(properties["class_code"])
                        feature_count += 1
                        yield {
                            "geometry": feature["geometry"],
                            "properties": {
                                "run_id": metadata.run_id,
                                "result_stream_id": metadata.stream_id,
                                "result_kind": metadata.result_kind,
                                "model_id": metadata.model_id,
                                "fusion_profile_id": metadata.fusion_profile_id,
                                "object_id": object_ids[part_id],
                                "part_id": part_id,
                                "class_code": class_code,
                                "class_name": str(
                                    metadata.class_names[str(class_code)]
                                ),
                                "confidence_mean": float(
                                    properties.get("conf_mean", 0.0)
                                ),
                                "confidence_std": float(
                                    properties.get("conf_std", 0.0)
                                ),
                                "model_version": metadata.model_version,
                                "source": (
                                    "semantic_model"
                                    if metadata.result_kind == "model"
                                    else "semantic_fusion"
                                ),
                                "fit_changed": int(
                                    str(properties.get("fit_status")) == "changed"
                                ),
                                "fit_methods": str(
                                    properties.get("fit_method") or "unchanged"
                                ),
                                "fit_version": str(
                                    properties.get("fit_version")
                                    or metadata.fit_version
                                ),
                                "fit_status": str(
                                    properties.get("fit_status") or "unchanged"
                                ),
                                "origin_unit_ids": unit_id,
                                "vertex_count_before": int(
                                    properties.get("vtx_before", 0)
                                ),
                                "vertex_count_after": int(
                                    properties.get("vtx_after", 0)
                                ),
                                "max_shift_px": float(properties.get("max_shift", 0.0)),
                                "mean_shift_px": float(
                                    properties.get("mean_shift", 0.0)
                                ),
                                "area_change_ratio": float(
                                    properties.get("area_ratio", 0.0)
                                ),
                                "created_at": metadata.created_at,
                            },
                        }
                if progress is not None:
                    progress(unit_index, len(units), feature_count)

        destination.writerecords(records())

    _write_atomic_gpkg(
        output_path,
        "semantic_polygons",
        FORMAL_STREAM_SCHEMA,
        crs,
        writer,
        storage_guard=storage_guard,
        storage_lock_path=storage_lock_path,
        estimated_write_bytes=estimate_source_gpkg_bytes(
            tuple(formal_by_unit.values())
        ),
        operation=f"stream_formal:{metadata.stream_id}",
    )
    return feature_count


def write_fitted_edges_output(
    *,
    edge_artifacts: Sequence[Mapping[str, Any]],
    output_path: Path,
    crs: str,
    stream_id: str,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    progress: ProgressSink | None = None,
) -> EdgeOutputMetrics:
    """Write fitted-edge shards in artifact order and return observed metrics."""

    feature_count = 0
    dense_curve_point_count = 0
    sparse_curve_point_count = 0
    max_chord_error_px = 0.0
    max_segment_arc_length_px = 0.0

    def writer(destination: _RecordDestination) -> None:
        def records() -> Iterator[Mapping[str, Any]]:
            nonlocal feature_count
            nonlocal dense_curve_point_count
            nonlocal sparse_curve_point_count
            nonlocal max_chord_error_px
            nonlocal max_segment_arc_length_px
            for artifact_index, artifact in enumerate(edge_artifacts, start=1):
                for feature in read_stream_vector_features(artifact["path"]):
                    feature_count += 1
                    properties = dict(feature["properties"])
                    properties.pop("part_id", None)
                    dense_curve_point_count += int(properties.get("dense_vtx") or 0)
                    sparse_curve_point_count += int(properties.get("sparse_vtx") or 0)
                    max_chord_error_px = max(
                        max_chord_error_px, float(properties.get("chord_err") or 0.0)
                    )
                    max_segment_arc_length_px = max(
                        max_segment_arc_length_px,
                        float(properties.get("arc_len") or 0.0),
                    )
                    yield {
                        "geometry": mapping(feature["geometry"]),
                        "properties": properties,
                    }
                if progress is not None:
                    progress(artifact_index, len(edge_artifacts), feature_count)

        destination.writerecords(records())

    _write_atomic_gpkg(
        output_path,
        "fitted_edges",
        FITTED_EDGE_SCHEMA,
        crs,
        writer,
        storage_guard=storage_guard,
        storage_lock_path=storage_lock_path,
        estimated_write_bytes=estimate_source_gpkg_bytes(
            tuple(str(item["path"]) for item in edge_artifacts)
        ),
        operation=f"stream_fitted_edges_stage:{stream_id}",
    )
    return EdgeOutputMetrics(
        feature_count=feature_count,
        dense_curve_point_count=dense_curve_point_count,
        sparse_curve_point_count=sparse_curve_point_count,
        max_chord_error_px=max_chord_error_px,
        max_segment_arc_length_px=max_segment_arc_length_px,
    )
