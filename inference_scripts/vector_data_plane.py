"""The single columnar vector exchange path used by boundary fitting.

Unit workers exchange GeoParquet shards and scalar boundary signatures only.
This module deliberately has no compatibility writer: a missing Arrow/Pyogrio
backend is a deployment error, rather than a reason to fall back to Fiona.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from shapely.geometry import shape
from shapely.wkb import loads as load_wkb


class VectorDataPlaneError(RuntimeError):
    pass


SIGNATURE_SCHEMA_VERSION = 1


def backend_status() -> dict[str, Any]:
    """Return the non-optional backend contract in a serializable form."""

    pyarrow = importlib.util.find_spec("pyarrow") is not None
    pyogrio = importlib.util.find_spec("pyogrio") is not None
    return {
        "available": pyarrow and pyogrio,
        "pyarrow": pyarrow,
        "pyogrio": pyogrio,
        "backend": "pyarrow+pyogrio" if pyarrow and pyogrio else None,
    }


def require_backend() -> None:
    status = backend_status()
    if not status["available"]:
        raise VectorDataPlaneError(
            "the required columnar vector data plane is unavailable; install "
            f"pyarrow and pyogrio: {status}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _geo_metadata(crs: str, geometries: list[bytes]) -> bytes:
    types = sorted({load_wkb(value).geom_type for value in geometries})
    try:
        from pyproj import CRS
        crs_value: Any = CRS.from_user_input(crs).to_json_dict()
    except (ImportError, ValueError):
        crs_value = str(crs)
    return json.dumps({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": types, "crs": crs_value}},
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_geoparquet(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    crs: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Atomically publish one validated GeoParquet shard and manifest."""

    require_backend()
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    rows: list[dict[str, Any]] = []
    for item in records:
        row = dict(item)
        geometry = row.pop("geometry", None)
        if geometry is None:
            raise VectorDataPlaneError("GeoParquet record has no geometry")
        if isinstance(geometry, Mapping):
            geometry = shape(geometry)
        if geometry.is_empty:
            raise VectorDataPlaneError("GeoParquet record has empty geometry")
        row["geometry"] = bytes(geometry.wkb)
        row["source_sha256"] = str(source_sha256)
        if not str(row.get("part_id") or ""):
            raise VectorDataPlaneError("GeoParquet record has no part_id")
        rows.append(row)
    rows.sort(key=lambda row: str(row["part_id"]))
    table = pa.Table.from_pylist(rows) if rows else pa.table({
        "part_id": pa.array([], type=pa.string()),
        "class_code": pa.array([], type=pa.int64()),
        "geometry": pa.array([], type=pa.binary()),
        "source_sha256": pa.array([], type=pa.string()),
    })
    if "geometry" not in table.column_names:
        raise VectorDataPlaneError("GeoParquet table has no geometry column")
    metadata = dict(table.schema.metadata or {})
    geometry_values = [bytes(item) for item in table["geometry"].to_pylist()]
    metadata[b"geo"] = _geo_metadata(str(crs), geometry_values)
    metadata[b"loess_vector_data_plane"] = json.dumps({
        "schema_version": 1,
        "source_sha256": str(source_sha256),
    }, sort_keys=True).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    manifest_path = target.with_name(f"{target.name}.manifest.json")
    try:
        pq.write_table(table, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        # Read back before publication; this is also the schema integrity gate.
        checked = pq.read_table(temporary)
        if checked.num_rows != len(rows) or "geometry" not in checked.column_names:
            raise VectorDataPlaneError("GeoParquet validation failed")
        os.replace(temporary, target)
        manifest = {
            "schema_version": 1, "status": "committed", "format": "GeoParquet",
            "path": str(target), "crs": str(crs), "feature_count": len(rows),
            "fields": list(table.column_names), "source_sha256": str(source_sha256),
            "artifact_sha256": _sha256(target),
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def read_geoparquet(path: str | Path) -> tuple[dict[str, Any], Any]:
    """Validate a committed shard and return its Arrow table."""

    require_backend()
    import pyarrow.parquet as pq

    target = Path(path)
    manifest = json.loads(target.with_name(f"{target.name}.manifest.json").read_text())
    if manifest.get("status") != "committed" or Path(manifest.get("path", "")) != target:
        raise VectorDataPlaneError(f"uncommitted GeoParquet shard: {target}")
    if _sha256(target) != manifest.get("artifact_sha256"):
        raise VectorDataPlaneError(f"GeoParquet checksum mismatch: {target}")
    table = pq.read_table(target)
    if table.num_rows != int(manifest["feature_count"]) or "geometry" not in table.column_names:
        raise VectorDataPlaneError(f"GeoParquet schema/count mismatch: {target}")
    return manifest, table


@dataclass(frozen=True, order=True)
class BoundaryInterval:
    part_id: str
    class_code: int
    axis: str
    fixed_coordinate: float
    interval_start: float
    interval_end: float


def _boundary_lines(geometry):
    boundary = geometry.boundary
    if boundary.geom_type == "LineString":
        yield boundary
    else:
        yield from (item for item in boundary.geoms if item.geom_type == "LineString")


def unit_boundary_signatures(
    records: Iterable[Mapping[str, Any]], *, stream_id: str, unit_id: str,
    pixel_window: Mapping[str, Any], tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    """Extract only scalar outer-window intervals; no geometry enters assembly."""

    limits = (("x", float(pixel_window["x0"]), "west"), ("x", float(pixel_window["x1"]), "east"),
              ("y", float(pixel_window["y0"]), "north"), ("y", float(pixel_window["y1"]), "south"))
    output: list[dict[str, Any]] = []
    for record in records:
        for line in _boundary_lines(record["geometry"]):
            for first, second in zip(line.coords, line.coords[1:]):
                x1, y1 = float(first[0]), float(first[1]); x2, y2 = float(second[0]), float(second[1])
                if x1 == x2 and y1 != y2:
                    axis, fixed, start, end = "x", x1, min(y1, y2), max(y1, y2)
                elif y1 == y2 and x1 != x2:
                    axis, fixed, start, end = "y", y1, min(x1, x2), max(x1, x2)
                else:
                    continue
                direction = next((side for candidate_axis, candidate_fixed, side in limits
                                  if axis == candidate_axis and abs(fixed - candidate_fixed) <= tolerance), "")
                if direction:
                    output.append({"stream_id": str(stream_id), "unit_id": str(unit_id),
                                   "part_id": str(record["polygon_id"]), "class_code": int(record["class_code"]),
                                   "edge_direction": direction, "axis": axis, "fixed_coordinate": fixed,
                                   "interval_start": start, "interval_end": end})
    return sorted(output, key=lambda item: (item["edge_direction"], item["axis"], item["fixed_coordinate"], item["interval_start"], item["part_id"]))


def write_boundary_signatures(path: str | Path, records: list[dict[str, Any]], *, stream_id: str, unit_id: str) -> dict[str, Any]:
    payload = {"schema_version": SIGNATURE_SCHEMA_VERSION, "stream_id": str(stream_id), "unit_id": str(unit_id), "records": records}
    payload["record_count"] = len(records)
    payload["records_sha256"] = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); _atomic_json(target, payload)
    return payload


def read_boundary_signatures(path: str | Path, *, stream_id: str, unit_id: str) -> list[BoundaryInterval]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records")
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if payload.get("schema_version") != SIGNATURE_SCHEMA_VERSION or payload.get("stream_id") != str(stream_id) or payload.get("unit_id") != str(unit_id) or payload.get("records_sha256") != digest or len(records) != int(payload.get("record_count", -1)):
        raise VectorDataPlaneError(f"boundary signature integrity failure: {path}")
    return [BoundaryInterval(str(item["part_id"]), int(item["class_code"]), str(item["axis"]), float(item["fixed_coordinate"]), float(item["interval_start"]), float(item["interval_end"])) for item in records]


def signature_links(left: Iterable[BoundaryInterval], right: Iterable[BoundaryInterval], *, tolerance: float) -> list[tuple[str, str, int]]:
    """Join scalar intervals deterministically, without spatial/object graphs."""

    index: dict[tuple[int, str, float], list[BoundaryInterval]] = {}
    for item in right:
        index.setdefault((item.class_code, item.axis, item.fixed_coordinate), []).append(item)
    links = set()
    for item in left:
        for peer in index.get((item.class_code, item.axis, item.fixed_coordinate), ()):
            if min(item.interval_end, peer.interval_end) - max(item.interval_start, peer.interval_start) > tolerance:
                links.add((item.part_id, peer.part_id, item.class_code))
    return sorted(links)
