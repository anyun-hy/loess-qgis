"""Spatial execution control-graph persistence for PostgreSQL Run state."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import json_value as _json
from labeling_tool.shared.state.state_values import row_dict as _row_dict
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = ["MAX_TILE_PAGE_SIZE", "RunControlGraphRepository"]

MAX_TILE_PAGE_SIZE = 500


class RunControlGraphRepository:
    """Own ordinary Work Package, spatial-unit, Partition, and Tile state."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def insert_work_packages(
        self, run_id: str, packages: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows() -> Iterator[tuple[Any, ...]]:
            nonlocal count
            for item in packages:
                count += 1
                yield (
                    str(run_id),
                    str(item["package_id"]),
                    int(item["sequence_no"]),
                    int(item.get("estimated_bytes", 0)),
                    _json(
                        {
                            "partition_ids": item.get("partition_ids") or [],
                            "tile_count": int(item.get("tile_count", 0)),
                            "tile_windows": item.get("tile_windows") or [],
                            "neighbor_package_ids": item.get("neighbor_package_ids")
                            or [],
                        }
                    ),
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO work_packages
                       (run_id, package_id, sequence_no, estimated_bytes, metadata_json,
                        status, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def get_work_package(self, run_id: str, package_id: str) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            row = connection.execute(
                "SELECT * FROM work_packages WHERE run_id=%s AND package_id=%s",
                (str(run_id), str(package_id)),
            ).fetchone()
        result: dict[str, Any] | None = _row_dict(row)
        if result is not None:
            result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def page_work_packages(
        self,
        run_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM work_packages WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        sql += " ORDER BY sequence_no LIMIT %s OFFSET %s"
        values.extend((max(1, min(int(limit), 500)), max(0, int(offset))))
        with self._session.connection() as connection:
            rows = connection.execute(sql, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def set_work_package_status(
        self,
        run_id: str,
        package_id: str,
        status: str,
        *,
        expected: str | Sequence[str] | None = None,
    ) -> bool:
        values: list[Any] = [str(status), _now(), str(run_id), str(package_id)]
        sql = (
            "UPDATE work_packages SET status=%s, updated_at=%s "
            "WHERE run_id=%s AND package_id=%s"
        )
        if expected is not None:
            states = [expected] if isinstance(expected, str) else list(expected)
            if not states:
                return False
            sql += " AND status IN (" + ",".join("%s" for _ in states) + ")"
            values.extend(str(item) for item in states)
        with self._session.transaction() as connection:
            return bool(connection.execute(sql, values).rowcount == 1)

    def package_partitions(self, run_id: str, package_id: str) -> list[dict[str, Any]]:
        with self._session.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM partitions WHERE run_id=%s AND package_id=%s
                       ORDER BY row_no, col_no""",
                (str(run_id), str(package_id)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["row"] = item.pop("row_no")
            item["col"] = item.pop("col_no")
            item["core_window"] = json.loads(item.pop("core_window_json"))
            item["halo_window"] = json.loads(item.pop("halo_window_json"))
            result.append(item)
        return result

    def partitions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return every Run Partition in deterministic row/column order."""
        with self._session.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM partitions WHERE run_id=%s
                       ORDER BY row_no, col_no""",
                (str(run_id),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["row"] = item.pop("row_no")
            item["col"] = item.pop("col_no")
            item["core_window"] = json.loads(item.pop("core_window_json"))
            item["halo_window"] = json.loads(item.pop("halo_window_json"))
            result.append(item)
        return result

    def get_partition(self, run_id: str, partition_id: str) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            row = connection.execute(
                "SELECT * FROM partitions WHERE run_id=%s AND partition_id=%s",
                (str(run_id), str(partition_id)),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["row"] = item.pop("row_no")
        item["col"] = item.pop("col_no")
        item["core_window"] = json.loads(item.pop("core_window_json"))
        item["halo_window"] = json.loads(item.pop("halo_window_json"))
        return item

    def package_tiles(self, run_id: str, package_id: str) -> list[dict[str, Any]]:
        package = self.get_work_package(run_id, package_id)
        if package is None:
            raise RunStateError(f"unknown Work Package: {package_id}")
        windows = list(package["metadata"].get("tile_windows") or [])
        selected: dict[str, dict[str, Any]] = {}
        with self._session.connection() as connection:
            for raw_window in windows:
                if not isinstance(raw_window, list) or len(raw_window) != 4:
                    raise RunStateError(
                        f"invalid Tile window in Work Package: {raw_window}"
                    )
                row_start, row_stop, col_start, col_stop = map(int, raw_window)
                for row in connection.execute(
                    """SELECT * FROM tiles WHERE run_id=%s
                           AND row_no>=%s AND row_no<%s AND col_no>=%s AND col_no<%s
                           ORDER BY row_no, col_no""",
                    (str(run_id), row_start, row_stop, col_start, col_stop),
                ):
                    selected[str(row["tile_id"])] = dict(row)
        result = sorted(
            selected.values(), key=lambda item: (item["row_no"], item["col_no"])
        )
        expected = int(package["metadata"].get("tile_count", len(result)))
        if len(result) != expected:
            raise RunStateError(
                f"Work Package Tile count mismatch: expected {expected}, found {len(result)}"
            )
        return result

    def releasable_package_tile_ids(
        self,
        run_id: str,
        completing_package_id: str,
    ) -> list[str]:
        """Return current Package Tiles with no unfinished Package consumers."""
        current_tiles = self.package_tiles(run_id, completing_package_id)
        candidates = {
            (int(tile["row_no"]), int(tile["col_no"])): str(tile["tile_id"])
            for tile in current_tiles
            if str(tile.get("status")) != "excluded"
        }
        candidate_ids = set(candidates.values())
        blocked: set[str] = set()
        with self._session.connection() as connection:
            rows = connection.execute(
                """SELECT package_id, status, metadata_json
                       FROM work_packages
                       WHERE run_id=%s AND package_id!=%s AND status!='ready'
                       ORDER BY sequence_no""",
                (str(run_id), str(completing_package_id)),
            ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            windows = list(metadata.get("tile_windows") or [])
            for raw_window in windows:
                if not isinstance(raw_window, list) or len(raw_window) != 4:
                    raise RunStateError(
                        "invalid Tile window in Work Package: "
                        f"{row['package_id']}={raw_window}"
                    )
                row_start, row_stop, col_start, col_stop = map(int, raw_window)
                for (tile_row, tile_col), tile_id in candidates.items():
                    if (
                        row_start <= tile_row < row_stop
                        and col_start <= tile_col < col_stop
                    ):
                        blocked.add(tile_id)
        return [
            str(tile["tile_id"])
            for tile in current_tiles
            if str(tile["tile_id"]) in candidate_ids
            and str(tile["tile_id"]) not in blocked
        ]

    def work_package_counts(self, run_id: str) -> dict[str, int]:
        with self._session.connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM work_packages
                           WHERE run_id=%s GROUP BY status""",
                    (str(run_id),),
                ).fetchall()
            }

    def insert_partitions(
        self, run_id: str, partitions: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows() -> Iterator[tuple[Any, ...]]:
            nonlocal count
            for item in partitions:
                count += 1
                yield (
                    str(run_id),
                    str(item["partition_id"]),
                    int(item["row"]),
                    int(item["col"]),
                    _json(item["core_window"]),
                    _json(item["halo_window"]),
                    str(item.get("package_id") or "") or None,
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO partitions
                       (run_id, partition_id, row_no, col_no, core_window_json,
                        halo_window_json, package_id, status, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def insert_spatial_units(
        self, run_id: str, units: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        unit_values = list(units)
        rows = [
            (
                str(run_id),
                str(item["unit_id"]),
                str(item["unit_type"]),
                str(item["owner_key"]),
                _json(item["pixel_window"]),
                _json(item.get("dependency_ids") or []),
                str(item.get("status") or "queued"),
                now,
                now,
            )
            for item in unit_values
        ]
        dependency_rows = [
            (str(run_id), str(item["unit_id"]), str(partition_id))
            for item in unit_values
            for partition_id in item.get("dependency_ids") or []
        ]

        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO spatial_units
                       (run_id, unit_id, unit_type, owner_key, pixel_window_json,
                        dependency_ids_json, status, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows,
            )
            connection.executemany(
                """INSERT INTO unit_dependencies
                       (run_id, unit_id, partition_id) VALUES (%s, %s, %s)""",
                dependency_rows,
            )
        return len(unit_values)

    def get_spatial_unit(self, run_id: str, unit_id: str) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            row = connection.execute(
                "SELECT * FROM spatial_units WHERE run_id=%s AND unit_id=%s",
                (str(run_id), str(unit_id)),
            ).fetchone()
        result: dict[str, Any] | None = _row_dict(row)
        if result is not None:
            result["pixel_window"] = json.loads(result.pop("pixel_window_json"))
            result["dependency_ids"] = json.loads(result.pop("dependency_ids_json"))
        return result

    def spatial_units_for_stream(
        self, run_id: str, stream_id: str
    ) -> list[dict[str, Any]]:
        """Return only geometry units registered for one result stream."""

        with self._session.connection() as connection:
            rows = connection.execute(
                """SELECT u.* FROM spatial_units u
                       JOIN stream_units su
                         ON su.run_id=u.run_id AND su.unit_id=u.unit_id
                       WHERE u.run_id=%s AND su.stream_id=%s
                       ORDER BY u.unit_type, u.unit_id""",
                (str(run_id), str(stream_id)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["pixel_window"] = json.loads(item.pop("pixel_window_json"))
            item["dependency_ids"] = json.loads(item.pop("dependency_ids_json"))
            result.append(item)
        return result

    def insert_stream_units(
        self,
        run_id: str,
        stream_ids: Iterable[str],
        unit_ids: Iterable[str],
    ) -> int:
        now = _now()
        streams = [str(value) for value in stream_ids]
        units = [str(value) for value in unit_ids]
        rows = [
            (str(run_id), stream_id, unit_id, now, now)
            for stream_id in streams
            for unit_id in units
        ]
        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO stream_units
                       (run_id, stream_id, unit_id, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                rows,
            )
        return len(rows)

    def set_stream_unit_status(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        status: str,
        *,
        error: str = "",
    ) -> bool:
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE stream_units SET status=%s, error=%s, updated_at=%s
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s""",
                    (
                        str(status),
                        str(error),
                        _now(),
                        str(run_id),
                        str(stream_id),
                        str(unit_id),
                    ),
                ).rowcount
                == 1
            )

    def stream_unit_counts(self, run_id: str, stream_id: str) -> dict[str, int]:
        with self._session.connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM stream_units
                           WHERE run_id=%s AND stream_id=%s GROUP BY status""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            }

    def insert_tiles(self, run_id: str, tiles: Iterable[Mapping[str, Any]]) -> int:
        now = _now()
        count = 0

        def rows() -> Iterator[tuple[Any, ...]]:
            nonlocal count
            for item in tiles:
                count += 1
                yield (
                    str(run_id),
                    str(item["tile_id"]),
                    int(item["row"]),
                    int(item["col"]),
                    int(item.get("width", 512)),
                    int(item.get("height", 512)),
                    _json(item.get("pixel_window") or {}),
                    _json(item.get("bounds") or {}),
                    str(item.get("raster_path") or item.get("path") or ""),
                    str(item.get("sha256") or ""),
                    str(item.get("partition_id") or "") or None,
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO tiles
                       (run_id, tile_id, row_no, col_no, width, height,
                        pixel_window_json, bounds_json, raster_path, sha256,
                        partition_id, status, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def count_tiles(
        self,
        run_id: str,
        *,
        status: str | None = None,
        search: str = "",
    ) -> int:
        sql = "SELECT COUNT(*) FROM tiles WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        if search:
            sql += " AND tile_id LIKE %s ESCAPE '\\'"
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            values.append(f"%{escaped}%")
        with self._session.connection() as connection:
            return int(connection.execute(sql, values).fetchone()[0])

    def update_tile_raster(
        self,
        run_id: str,
        tile_id: str,
        *,
        raster_path: str,
        sha256: str,
    ) -> bool:
        """Record a Tile materialized lazily by a Work Package."""
        digest = str(sha256).lower()
        if len(digest) != 64:
            raise ValueError("Tile sha256 must contain 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "Tile sha256 must contain 64 hexadecimal characters"
            ) from error
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE tiles SET raster_path=%s, sha256=%s, updated_at=%s
                       WHERE run_id=%s AND tile_id=%s AND status!='excluded'""",
                    (
                        str(raster_path),
                        digest,
                        _now(),
                        str(run_id),
                        str(tile_id),
                    ),
                ).rowcount
                == 1
            )

    def page_tiles(
        self,
        run_id: str,
        *,
        limit: int = MAX_TILE_PAGE_SIZE,
        offset: int = 0,
        status: str | None = None,
        partition_id: str | None = None,
        search: str = "",
    ) -> list[dict[str, Any]]:
        page_size = max(1, min(int(limit), MAX_TILE_PAGE_SIZE))
        sql = "SELECT * FROM tiles WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        if partition_id is not None:
            sql += " AND partition_id=%s"
            values.append(str(partition_id))
        if search:
            sql += " AND tile_id LIKE %s ESCAPE '\\'"
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            values.append(f"%{escaped}%")
        sql += " ORDER BY row_no, col_no LIMIT %s OFFSET %s"
        values.extend((page_size, max(0, int(offset))))
        with self._session.connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def count_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        unit_type: str = "",
        status: str = "",
        search: str = "",
    ) -> int:
        sql = (
            "SELECT COUNT(*) FROM stream_units su "
            "JOIN spatial_units u ON u.run_id=su.run_id AND u.unit_id=su.unit_id "
            "WHERE su.run_id=%s AND su.stream_id=%s"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=%s"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=%s"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE %s ESCAPE '\\'"
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            values.append(f"%{escaped}%")
        with self._session.connection() as connection:
            return int(connection.execute(sql, values).fetchone()[0])

    def page_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        limit: int = MAX_TILE_PAGE_SIZE,
        offset: int = 0,
        unit_type: str = "",
        status: str = "",
        search: str = "",
    ) -> list[dict[str, Any]]:
        page_size = max(1, min(int(limit), MAX_TILE_PAGE_SIZE))
        sql = (
            "SELECT su.stream_id, su.unit_id, u.unit_type, su.status, su.error, "
            "u.owner_key, u.pixel_window_json FROM stream_units su "
            "JOIN spatial_units u ON u.run_id=su.run_id AND u.unit_id=su.unit_id "
            "WHERE su.run_id=%s AND su.stream_id=%s"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=%s"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=%s"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE %s ESCAPE '\\'"
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            values.append(f"%{escaped}%")
        sql += " ORDER BY u.unit_type, su.unit_id LIMIT %s OFFSET %s"
        values.extend((page_size, max(0, int(offset))))
        with self._session.connection() as connection:
            rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
        for row in rows:
            row["pixel_window"] = json.loads(row.pop("pixel_window_json"))
        return rows
