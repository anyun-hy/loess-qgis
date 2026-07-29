"""SQLite/WAL state store for large, resumable inference runs."""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
MAX_TILE_PAGE_SIZE = 500


class RunStateError(RuntimeError):
    pass


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class RunStateDB:
    """Short-transaction database API safe for QGIS and worker processes."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000):
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RunStateError(f"cannot enable SQLite WAL: {journal_mode}")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    run_spec_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS streams (
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    profile_id TEXT NOT NULL DEFAULT '',
                    version TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, stream_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS work_packages (
                    run_id TEXT NOT NULL,
                    package_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    estimated_bytes INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, package_id),
                    UNIQUE (run_id, sequence_no),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS partitions (
                    run_id TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    row_no INTEGER NOT NULL,
                    col_no INTEGER NOT NULL,
                    core_window_json TEXT NOT NULL,
                    halo_window_json TEXT NOT NULL,
                    package_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, partition_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, package_id)
                        REFERENCES work_packages(run_id, package_id) ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS tiles (
                    run_id TEXT NOT NULL,
                    tile_id TEXT NOT NULL,
                    row_no INTEGER NOT NULL,
                    col_no INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    pixel_window_json TEXT NOT NULL DEFAULT '{}',
                    bounds_json TEXT NOT NULL DEFAULT '{}',
                    raster_path TEXT NOT NULL DEFAULT '',
                    sha256 TEXT NOT NULL DEFAULT '',
                    partition_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, tile_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, partition_id)
                        REFERENCES partitions(run_id, partition_id) ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS spatial_units (
                    run_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    owner_key TEXT NOT NULL,
                    pixel_window_json TEXT NOT NULL,
                    dependency_ids_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, unit_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS unit_dependencies (
                    run_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    PRIMARY KEY (run_id, unit_id, partition_id),
                    FOREIGN KEY (run_id, unit_id)
                        REFERENCES spatial_units(run_id, unit_id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, partition_id)
                        REFERENCES partitions(run_id, partition_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS stream_units (
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, stream_id, unit_id),
                    FOREIGN KEY (run_id, stream_id)
                        REFERENCES streams(run_id, stream_id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id, unit_id)
                        REFERENCES spatial_units(run_id, unit_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    stream_id TEXT NOT NULL DEFAULT '',
                    tile_id TEXT NOT NULL DEFAULT '',
                    unit_id TEXT NOT NULL DEFAULT '',
                    package_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT NOT NULL DEFAULT '',
                    lease_token TEXT NOT NULL DEFAULT '',
                    lease_expires REAL,
                    heartbeat_at TEXT,
                    pid INTEGER,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL DEFAULT '',
                    unit_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    byte_count INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'writing',
                    ref_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (run_id, stream_id, unit_id, kind, path),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifact_dependencies (
                    job_id INTEGER NOT NULL,
                    artifact_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, artifact_id),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS unit_report_summaries (
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fit_version TEXT NOT NULL DEFAULT '',
                    chain_count INTEGER NOT NULL DEFAULT 0,
                    shared_chain_count INTEGER NOT NULL DEFAULT 0,
                    spline_count INTEGER NOT NULL DEFAULT 0,
                    unchanged_count INTEGER NOT NULL DEFAULT 0,
                    skipped_invalid_count INTEGER NOT NULL DEFAULT 0,
                    max_displacement_px REAL NOT NULL DEFAULT 0,
                    diagnostic_count INTEGER NOT NULL DEFAULT 0,
                    fitted_edge_count INTEGER NOT NULL DEFAULT 0,
                    report_path TEXT NOT NULL,
                    report_byte_count INTEGER NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    report_mtime_ns INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, stream_id, unit_id),
                    FOREIGN KEY (run_id, stream_id, unit_id)
                        REFERENCES stream_units(run_id, stream_id, unit_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS object_links (
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    left_part_id TEXT NOT NULL,
                    right_part_id TEXT NOT NULL,
                    class_code INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, stream_id, left_part_id, right_part_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS object_nodes (
                    run_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    class_code INTEGER NOT NULL,
                    unit_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    rank_value INTEGER NOT NULL DEFAULT 0,
                    object_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, stream_id, part_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    stream_id TEXT NOT NULL DEFAULT '',
                    job_id INTEGER,
                    message TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_streams_status
                    ON streams(run_id, status, stream_id);
                CREATE INDEX IF NOT EXISTS idx_packages_status
                    ON work_packages(run_id, status, sequence_no);
                CREATE INDEX IF NOT EXISTS idx_partitions_status
                    ON partitions(run_id, status, package_id);
                CREATE INDEX IF NOT EXISTS idx_tiles_grid_status
                    ON tiles(run_id, row_no, col_no, status);
                CREATE INDEX IF NOT EXISTS idx_tiles_partition
                    ON tiles(run_id, partition_id, status);
                CREATE INDEX IF NOT EXISTS idx_units_status
                    ON spatial_units(run_id, unit_type, status, unit_id);
                CREATE INDEX IF NOT EXISTS idx_unit_dependencies_partition
                    ON unit_dependencies(run_id, partition_id, unit_id);
                CREATE INDEX IF NOT EXISTS idx_stream_units_status
                    ON stream_units(run_id, stream_id, status, unit_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_claim
                    ON jobs(run_id, status, priority DESC, job_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_stream_unit
                    ON jobs(run_id, stream_id, unit_id, status);
                CREATE INDEX IF NOT EXISTS idx_jobs_tile
                    ON jobs(run_id, stream_id, tile_id, status);
                CREATE INDEX IF NOT EXISTS idx_artifacts_state
                    ON artifacts(run_id, status, ref_count, artifact_id);
                CREATE INDEX IF NOT EXISTS idx_unit_report_summaries_stream
                    ON unit_report_summaries(run_id, stream_id, status, unit_id);
                CREATE INDEX IF NOT EXISTS idx_object_nodes_root
                    ON object_nodes(run_id, stream_id, parent_id, part_id);
                CREATE INDEX IF NOT EXISTS idx_events_time
                    ON events(run_id, event_id, timestamp);

                CREATE TRIGGER IF NOT EXISTS artifact_dependency_after_insert
                AFTER INSERT ON artifact_dependencies
                BEGIN
                    UPDATE artifacts SET ref_count=ref_count + 1, updated_at=NEW.created_at
                    WHERE artifact_id=NEW.artifact_id;
                END;

                CREATE TRIGGER IF NOT EXISTS artifact_dependency_after_delete
                AFTER DELETE ON artifact_dependencies
                BEGIN
                    UPDATE artifacts SET ref_count=MAX(0, ref_count - 1), updated_at=CURRENT_TIMESTAMP
                    WHERE artifact_id=OLD.artifact_id;
                END;
                """
            )
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current not in (0, SCHEMA_VERSION):
                raise RunStateError(
                    f"unsupported run-state schema {current}; expected {SCHEMA_VERSION}"
                )
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def pragmas(self) -> dict[str, Any]:
        with self._connect() as connection:
            return {
                "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
                "foreign_keys": connection.execute("PRAGMA foreign_keys").fetchone()[0],
                "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
                "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            }

    def create_run(
        self,
        run_id: str,
        run_spec_sha256: str,
        *,
        status: str = "preflight",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO runs
                   (run_id, schema_version, status, run_spec_sha256,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(run_id),
                    SCHEMA_VERSION,
                    str(status),
                    str(run_spec_sha256),
                    _json(dict(metadata or {})),
                    now,
                    now,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id=?", (str(run_id),)
                ).fetchone()
            )

    def set_run_status(
        self,
        run_id: str,
        status: str,
        *,
        expected: str | Sequence[str] | None = None,
    ) -> bool:
        values: list[Any] = [str(status), _now(), str(run_id)]
        sql = "UPDATE runs SET status=?, updated_at=? WHERE run_id=?"
        if expected is not None:
            states = [expected] if isinstance(expected, str) else list(expected)
            if not states:
                return False
            sql += " AND status IN (" + ",".join("?" for _ in states) + ")"
            values.extend(str(item) for item in states)
        with self.transaction() as connection:
            return connection.execute(sql, values).rowcount == 1

    def register_streams(
        self, run_id: str, streams: Iterable[Mapping[str, Any]]
    ) -> None:
        now = _now()
        rows = (
            (
                str(run_id),
                str(item["stream_id"]),
                str(item["kind"]),
                str(item.get("model_id") or ""),
                str(item.get("profile_id") or ""),
                str(item.get("version") or ""),
                str(item.get("status") or "pending"),
                now,
                now,
            )
            for item in streams
        )
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO streams
                   (run_id, stream_id, kind, model_id, profile_id, version,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def set_stream_status(
        self,
        run_id: str,
        stream_id: str,
        status: str,
        *,
        error: str = "",
    ) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE streams SET status=?, error=?, updated_at=?
                   WHERE run_id=? AND stream_id=?""",
                (str(status), str(error), _now(), str(run_id), str(stream_id)),
            ).rowcount == 1

    def insert_work_packages(
        self, run_id: str, packages: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
            nonlocal count
            for item in packages:
                count += 1
                yield (
                    str(run_id),
                    str(item["package_id"]),
                    int(item["sequence_no"]),
                    int(item.get("estimated_bytes", 0)),
                    _json({
                        "partition_ids": item.get("partition_ids") or [],
                        "tile_count": int(item.get("tile_count", 0)),
                        "tile_windows": item.get("tile_windows") or [],
                        "neighbor_package_ids": item.get("neighbor_package_ids") or [],
                    }),
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO work_packages
                   (run_id, package_id, sequence_no, estimated_bytes, metadata_json,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows(),
            )
        return count

    def get_work_package(self, run_id: str, package_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM work_packages WHERE run_id=? AND package_id=?",
                (str(run_id), str(package_id)),
            ).fetchone()
        result = _row_dict(row)
        if result is not None:
            result["metadata"] = json.loads(result.pop("metadata_json"))
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
            "UPDATE work_packages SET status=?, updated_at=? "
            "WHERE run_id=? AND package_id=?"
        )
        if expected is not None:
            states = [expected] if isinstance(expected, str) else list(expected)
            if not states:
                return False
            sql += " AND status IN (" + ",".join("?" for _ in states) + ")"
            values.extend(str(item) for item in states)
        with self.transaction() as connection:
            return connection.execute(sql, values).rowcount == 1

    def package_partitions(self, run_id: str, package_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM partitions WHERE run_id=? AND package_id=?
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

    def get_partition(self, run_id: str, partition_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM partitions WHERE run_id=? AND partition_id=?",
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
        with self._connect() as connection:
            for raw_window in windows:
                if not isinstance(raw_window, list) or len(raw_window) != 4:
                    raise RunStateError(f"invalid Tile window in Work Package: {raw_window}")
                row_start, row_stop, col_start, col_stop = map(int, raw_window)
                for row in connection.execute(
                    """SELECT * FROM tiles WHERE run_id=?
                       AND row_no>=? AND row_no<? AND col_no>=? AND col_no<?
                       ORDER BY row_no, col_no""",
                    (str(run_id), row_start, row_stop, col_start, col_stop),
                ):
                    selected[str(row["tile_id"])] = dict(row)
        result = sorted(selected.values(), key=lambda item: (item["row_no"], item["col_no"]))
        expected = int(package["metadata"].get("tile_count", len(result)))
        if len(result) != expected:
            raise RunStateError(
                f"Work Package Tile count mismatch: expected {expected}, found {len(result)}"
            )
        return result

    def work_package_counts(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM work_packages
                       WHERE run_id=? GROUP BY status""",
                    (str(run_id),),
                ).fetchall()
            }

    def open_frontier_summary(self, run_id: str) -> dict[str, Any]:
        """Return cross-package spatial units with only some dependencies ready."""
        unit_state = """
            SELECT ud.unit_id,
                   SUM(CASE WHEN wp.status='ready' THEN 1 ELSE 0 END) AS ready_count,
                   COUNT(*) AS dependency_count
            FROM unit_dependencies ud
            JOIN partitions p
              ON p.run_id=ud.run_id AND p.partition_id=ud.partition_id
            JOIN work_packages wp
              ON wp.run_id=p.run_id AND wp.package_id=p.package_id
            WHERE ud.run_id=?
            GROUP BY ud.unit_id
        """
        with self._connect() as connection:
            rows = connection.execute(unit_state, (str(run_id),)).fetchall()
            open_unit_ids = [
                str(row["unit_id"])
                for row in rows
                if 0 < int(row["ready_count"]) < int(row["dependency_count"])
            ]
            if not open_unit_ids:
                return {"unit_count": 0, "unit_ids": [], "neighbor_package_ids": []}
            placeholders = ",".join("?" for _ in open_unit_ids)
            package_rows = connection.execute(
                f"""SELECT DISTINCT wp.package_id, wp.sequence_no
                    FROM unit_dependencies ud
                    JOIN partitions p
                      ON p.run_id=ud.run_id AND p.partition_id=ud.partition_id
                    JOIN work_packages wp
                      ON wp.run_id=p.run_id AND wp.package_id=p.package_id
                    WHERE ud.run_id=? AND ud.unit_id IN ({placeholders})
                      AND wp.status IN ('queued','interrupted')
                    ORDER BY wp.sequence_no""",
                [str(run_id), *open_unit_ids],
            ).fetchall()
        return {
            "unit_count": len(open_unit_ids),
            "unit_ids": open_unit_ids,
            "neighbor_package_ids": [str(row["package_id"]) for row in package_rows],
        }

    def page_work_packages(
        self,
        run_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM work_packages WHERE run_id=?"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=?"
            values.append(str(status))
        sql += " ORDER BY sequence_no LIMIT ? OFFSET ?"
        values.extend((max(1, min(int(limit), 500)), max(0, int(offset))))
        with self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def insert_partitions(
        self, run_id: str, partitions: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
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

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO partitions
                   (run_id, partition_id, row_no, col_no, core_window_json,
                    halo_window_json, package_id, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO spatial_units
                   (run_id, unit_id, unit_type, owner_key, pixel_window_json,
                    dependency_ids_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            connection.executemany(
                """INSERT INTO unit_dependencies
                   (run_id, unit_id, partition_id) VALUES (?, ?, ?)""",
                dependency_rows,
            )
        return len(unit_values)

    def get_spatial_unit(self, run_id: str, unit_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM spatial_units WHERE run_id=? AND unit_id=?",
                (str(run_id), str(unit_id)),
            ).fetchone()
        result = _row_dict(row)
        if result is not None:
            result["pixel_window"] = json.loads(result.pop("pixel_window_json"))
            result["dependency_ids"] = json.loads(result.pop("dependency_ids_json"))
        return result

    def spatial_units(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM spatial_units WHERE run_id=?
                   ORDER BY unit_type, unit_id""",
                (str(run_id),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["pixel_window"] = json.loads(item.pop("pixel_window_json"))
            item["dependency_ids"] = json.loads(item.pop("dependency_ids_json"))
            result.append(item)
        return result

    def set_spatial_unit_status(
        self, run_id: str, unit_id: str, status: str
    ) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE spatial_units SET status=?, updated_at=?
                   WHERE run_id=? AND unit_id=?""",
                (str(status), _now(), str(run_id), str(unit_id)),
            ).rowcount == 1

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
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO stream_units
                   (run_id, stream_id, unit_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
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
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE stream_units SET status=?, error=?, updated_at=?
                   WHERE run_id=? AND stream_id=? AND unit_id=?""",
                (
                    str(status),
                    str(error),
                    _now(),
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                ),
            ).rowcount == 1

    def stream_unit_counts(self, run_id: str, stream_id: str) -> dict[str, int]:
        with self._connect() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM stream_units
                       WHERE run_id=? AND stream_id=? GROUP BY status""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            }

    def insert_tiles(
        self, run_id: str, tiles: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
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

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO tiles
                   (run_id, tile_id, row_no, col_no, width, height,
                    pixel_window_json, bounds_json, raster_path, sha256,
                    partition_id, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows(),
            )
        return count

    def count_tiles(self, run_id: str, *, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM tiles WHERE run_id=?"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=?"
            values.append(str(status))
        with self._connect() as connection:
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
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE tiles SET raster_path=?, sha256=?, updated_at=?
                   WHERE run_id=? AND tile_id=? AND status!='excluded'""",
                (
                    str(raster_path),
                    digest,
                    _now(),
                    str(run_id),
                    str(tile_id),
                ),
            ).rowcount == 1

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
        sql = "SELECT * FROM tiles WHERE run_id=?"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=?"
            values.append(str(status))
        if partition_id is not None:
            sql += " AND partition_id=?"
            values.append(str(partition_id))
        if search:
            sql += " AND tile_id LIKE ? ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        sql += " ORDER BY row_no, col_no LIMIT ? OFFSET ?"
        values.extend((page_size, max(0, int(offset))))
        with self._connect() as connection:
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
            "WHERE su.run_id=? AND su.stream_id=?"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=?"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=?"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE ? ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        with self._connect() as connection:
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
            "WHERE su.run_id=? AND su.stream_id=?"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=?"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=?"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE ? ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        sql += " ORDER BY u.unit_type, su.unit_id LIMIT ? OFFSET ?"
        values.extend((page_size, max(0, int(offset))))
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
        for row in rows:
            row["pixel_window"] = json.loads(row.pop("pixel_window_json"))
        return rows

    def stream_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM streams WHERE run_id=? ORDER BY stream_id",
                    (str(run_id),),
                ).fetchall()
            ]

    def insert_jobs(
        self, run_id: str, jobs: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
            nonlocal count
            for item in jobs:
                count += 1
                yield (
                    str(run_id),
                    str(item["job_type"]),
                    str(item.get("stream_id") or ""),
                    str(item.get("tile_id") or ""),
                    str(item.get("unit_id") or ""),
                    str(item.get("package_id") or ""),
                    str(item.get("status") or "queued"),
                    int(item.get("priority", 0)),
                    int(item.get("max_attempts", 3)),
                    now,
                    now,
                )

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO jobs
                   (run_id, job_type, stream_id, tile_id, unit_id, package_id,
                    status, priority, max_attempts, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows(),
            )
        return count

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (int(job_id),)
                ).fetchone()
            )

    def lease_next_job(
        self,
        run_id: str,
        worker_id: str,
        *,
        job_types: Sequence[str] = (),
        lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        token = uuid.uuid4().hex
        expires = time.time() + max(1.0, float(lease_seconds))
        now = _now()
        sql = (
            "SELECT * FROM jobs WHERE run_id=? "
            "AND status IN ('queued','interrupted') AND attempt < max_attempts "
            "AND (job_type!='unit_fit' OR "
            "(SELECT COUNT(*) FROM artifact_dependencies ad WHERE ad.job_id=jobs.job_id)="
            "(SELECT COUNT(*) FROM unit_dependencies ud "
            " WHERE ud.run_id=jobs.run_id AND ud.unit_id=jobs.unit_id))"
        )
        values: list[Any] = [str(run_id)]
        if job_types:
            sql += " AND job_type IN (" + ",".join("?" for _ in job_types) + ")"
            values.extend(str(item) for item in job_types)
        sql += " ORDER BY priority DESC, job_id LIMIT 1"
        with self.transaction() as connection:
            row = connection.execute(sql, values).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """UPDATE jobs SET status='running', attempt=attempt+1,
                   worker_id=?, lease_token=?, lease_expires=?, heartbeat_at=?,
                   updated_at=? WHERE job_id=? AND status IN ('queued','interrupted')""",
                (str(worker_id), token, expires, now, now, int(row["job_id"])),
            )
            if updated.rowcount != 1:
                return None
            return _row_dict(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (int(row["job_id"]),)
                ).fetchone()
            )

    def lease_next_work_package(
        self,
        run_id: str,
        worker_id: str,
        *,
        max_open_frontier_units: int,
        lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        """Prefer a package that closes open Seam/Junction dependencies."""
        frontier = self.open_frontier_summary(run_id)
        preferred = list(frontier["neighbor_package_ids"])
        if int(frontier["unit_count"]) >= max(1, int(max_open_frontier_units)) and preferred:
            placeholders = ",".join("?" for _ in preferred)
            with self._connect() as connection:
                row = connection.execute(
                    f"""SELECT j.job_id
                        FROM jobs j
                        JOIN work_packages wp
                          ON wp.run_id=j.run_id AND wp.package_id=j.package_id
                        WHERE j.run_id=? AND j.job_type='work_package'
                          AND j.status IN ('queued','interrupted')
                          AND j.attempt < j.max_attempts
                          AND j.package_id IN ({placeholders})
                        ORDER BY j.priority DESC, wp.sequence_no, j.job_id LIMIT 1""",
                    [str(run_id), *preferred],
                ).fetchone()
            if row is not None:
                return self.lease_job(
                    int(row["job_id"]), worker_id, lease_seconds=lease_seconds
                )
        return self.lease_next_job(
            run_id,
            worker_id,
            job_types=("work_package",),
            lease_seconds=lease_seconds,
        )

    def lease_job(
        self,
        job_id: int,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        token = uuid.uuid4().hex
        expires = time.time() + max(1.0, float(lease_seconds))
        now = _now()
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT * FROM jobs WHERE job_id=?
                   AND status IN ('queued','interrupted') AND attempt < max_attempts
                   AND (job_type!='unit_fit' OR
                     (SELECT COUNT(*) FROM artifact_dependencies ad
                      WHERE ad.job_id=jobs.job_id)=
                     (SELECT COUNT(*) FROM unit_dependencies ud
                      WHERE ud.run_id=jobs.run_id AND ud.unit_id=jobs.unit_id))""",
                (int(job_id),),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """UPDATE jobs SET status='running', attempt=attempt+1,
                   worker_id=?, lease_token=?, lease_expires=?, heartbeat_at=?,
                   updated_at=? WHERE job_id=? AND status IN ('queued','interrupted')""",
                (str(worker_id), token, expires, now, now, int(job_id)),
            )
            if updated.rowcount != 1:
                return None
            return _row_dict(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (int(job_id),)
                ).fetchone()
            )

    def heartbeat(
        self,
        job_id: int,
        lease_token: str,
        *,
        current: int,
        total: int,
        lease_seconds: float = 60.0,
    ) -> bool:
        now = _now()
        expires = time.time() + max(1.0, float(lease_seconds))
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET progress_current=?, progress_total=?,
                   heartbeat_at=?, lease_expires=?, updated_at=?
                   WHERE job_id=? AND status='running' AND lease_token=?""",
                (
                    max(0, int(current)),
                    max(0, int(total)),
                    now,
                    expires,
                    now,
                    int(job_id),
                    str(lease_token),
                ),
            ).rowcount == 1

    def finish_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        status: str = "ready",
        error: str = "",
    ) -> bool:
        if status not in {"ready", "failed", "stopped"}:
            raise ValueError(f"invalid terminal job status: {status}")
        now = _now()
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status=?, error=?, worker_id='',
                   lease_token='', lease_expires=NULL, heartbeat_at=?, updated_at=?
                   WHERE job_id=? AND status='running' AND lease_token=?""",
                (str(status), str(error), now, now, int(job_id), str(lease_token)),
            ).rowcount == 1

    def interrupt_job(self, job_id: int, lease_token: str) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='', lease_token='',
                   lease_expires=NULL, updated_at=?
                   WHERE job_id=? AND status='running' AND lease_token=?""",
                (_now(), int(job_id), str(lease_token)),
            ).rowcount == 1

    def requeue_failed_job(self, job_id: int) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='queued', error='', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=?
                   WHERE job_id=? AND status='failed' AND attempt < max_attempts""",
                (_now(), int(job_id)),
            ).rowcount == 1

    def interrupt_expired_jobs(self, *, now_epoch: float | None = None) -> int:
        now_value = time.time() if now_epoch is None else float(now_epoch)
        now = _now()
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=?
                   WHERE status='running' AND lease_expires IS NOT NULL
                   AND lease_expires < ?""",
                (now, now_value),
            ).rowcount

    def interrupt_run_jobs(self, run_id: str) -> int:
        """Recover only the selected run after a QGIS/process interruption."""
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=?
                   WHERE run_id=? AND status='running'""",
                (_now(), str(run_id)),
            ).rowcount

    def requeue_failed_jobs(self, run_id: str) -> int:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='queued', error='', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=?
                   WHERE run_id=? AND status='failed' AND attempt < max_attempts""",
                (_now(), str(run_id)),
            ).rowcount

    def register_artifact(
        self,
        run_id: str,
        kind: str,
        path: str | Path,
        *,
        stream_id: str = "",
        unit_id: str = "",
    ) -> int:
        """Register an artifact before its writer starts the atomic file write."""
        now = _now()
        resolved_path = str(Path(path).expanduser().resolve())
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts
                   (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(kind),
                    resolved_path,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """SELECT artifact_id FROM artifacts
                   WHERE run_id=? AND stream_id=? AND unit_id=? AND kind=? AND path=?""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(kind),
                    resolved_path,
                ),
            ).fetchone()
            if row is None:
                raise RunStateError("artifact registration did not create or find a row")
            return int(row["artifact_id"])

    def get_artifact(self, artifact_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=?", (int(artifact_id),)
                ).fetchone()
            )

    def mark_artifact_ready(
        self,
        artifact_id: int,
        *,
        byte_count: int,
        sha256: str,
    ) -> bool:
        if int(byte_count) < 0:
            raise ValueError("artifact byte_count cannot be negative")
        if len(str(sha256)) != 64:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        try:
            int(str(sha256), 16)
        except ValueError as error:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters") from error
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status='ready', byte_count=?, sha256=?, updated_at=?
                   WHERE artifact_id=? AND status IN ('writing','failed')""",
                (int(byte_count), str(sha256).lower(), _now(), int(artifact_id)),
            ).rowcount == 1

    def mark_artifact_failed(self, artifact_id: int) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status='failed', updated_at=?
                   WHERE artifact_id=? AND status='writing'""",
                (_now(), int(artifact_id)),
            ).rowcount == 1

    def add_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Attach a ready input to a job; the trigger updates ref_count atomically."""
        with self.transaction() as connection:
            relation = connection.execute(
                """SELECT 1 FROM jobs j JOIN artifacts a ON a.run_id=j.run_id
                   WHERE j.job_id=? AND a.artifact_id=? AND a.status='ready'""",
                (int(job_id), int(artifact_id)),
            ).fetchone()
            if relation is None:
                raise RunStateError(
                    "artifact dependency requires a ready artifact from the same run"
                )
            cursor = connection.execute(
                """INSERT OR IGNORE INTO artifact_dependencies
                   (job_id, artifact_id, created_at) VALUES (?, ?, ?)""",
                (int(job_id), int(artifact_id), _now()),
            )
            return cursor.rowcount == 1

    def link_partition_artifact(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        artifact_id: int,
    ) -> int:
        """Link one ready Partition probability to every dependent unit job."""
        with self.transaction() as connection:
            artifact = connection.execute(
                """SELECT 1 FROM artifacts WHERE artifact_id=? AND run_id=?
                   AND stream_id=? AND unit_id=? AND kind='partition_probability'
                   AND status='ready'""",
                (int(artifact_id), str(run_id), str(stream_id), str(partition_id)),
            ).fetchone()
            if artifact is None:
                raise RunStateError("Partition dependency Artifact is not ready or mismatched")
            job_ids = [
                int(row["job_id"])
                for row in connection.execute(
                    """SELECT j.job_id FROM jobs j
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       WHERE j.run_id=? AND j.stream_id=? AND j.job_type='unit_fit'
                         AND d.partition_id=?""",
                    (str(run_id), str(stream_id), str(partition_id)),
                ).fetchall()
            ]
            now = _now()
            inserted = 0
            for job_id in job_ids:
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO artifact_dependencies
                       (job_id, artifact_id, created_at) VALUES (?, ?, ?)""",
                    (job_id, int(artifact_id), now),
                )
                inserted += cursor.rowcount
            return inserted

    def release_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Release one job input; the trigger prevents a negative ref_count."""
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=? AND artifact_id=?",
                (int(job_id), int(artifact_id)),
            )
            return cursor.rowcount == 1

    def release_job_artifacts(self, job_id: int) -> int:
        with self.transaction() as connection:
            return connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=?", (int(job_id),)
            ).rowcount

    def job_for_unit(
        self, run_id: str, stream_id: str, unit_id: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM jobs WHERE run_id=? AND stream_id=?
                       AND unit_id=? AND job_type='unit_fit'""",
                    (str(run_id), str(stream_id), str(unit_id)),
                ).fetchone()
            )

    def cleanup_candidates(
        self,
        run_id: str,
        *,
        limit: int = 100,
        kinds: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return ready, unreferenced artifacts; deletion remains an explicit caller action."""
        sql = "SELECT * FROM artifacts WHERE run_id=? AND status='ready' AND ref_count=0"
        values: list[Any] = [str(run_id)]
        if kinds:
            sql += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            values.extend(str(item) for item in kinds)
        sql += " ORDER BY artifact_id LIMIT ?"
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def claim_artifact_cleanup(self, artifact_id: int) -> dict[str, Any] | None:
        """Atomically reserve one unreferenced ready Artifact for deletion."""
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE artifacts SET status='cleaning', updated_at=?
                   WHERE artifact_id=? AND status='ready' AND ref_count=0""",
                (_now(), int(artifact_id)),
            ).rowcount
            if changed != 1:
                return None
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=?", (int(artifact_id),)
                ).fetchone()
            )

    def finish_artifact_cleanup(
        self,
        artifact_id: int,
        *,
        success: bool,
    ) -> bool:
        """Commit cleanup or return the claimed Artifact to ready state."""
        next_status = "cleaned" if success else "ready"
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status=?, updated_at=?
                   WHERE artifact_id=? AND status='cleaning' AND ref_count=0""",
                (next_status, _now(), int(artifact_id)),
            ).rowcount == 1

    def artifact_cleanup_summary(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS artifact_count,
                          COALESCE(SUM(byte_count), 0) AS cleaned_bytes
                   FROM artifacts WHERE run_id=? AND status='cleaned'""",
                (str(run_id),),
            ).fetchone()
        return {
            "artifact_count": int(row["artifact_count"]),
            "cleaned_bytes": int(row["cleaned_bytes"]),
        }

    def artifacts_for_stream(
        self,
        run_id: str,
        stream_id: str,
        *,
        kind: str | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts WHERE run_id=? AND stream_id=?"
        values: list[Any] = [str(run_id), str(stream_id)]
        if kind is not None:
            sql += " AND kind=?"
            values.append(str(kind))
        if status is not None:
            sql += " AND status=?"
            values.append(str(status))
        sql += " ORDER BY unit_id, artifact_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def upsert_unit_report_summary(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        report: Mapping[str, Any],
        *,
        fitted_edge_count: int = 0,
    ) -> None:
        """Persist scalar report evidence after its JSON Artifact is ready."""
        artifact = self.artifact_for_stream_unit(
            run_id,
            stream_id,
            unit_id,
            "unit_boundary_report",
        )
        if artifact is None:
            raise RunStateError(
                "unit report summary requires a ready unit_boundary_report Artifact"
            )
        report_path = Path(str(artifact["path"]))
        if not report_path.is_file():
            raise RunStateError(f"unit report Artifact is missing: {report_path}")
        stat = report_path.stat()
        if int(artifact["byte_count"]) != int(stat.st_size):
            raise RunStateError(f"unit report Artifact size changed: {report_path}")
        diagnostics = report.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            raise RunStateError("unit boundary report diagnostics must be a list")
        edge_count = int(fitted_edge_count)
        if edge_count < 0 or edge_count > len(diagnostics):
            raise RunStateError(
                "unit fitted edge count is outside the diagnostic report range"
            )
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO unit_report_summaries
                   (run_id, stream_id, unit_id, status, fit_version,
                    chain_count, shared_chain_count, spline_count,
                    unchanged_count, skipped_invalid_count,
                    max_displacement_px, diagnostic_count, fitted_edge_count,
                    report_path, report_byte_count, report_sha256,
                    report_mtime_ns, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, stream_id, unit_id) DO UPDATE SET
                     status=excluded.status,
                     fit_version=excluded.fit_version,
                     chain_count=excluded.chain_count,
                     shared_chain_count=excluded.shared_chain_count,
                     spline_count=excluded.spline_count,
                     unchanged_count=excluded.unchanged_count,
                     skipped_invalid_count=excluded.skipped_invalid_count,
                     max_displacement_px=excluded.max_displacement_px,
                     diagnostic_count=excluded.diagnostic_count,
                     fitted_edge_count=excluded.fitted_edge_count,
                     report_path=excluded.report_path,
                     report_byte_count=excluded.report_byte_count,
                     report_sha256=excluded.report_sha256,
                     report_mtime_ns=excluded.report_mtime_ns,
                     updated_at=excluded.updated_at""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(report.get("status") or ""),
                    str(report.get("fit_version") or ""),
                    int(report.get("chain_count", 0)),
                    int(report.get("shared_chain_count", 0)),
                    int(report.get("spline_count", 0)),
                    int(report.get("unchanged_count", 0)),
                    int(report.get("skipped_invalid_count", 0)),
                    float(report.get("max_displacement_px", 0.0)),
                    len(diagnostics),
                    edge_count,
                    str(report_path.resolve()),
                    int(stat.st_size),
                    str(artifact["sha256"]),
                    int(stat.st_mtime_ns),
                    now,
                    now,
                ),
            )

    def unit_report_summaries(
        self,
        run_id: str,
        stream_id: str,
    ) -> list[dict[str, Any]]:
        """Return summary rows, or no rows when the current contract is incomplete."""
        with self._connect() as connection:
            exists = connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type='table' AND name='unit_report_summaries'"""
            ).fetchone()
            if exists is None:
                return []
            return [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM unit_report_summaries
                       WHERE run_id=? AND stream_id=? ORDER BY unit_id""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            ]

    def unit_report_summary_aggregate(
        self,
        run_id: str,
        stream_id: str,
    ) -> dict[str, Any]:
        """Aggregate report scalars inside SQLite without loading report JSON."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS unit_count,
                          COALESCE(SUM(chain_count), 0) AS chain_count,
                          COALESCE(SUM(shared_chain_count), 0)
                            AS shared_chain_count,
                          COALESCE(SUM(spline_count), 0) AS spline_count,
                          COALESCE(SUM(unchanged_count), 0) AS unchanged_count,
                          COALESCE(SUM(skipped_invalid_count), 0)
                            AS skipped_invalid_count,
                          COALESCE(SUM(CASE WHEN status='passed' THEN 0 ELSE 1 END), 0)
                            AS failed_unit_count,
                          COALESCE(MAX(max_displacement_px), 0)
                            AS max_displacement_px,
                          COALESCE(SUM(diagnostic_count), 0)
                            AS diagnostic_count,
                          COALESCE(SUM(fitted_edge_count), 0)
                            AS fitted_edge_count
                   FROM unit_report_summaries
                   WHERE run_id=? AND stream_id=?""",
                (str(run_id), str(stream_id)),
            ).fetchone()
        return dict(row)

    def object_ids_for_parts(
        self,
        run_id: str,
        stream_id: str,
        part_ids: Sequence[str],
    ) -> dict[str, str]:
        """Resolve a bounded part batch using one connection, not one per feature."""
        values = [str(part_id) for part_id in part_ids]
        if not values:
            return {}
        result: dict[str, str] = {}
        with self._connect() as connection:
            for offset in range(0, len(values), 400):
                batch = values[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""SELECT part_id, object_id FROM object_nodes
                        WHERE run_id=? AND stream_id=?
                          AND part_id IN ({placeholders})""",
                    (str(run_id), str(stream_id), *batch),
                ).fetchall()
                result.update(
                    {
                        str(row["part_id"]): str(row["object_id"])
                        for row in rows
                        if row["object_id"]
                    }
                )
        missing = [part_id for part_id in values if part_id not in result]
        if missing:
            raise RunStateError(
                f"object components are unresolved: {missing[:3]}"
            )
        return result

    def register_object_parts(
        self,
        run_id: str,
        stream_id: str,
        parts: Iterable[Mapping[str, Any]],
    ) -> int:
        now = _now()
        rows = [
            (
                str(run_id),
                str(stream_id),
                str(item["part_id"]),
                int(item["class_code"]),
                str(item["unit_id"]),
                str(item["part_id"]),
                now,
                now,
            )
            for item in parts
        ]
        with self.transaction() as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO object_nodes
                   (run_id, stream_id, part_id, class_code, unit_id,
                    parent_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return len(rows)

    def add_object_link(
        self,
        run_id: str,
        stream_id: str,
        left_part_id: str,
        right_part_id: str,
        class_code: int,
    ) -> bool:
        left, right = sorted((str(left_part_id), str(right_part_id)))
        if left == right:
            return False
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT part_id, class_code FROM object_nodes
                   WHERE run_id=? AND stream_id=? AND part_id IN (?, ?)""",
                (str(run_id), str(stream_id), left, right),
            ).fetchall()
            if len(rows) != 2 or any(int(row["class_code"]) != int(class_code) for row in rows):
                raise RunStateError("object link parts are missing or have different classes")
            cursor = connection.execute(
                """INSERT OR IGNORE INTO object_links
                   (run_id, stream_id, left_part_id, right_part_id, class_code, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(run_id), str(stream_id), left, right, int(class_code), _now()),
            )
            return cursor.rowcount == 1

    def object_link_count(self, run_id: str, stream_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM object_links
                       WHERE run_id=? AND stream_id=?""",
                    (str(run_id), str(stream_id)),
                ).fetchone()[0]
            )

    def resolve_object_components(self, run_id: str, stream_id: str) -> int:
        """Resolve object links with a disk-backed union-find and deterministic IDs."""
        import hashlib

        with self.transaction() as connection:
            def find(part_id: str) -> str:
                path = []
                current = part_id
                while True:
                    row = connection.execute(
                        """SELECT parent_id FROM object_nodes
                           WHERE run_id=? AND stream_id=? AND part_id=?""",
                        (str(run_id), str(stream_id), current),
                    ).fetchone()
                    if row is None:
                        raise RunStateError(f"object node is missing: {current}")
                    parent = str(row["parent_id"])
                    if parent == current:
                        root = current
                        break
                    path.append(current)
                    current = parent
                for child in path:
                    connection.execute(
                        """UPDATE object_nodes SET parent_id=?, updated_at=?
                           WHERE run_id=? AND stream_id=? AND part_id=?""",
                        (root, _now(), str(run_id), str(stream_id), child),
                    )
                return root

            links = connection.execute(
                """SELECT left_part_id, right_part_id FROM object_links
                   WHERE run_id=? AND stream_id=? ORDER BY left_part_id, right_part_id""",
                (str(run_id), str(stream_id)),
            )
            for link in links:
                left_root = find(str(link["left_part_id"]))
                right_root = find(str(link["right_part_id"]))
                if left_root == right_root:
                    continue
                left_rank = int(
                    connection.execute(
                        """SELECT rank_value FROM object_nodes
                           WHERE run_id=? AND stream_id=? AND part_id=?""",
                        (str(run_id), str(stream_id), left_root),
                    ).fetchone()[0]
                )
                right_rank = int(
                    connection.execute(
                        """SELECT rank_value FROM object_nodes
                           WHERE run_id=? AND stream_id=? AND part_id=?""",
                        (str(run_id), str(stream_id), right_root),
                    ).fetchone()[0]
                )
                if left_rank < right_rank or (left_rank == right_rank and left_root > right_root):
                    left_root, right_root = right_root, left_root
                    left_rank, right_rank = right_rank, left_rank
                connection.execute(
                    """UPDATE object_nodes SET parent_id=?, updated_at=?
                       WHERE run_id=? AND stream_id=? AND part_id=?""",
                    (left_root, _now(), str(run_id), str(stream_id), right_root),
                )
                if left_rank == right_rank:
                    connection.execute(
                        """UPDATE object_nodes SET rank_value=rank_value+1, updated_at=?
                           WHERE run_id=? AND stream_id=? AND part_id=?""",
                        (_now(), str(run_id), str(stream_id), left_root),
                    )

            part_ids = [
                str(row["part_id"])
                for row in connection.execute(
                    """SELECT part_id FROM object_nodes
                       WHERE run_id=? AND stream_id=? ORDER BY part_id""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            ]
            roots = {part_id: find(part_id) for part_id in part_ids}
            for part_id, root in roots.items():
                digest = hashlib.sha1(
                    f"{run_id}|{stream_id}|{root}".encode("utf-8")
                ).hexdigest()[:24]
                connection.execute(
                    """UPDATE object_nodes SET object_id=?, updated_at=?
                       WHERE run_id=? AND stream_id=? AND part_id=?""",
                    (f"obj_{digest}", _now(), str(run_id), str(stream_id), part_id),
                )
            return len(set(roots.values()))

    def object_id_for_part(
        self, run_id: str, stream_id: str, part_id: str
    ) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT object_id FROM object_nodes
                   WHERE run_id=? AND stream_id=? AND part_id=?""",
                (str(run_id), str(stream_id), str(part_id)),
            ).fetchone()
        if row is None or not row["object_id"]:
            raise RunStateError(f"object component is unresolved: {part_id}")
        return str(row["object_id"])

    def artifact_for_stream_unit(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        kind: str,
        *,
        status: str = "ready",
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM artifacts WHERE run_id=? AND stream_id=?
                       AND unit_id=? AND kind=? AND status=?""",
                    (str(run_id), str(stream_id), str(unit_id), str(kind), str(status)),
                ).fetchone()
            )

    def append_event(
        self,
        run_id: str,
        event_type: str,
        *,
        level: str = "info",
        stream_id: str = "",
        job_id: int | None = None,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO events
                   (run_id, timestamp, level, event_type, stream_id, job_id,
                    message, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(run_id),
                    _now(),
                    str(level),
                    str(event_type),
                    str(stream_id),
                    job_id,
                    str(message),
                    _json(dict(payload or {})),
                ),
            )
            return int(cursor.lastrowid)

    def job_counts(self, run_id: str, *, stream_id: str = "") -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=?"
        values: list[Any] = [str(run_id)]
        if stream_id:
            sql += " AND stream_id=?"
            values.append(str(stream_id))
        sql += " GROUP BY status"
        with self._connect() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(sql, values).fetchall()
            }
