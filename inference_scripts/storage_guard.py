"""Runtime storage fencing for bounded Work Package writes.

The preflight chooses a frozen cache budget and a filesystem reserve.  This
module enforces those values again immediately before every large atomic write;
it does not try to recover space by deleting user data.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Mapping


class StorageReserveError(RuntimeError):
    """Raised before a managed write would cross a frozen storage boundary."""

    def __init__(
        self,
        operation: str,
        *,
        free_bytes: int,
        required_free_bytes: int,
        write_bytes: int,
        managed_bytes: int,
        managed_budget_bytes: int,
    ) -> None:
        self.operation = str(operation)
        self.free_bytes = int(free_bytes)
        self.required_free_bytes = int(required_free_bytes)
        self.write_bytes = int(write_bytes)
        self.managed_bytes = int(managed_bytes)
        self.managed_budget_bytes = int(managed_budget_bytes)
        budget_exceeded = (
            self.managed_budget_bytes > 0
            and self.managed_bytes > self.managed_budget_bytes
        )
        self.reason = (
            "managed_budget" if budget_exceeded else "filesystem_reserve"
        )
        self.transient = (
            not budget_exceeded and self.free_bytes < self.required_free_bytes
        )
        super().__init__(
            "insufficient disk for managed write "
            f"{self.operation}: free={self.free_bytes}, "
            f"required_free={self.required_free_bytes}, "
            f"write={self.write_bytes}, managed={self.managed_bytes}, "
            f"managed_budget={self.managed_budget_bytes}"
        )


class StorageReservation:
    """One write reservation that can be reconciled exactly once.

    The guard charges the projected managed growth and physical write before
    the caller starts mutating the filesystem.  ``settle`` releases the
    physical-write reservation and replaces the projected growth with the
    actual byte delta left on disk.  Keeping this state in a one-shot object
    prevents an exception path from accidentally settling the same reservation
    twice while another writer is active.
    """

    def __init__(
        self,
        guard: "StorageGuard",
        *,
        operation: str,
        reserved_write_bytes: int,
        reserved_growth_bytes: int,
    ) -> None:
        self._guard = guard
        self.operation = str(operation)
        self.reserved_write_bytes = max(0, int(reserved_write_bytes))
        self.reserved_growth_bytes = max(0, int(reserved_growth_bytes))
        self._settled = False
        self._lock = threading.Lock()

    @property
    def settled(self) -> bool:
        with self._lock:
            return self._settled

    def settle(self, actual_managed_growth_bytes: int = 0) -> int:
        """Replace projected growth with the actual delta left on disk."""

        with self._lock:
            if self._settled:
                raise RuntimeError(
                    f"storage reservation already settled: {self.operation}"
                )
            managed_bytes = self._guard.adjust(
                int(actual_managed_growth_bytes) - self.reserved_growth_bytes,
                settled_write_bytes=self.reserved_write_bytes,
            )
            self._settled = True
            return managed_bytes

    def release(self) -> int:
        """Release a failed write that left no managed bytes on disk."""

        return self.settle(0)


class StorageGuard:
    """Thread-safe runtime guard with an incremental managed-byte ledger."""

    def __init__(
        self,
        root: str | Path,
        *,
        min_free_bytes: int,
        managed_budget_bytes: int = 0,
        initial_managed_bytes: int = 0,
        remaining_permanent_bytes: Callable[[], int] | None = None,
        disk_usage: Callable[[str | Path], object] = shutil.disk_usage,
    ) -> None:
        self.root = Path(root).resolve()
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.managed_budget_bytes = max(0, int(managed_budget_bytes))
        self._managed_bytes = max(0, int(initial_managed_bytes))
        self._peak_managed_bytes = self._managed_bytes
        self._pending_write_bytes = 0
        self._remaining_permanent_bytes = remaining_permanent_bytes or (lambda: 0)
        self._disk_usage = disk_usage
        self._lock = threading.Lock()

    @property
    def managed_bytes(self) -> int:
        with self._lock:
            return self._managed_bytes

    @property
    def peak_managed_bytes(self) -> int:
        with self._lock:
            return self._peak_managed_bytes

    @property
    def pending_write_bytes(self) -> int:
        with self._lock:
            return self._pending_write_bytes

    def check(
        self,
        operation: str,
        *,
        write_bytes: int = 0,
        additional_reserve_bytes: int = 0,
        managed_growth_bytes: int | None = None,
        reserve_managed_growth: bool = False,
    ) -> dict[str, int]:
        """Reject a write before it crosses disk reserve or cache high-water."""

        write_size = max(0, int(write_bytes))
        growth = write_size if managed_growth_bytes is None else max(
            0, int(managed_growth_bytes)
        )
        with self._lock:
            permanent = max(0, int(self._remaining_permanent_bytes()))
            extra = max(0, int(additional_reserve_bytes))
            usage = self._disk_usage(self.root)
            free_bytes = int(getattr(usage, "free"))
            pending_before = self._pending_write_bytes
            required_free = (
                self.min_free_bytes
                + permanent
                + extra
                + pending_before
                + write_size
            )
            next_managed = self._managed_bytes + growth
            if (
                free_bytes < required_free
                or (
                    self.managed_budget_bytes > 0
                    and next_managed > self.managed_budget_bytes
                )
            ):
                raise StorageReserveError(
                    operation,
                    free_bytes=free_bytes,
                    required_free_bytes=required_free,
                    write_bytes=write_size,
                    managed_bytes=next_managed,
                    managed_budget_bytes=self.managed_budget_bytes,
                )
            if reserve_managed_growth:
                self._managed_bytes = next_managed
                self._peak_managed_bytes = max(
                    self._peak_managed_bytes, self._managed_bytes
                )
                self._pending_write_bytes += write_size
            pending_after = self._pending_write_bytes
        return {
            "free_bytes": free_bytes,
            "required_free_bytes": required_free,
            "write_bytes": write_size,
            "managed_growth_bytes": growth,
            "remaining_permanent_bytes": permanent,
            "reserved_growth_bytes": growth if reserve_managed_growth else 0,
            "reserved_write_bytes": write_size if reserve_managed_growth else 0,
            "pending_write_bytes": pending_after,
        }

    def reserve(
        self,
        operation: str,
        *,
        write_bytes: int = 0,
        additional_reserve_bytes: int = 0,
        managed_growth_bytes: int | None = None,
    ) -> StorageReservation:
        """Atomically reserve one physical write and its managed growth.

        This deliberately delegates to ``check`` so existing instrumentation
        and fault injection observe the same decision boundary.  ``check``
        performs the approval and ledger mutation under one lock; constructing
        the one-shot handle afterwards does not open a second check/write race.
        """

        report = self.check(
            operation,
            write_bytes=write_bytes,
            additional_reserve_bytes=additional_reserve_bytes,
            managed_growth_bytes=managed_growth_bytes,
            reserve_managed_growth=True,
        )
        return StorageReservation(
            self,
            operation=operation,
            reserved_write_bytes=report["reserved_write_bytes"],
            reserved_growth_bytes=report["reserved_growth_bytes"],
        )

    def committed(self, byte_count: int) -> int:
        with self._lock:
            self._managed_bytes += max(0, int(byte_count))
            self._peak_managed_bytes = max(
                self._peak_managed_bytes, self._managed_bytes
            )
            return self._managed_bytes

    def adjust(
        self,
        byte_delta: int,
        *,
        settled_write_bytes: int = 0,
    ) -> int:
        """Reconcile managed bytes and settle an approved physical write.

        ``reserve_managed_growth=True`` accounts both the projected managed
        growth and the not-yet-landed physical write.  The caller must settle
        that physical reservation after either success or failure.  The byte
        delta then converts the projected managed growth to the bytes that
        actually remain on disk.
        """

        value = int(byte_delta)
        settled = int(settled_write_bytes)
        if settled < 0:
            raise ValueError("settled write bytes cannot be negative")
        with self._lock:
            if settled > self._pending_write_bytes:
                raise ValueError(
                    "settled write bytes exceed pending reservations: "
                    f"settled={settled}, pending={self._pending_write_bytes}"
                )
            self._pending_write_bytes -= settled
            self._managed_bytes = max(0, self._managed_bytes + value)
            self._peak_managed_bytes = max(
                self._peak_managed_bytes, self._managed_bytes
            )
            return self._managed_bytes

    def released(self, byte_count: int) -> int:
        with self._lock:
            self._managed_bytes = max(
                0, self._managed_bytes - max(0, int(byte_count))
            )
            return self._managed_bytes


def exact_remaining_permanent_bytes(
    spec: Mapping[str, Any],
    database: Any,
) -> int:
    """Return exact unwritten Core raster bytes plus non-decaying reserve.

    Edge Partitions can have much smaller Core windows than interior
    Partitions.  Counting ready Artifact rows and subtracting an average byte
    amount can therefore release storage that a later large Core still needs.
    Reconstruct the same per-Core byte ledger frozen by the schema-2 preflight
    and subtract only the exact ready mask/confidence entries.
    """

    storage = dict(spec.get("storage_preflight") or {})
    if int(storage.get("storage_tuning_schema_version") or 0) < 2:
        return 0
    raster_bytes = int(
        storage.get("estimated_permanent_raster_bytes")
        or storage.get("estimated_permanent_bytes")
        or 0
    )
    nondecaying_bytes = int(
        storage.get("nondecaying_permanent_reserve_bytes")
        if "nondecaying_permanent_reserve_bytes" in storage
        else storage.get("permanent_uncertainty_bytes")
        or 0
    )
    if raster_bytes <= 0:
        return max(0, nondecaying_bytes)

    run_id = str(spec["run_id"])
    stream_ids = [
        str(item.get("stream_id") or "")
        for item in spec.get("streams") or []
        if str(item.get("stream_id") or "")
    ]
    if not stream_ids:
        raise ValueError("schema-2 permanent raster reserve has no result streams")
    bytes_by_key: dict[tuple[str, str, str], int] = {}
    for partition in database.partitions_for_run(run_id):
        core = partition.get("core_window") or {}
        try:
            width = int(core["x1"]) - int(core["x0"])
            height = int(core["y1"]) - int(core["y0"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Partition Core window is incomplete or invalid") from error
        if width < 1 or height < 1:
            raise ValueError("Partition Core window must have positive area")
        core_pixels = width * height
        partition_id = str(partition["partition_id"])
        for stream_id in stream_ids:
            bytes_by_key[(stream_id, partition_id, "core_mask")] = (
                core_pixels * 2
            )
            bytes_by_key[(stream_id, partition_id, "core_confidence")] = (
                core_pixels * 4
            )
    if sum(bytes_by_key.values()) != raster_bytes:
        raise ValueError(
            "frozen permanent raster reserve does not match exact Partition Core windows"
        )

    ready_keys = {
        (stream_id, str(artifact["unit_id"]), kind)
        for stream_id in stream_ids
        for kind in ("core_mask", "core_confidence")
        for artifact in database.artifacts_for_stream(
            run_id,
            stream_id,
            kind=kind,
            status="ready",
        )
    }
    remaining_raster = sum(
        byte_count
        for key, byte_count in bytes_by_key.items()
        if key not in ready_keys
    )
    return max(0, remaining_raster) + max(0, nondecaying_bytes)
