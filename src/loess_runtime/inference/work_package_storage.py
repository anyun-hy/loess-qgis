"""Work Package storage budget and permanent-output accounting."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from loess_runtime.system.runtime_errors import WorkPackageRuntimeError
from loess_runtime.system.runtime_metrics import directory_size
from loess_runtime.system.storage_guard import (
    StorageGuard,
    StorageReservation,
    managed_temporary_budget_bytes,
)

PermanentArtifactKey = tuple[str, str, str]


class _WorkingCacheGuard(StorageGuard):
    """Charge working writes to both the package and total temporary ledgers.

    A separate lock keeps each paired reservation/settlement indivisible with
    respect to other working writers. Shared-input writers only use the total
    guard, so neither lock order nor accounting depends on their cleanup pace.
    """

    def __init__(
        self,
        total: WorkPackageStorageBudget,
        roots: Iterable[str | Path],
        budget_bytes: int,
    ) -> None:
        self._total = total
        self._pair_lock = threading.Lock()
        super().__init__(
            total.root,
            min_free_bytes=0,
            managed_budget_bytes=budget_bytes,
            initial_managed_bytes=sum(directory_size(Path(path)) for path in roots),
        )

    def check(self, operation: str, **kwargs: Any) -> dict[str, int]:
        if self._total._lease_guard is not None:
            self._total._lease_guard()
        with self._pair_lock:
            report = self._total.check(operation, **kwargs)
            try:
                super().check(operation, **kwargs)
            except Exception:
                self._total.adjust(
                    -report["reserved_growth_bytes"],
                    settled_write_bytes=report["reserved_write_bytes"],
                )
                raise
            return report

    def adjust(self, byte_delta: int, *, settled_write_bytes: int = 0) -> int:
        with self._pair_lock:
            value = super().adjust(byte_delta, settled_write_bytes=settled_write_bytes)
            self._total.adjust(byte_delta, settled_write_bytes=settled_write_bytes)
            return value

    def released(self, byte_count: int) -> int:
        with self._pair_lock:
            value = super().released(byte_count)
            self._total.released(byte_count)
            return value

    def committed(self, byte_count: int) -> int:
        with self._pair_lock:
            value = super().committed(byte_count)
            self._total.committed(byte_count)
            return value


class WorkPackageStorageBudget(StorageGuard):
    """Own the frozen cache budget and decaying permanent raster reserve.

    Partition and ready-artifact inputs may be lazy.  They are consumed only
    when the frozen permanent estimate is positive, and ready artifacts are
    not read until every Partition Core has passed exact-total validation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        managed_roots: Iterable[str | Path],
        storage_preflight: Mapping[str, Any],
        fallback_min_free_disk_gb: float,
        stream_ids: Iterable[str],
        partitions: Iterable[Mapping[str, Any]],
        ready_permanent_keys: Iterable[PermanentArtifactKey],
        remaining_deferred_bytes: Callable[[], int],
        working_roots: Iterable[str | Path] = (),
        lease_guard: Callable[[], None] | None = None,
    ) -> None:
        storage = dict(storage_preflight)
        self.storage_schema = int(storage.get("storage_tuning_schema_version") or 0)
        self.permanent_estimated_bytes = (
            int(storage.get("estimated_permanent_bytes") or 0)
            if self.storage_schema >= 2
            else 0
        )
        self.permanent_uncertainty_bytes = (
            int(storage.get("permanent_uncertainty_bytes") or 0)
            if self.storage_schema >= 2
            else 0
        )
        managed_budget_bytes = managed_temporary_budget_bytes(storage)
        min_free_bytes = int(
            storage.get("effective_min_free_disk_bytes")
            or float(fallback_min_free_disk_gb) * 1024**3
        )

        self._lease_guard = lease_guard
        self._remaining_deferred_bytes = remaining_deferred_bytes
        self._ready_lock = threading.Lock()
        self._ready_permanent_keys: set[PermanentArtifactKey] = set()
        self._permanent_bytes_by_key: dict[PermanentArtifactKey, int] = {}

        if self.permanent_estimated_bytes > 0:
            normalized_stream_ids = tuple(
                str(stream_id) for stream_id in stream_ids if str(stream_id)
            )
            for partition in partitions:
                core = partition["core_window"]
                core_pixels = (int(core["x1"]) - int(core["x0"])) * (
                    int(core["y1"]) - int(core["y0"])
                )
                if core_pixels < 1:
                    raise WorkPackageRuntimeError(
                        f"Partition Core has invalid area: {partition['partition_id']}"
                    )
                partition_id = str(partition["partition_id"])
                for stream_id in normalized_stream_ids:
                    self._permanent_bytes_by_key[
                        (stream_id, partition_id, "core_mask")
                    ] = core_pixels * 2
                    self._permanent_bytes_by_key[
                        (stream_id, partition_id, "core_confidence")
                    ] = core_pixels * 4
            if (
                sum(self._permanent_bytes_by_key.values())
                != self.permanent_estimated_bytes
            ):
                raise WorkPackageRuntimeError(
                    "frozen permanent raster reserve does not match exact "
                    "Partition Core windows"
                )
            self._ready_permanent_keys.update(
                (str(stream_id), str(partition_id), str(kind))
                for stream_id, partition_id, kind in ready_permanent_keys
            )

        super().__init__(
            root,
            min_free_bytes=min_free_bytes,
            managed_budget_bytes=managed_budget_bytes,
            initial_managed_bytes=sum(
                directory_size(Path(path)) for path in managed_roots
            ),
            remaining_permanent_bytes=self.remaining_permanent_bytes,
        )
        self.working_cache = _WorkingCacheGuard(
            self,
            working_roots,
            int(
                storage.get("working_cache_budget_bytes")
                or storage.get("resolved_score_cache_budget_bytes")
                or 0
            ),
        )

    def remaining_permanent_bytes(self) -> int:
        """Return the still-reserved permanent and deferred byte allowance."""

        deferred_remaining = max(0, int(self._remaining_deferred_bytes()))
        if self.permanent_estimated_bytes <= 0:
            return self.permanent_uncertainty_bytes + deferred_remaining
        with self._ready_lock:
            remaining = sum(
                byte_count
                for key, byte_count in self._permanent_bytes_by_key.items()
                if key not in self._ready_permanent_keys
            )
        return remaining + self.permanent_uncertainty_bytes + deferred_remaining

    def mark_permanent_ready(
        self, stream_id: str, partition_id: str, kind: str
    ) -> None:
        """Release one exact Core raster from future permanent reserve checks."""

        with self._ready_lock:
            self._ready_permanent_keys.add(
                (str(stream_id), str(partition_id), str(kind))
            )

    def reserve_write(
        self,
        operation: str,
        write_bytes: int = 0,
        *,
        managed_growth_bytes: int | None = None,
    ) -> StorageReservation:
        """Fence the lease, then reserve one atomic Work Package write."""

        if self._lease_guard is not None:
            self._lease_guard()
        return super().reserve(
            operation,
            write_bytes=max(0, int(write_bytes)),
            managed_growth_bytes=managed_growth_bytes,
        )

    def reserve_materialized_write(self, operation: str, write_bytes: int) -> int:
        """Reserve one concurrent tile-cache write using the existing callback API."""

        return int(
            self.working_cache.check(
                operation,
                write_bytes=max(0, int(write_bytes)),
                managed_growth_bytes=max(0, int(write_bytes)),
                reserve_managed_growth=True,
            )["reserved_growth_bytes"]
        )
