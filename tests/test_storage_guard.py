from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from storage_guard import StorageGuard, StorageReserveError


def test_storage_guard_enforces_reserve_and_managed_high_water(tmp_path):
    usage = SimpleNamespace(total=1_000, used=400, free=600)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=100,
        managed_budget_bytes=300,
        initial_managed_bytes=100,
        remaining_permanent_bytes=lambda: 200,
        disk_usage=lambda _path: usage,
    )

    report = guard.check("checkpoint", write_bytes=100)
    assert report["required_free_bytes"] == 400
    guard.committed(100)
    assert guard.managed_bytes == 200
    assert guard.peak_managed_bytes == 200

    with pytest.raises(StorageReserveError, match="managed write cache") as budget:
        guard.check("cache", write_bytes=101)
    assert budget.value.reason == "managed_budget"
    assert budget.value.transient is False

    usage.free = 350
    with pytest.raises(StorageReserveError, match="managed write raster") as reserve:
        guard.check(
            "raster",
            write_bytes=25,
            additional_reserve_bytes=50,
            managed_growth_bytes=0,
        )
    assert reserve.value.reason == "filesystem_reserve"
    assert reserve.value.transient is True

    guard.released(75)
    assert guard.managed_bytes == 125
    guard.adjust(25)
    guard.adjust(-10)
    assert guard.managed_bytes == 140


def test_storage_guard_reserve_is_recomputed_without_expanding_budget(tmp_path):
    permanent = {"value": 300}
    usage = SimpleNamespace(total=2_000, used=1_000, free=1_000)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=200,
        managed_budget_bytes=500,
        remaining_permanent_bytes=lambda: permanent["value"],
        disk_usage=lambda _path: usage,
    )
    assert guard.check("first", write_bytes=100)["required_free_bytes"] == 600
    permanent["value"] = 700
    with pytest.raises(StorageReserveError):
        guard.check("second", write_bytes=101)


def test_concurrent_write_reservation_is_counted_before_commit(tmp_path):
    usage = SimpleNamespace(total=10_000, used=0, free=10_000)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        managed_budget_bytes=150,
        disk_usage=lambda _path: usage,
    )
    first = guard.check(
        "tile-a",
        write_bytes=100,
        managed_growth_bytes=100,
        reserve_managed_growth=True,
    )
    assert first["reserved_growth_bytes"] == 100
    assert first["reserved_write_bytes"] == 100
    assert guard.managed_bytes == 100
    assert guard.pending_write_bytes == 100
    with pytest.raises(StorageReserveError) as second:
        guard.check(
            "tile-b",
            write_bytes=60,
            managed_growth_bytes=60,
            reserve_managed_growth=True,
        )
    assert second.value.reason == "managed_budget"
    guard.adjust(
        42 - first["reserved_growth_bytes"],
        settled_write_bytes=first["reserved_write_bytes"],
    )
    assert guard.managed_bytes == 42
    assert guard.pending_write_bytes == 0


def test_pending_concurrent_write_is_included_in_filesystem_reserve(tmp_path):
    usage = SimpleNamespace(total=1_000, used=850, free=150)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=100,
        managed_budget_bytes=1_000,
        disk_usage=lambda _path: usage,
    )

    first_reserved = threading.Event()
    release_failed_first = threading.Event()

    def failed_first_writer():
        reservation = guard.check(
            "tile-a",
            write_bytes=40,
            managed_growth_bytes=40,
            reserve_managed_growth=True,
        )
        first_reserved.set()
        assert release_failed_first.wait(timeout=2)
        guard.adjust(
            -reservation["reserved_growth_bytes"],
            settled_write_bytes=reservation["reserved_write_bytes"],
        )
        return reservation

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(failed_first_writer)
        assert first_reserved.wait(timeout=2)
        try:
            with pytest.raises(StorageReserveError) as second:
                guard.check(
                    "tile-b",
                    write_bytes=40,
                    managed_growth_bytes=40,
                    reserve_managed_growth=True,
                )
            assert second.value.reason == "filesystem_reserve"
            assert second.value.transient is True
            assert guard.pending_write_bytes == 40
            assert guard.managed_bytes == 40
        finally:
            release_failed_first.set()
        first = first_future.result(timeout=2)

    assert first["required_free_bytes"] == 140
    assert guard.pending_write_bytes == 0
    assert guard.managed_bytes == 0
    retry = guard.check(
        "tile-b-retry",
        write_bytes=40,
        managed_growth_bytes=40,
        reserve_managed_growth=True,
    )
    assert retry["reserved_write_bytes"] == 40
    guard.adjust(
        -retry["reserved_growth_bytes"],
        settled_write_bytes=retry["reserved_write_bytes"],
    )


def test_sequential_write_settlement_reconciles_actual_managed_bytes(tmp_path):
    usage = SimpleNamespace(total=1_000, used=800, free=200)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=100,
        managed_budget_bytes=1_000,
        initial_managed_bytes=10,
        disk_usage=lambda _path: usage,
    )

    first = guard.check(
        "first",
        write_bytes=60,
        managed_growth_bytes=60,
        reserve_managed_growth=True,
    )
    guard.adjust(
        45 - first["reserved_growth_bytes"],
        settled_write_bytes=first["reserved_write_bytes"],
    )
    assert guard.pending_write_bytes == 0
    assert guard.managed_bytes == 55

    usage.free = 155
    second = guard.check(
        "second",
        write_bytes=55,
        managed_growth_bytes=55,
        reserve_managed_growth=True,
    )
    assert second["required_free_bytes"] == 155
    guard.adjust(
        -second["reserved_growth_bytes"],
        settled_write_bytes=second["reserved_write_bytes"],
    )
    assert guard.pending_write_bytes == 0
    assert guard.managed_bytes == 55


def test_settlement_cannot_release_an_unowned_pending_write(tmp_path):
    guard = StorageGuard(tmp_path, min_free_bytes=0)

    with pytest.raises(ValueError, match="exceed pending reservations"):
        guard.adjust(0, settled_write_bytes=1)
