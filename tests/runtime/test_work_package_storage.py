from concurrent.futures import ThreadPoolExecutor

import pytest

from loess_runtime.inference.work_package_storage import WorkPackageStorageBudget
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError
from loess_runtime.system.storage_guard import StorageReserveError


def _budget(tmp_path, **overrides):
    arguments = {
        "managed_roots": (),
        "storage_preflight": {},
        "fallback_min_free_disk_gb": 0.0,
        "stream_ids": (),
        "partitions": (),
        "ready_permanent_keys": (),
        "remaining_deferred_bytes": lambda: 0,
    }
    arguments.update(overrides)
    return WorkPackageStorageBudget(tmp_path, **arguments)


def test_budget_skips_partition_and_ready_reads_without_permanent_estimate(
    tmp_path,
):
    def forbidden_rows():
        raise AssertionError("permanent rows must remain lazy")
        yield {}

    budget = _budget(
        tmp_path,
        storage_preflight={
            "storage_tuning_schema_version": 2,
            "estimated_permanent_bytes": 0,
            "permanent_uncertainty_bytes": 7,
        },
        partitions=forbidden_rows(),
        ready_permanent_keys=forbidden_rows(),
        remaining_deferred_bytes=lambda: 3,
    )

    assert budget.remaining_permanent_bytes() == 10


def test_budget_validates_exact_core_total_before_reading_ready_artifacts(
    tmp_path,
):
    consumed = []

    def partitions():
        consumed.append("partitions")
        yield {
            "partition_id": "edge",
            "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
        }

    def ready_keys():
        consumed.append("ready")
        yield "model:a", "edge", "core_mask"

    with pytest.raises(
        WorkPackageRuntimeError,
        match="frozen permanent raster reserve does not match",
    ):
        _budget(
            tmp_path,
            storage_preflight={
                "storage_tuning_schema_version": 2,
                "estimated_permanent_bytes": 7,
            },
            stream_ids=("model:a",),
            partitions=partitions(),
            ready_permanent_keys=ready_keys(),
        )

    assert consumed == ["partitions"]


def test_budget_releases_exact_edge_core_bytes_and_dynamic_deferred_reserve(
    tmp_path,
):
    deferred = {"bytes": 11}
    budget = _budget(
        tmp_path,
        storage_preflight={
            "storage_tuning_schema_version": 2,
            "estimated_permanent_bytes": 48,
            "permanent_uncertainty_bytes": 7,
        },
        stream_ids=("model:a",),
        partitions=(
            {
                "partition_id": "interior",
                "core_window": {"x0": 0, "y0": 0, "x1": 3, "y1": 2},
            },
            {
                "partition_id": "edge",
                "core_window": {"x0": 3, "y0": 0, "x1": 4, "y1": 2},
            },
        ),
        ready_permanent_keys=(("model:a", "edge", "core_mask"),),
        remaining_deferred_bytes=lambda: deferred["bytes"],
    )

    assert budget.remaining_permanent_bytes() == 62
    budget.mark_permanent_ready("model:a", "interior", "core_mask")
    assert budget.remaining_permanent_bytes() == 50
    budget.mark_permanent_ready("model:a", "interior", "core_mask")
    assert budget.remaining_permanent_bytes() == 50
    budget.mark_permanent_ready("model:a", "edge", "core_confidence")
    assert budget.remaining_permanent_bytes() == 42
    deferred["bytes"] = 1
    assert budget.remaining_permanent_bytes() == 32


@pytest.mark.parametrize("method_name", ("reserve_write", "reserve_materialized_write"))
def test_budget_fences_lease_before_charging_a_write(tmp_path, method_name):
    class LeaseFailure(RuntimeError):
        pass

    def reject_lease():
        raise LeaseFailure("lease lost")

    budget = _budget(tmp_path, lease_guard=reject_lease)

    with pytest.raises(LeaseFailure, match="lease lost"):
        getattr(budget, method_name)("write", 8)

    assert budget.managed_bytes == 0
    assert budget.pending_write_bytes == 0


def test_budget_preserves_atomic_and_materialized_write_settlement(tmp_path):
    lease_checks = []
    budget = _budget(tmp_path, lease_guard=lambda: lease_checks.append("checked"))

    reservation = budget.reserve_write("atomic", 10, managed_growth_bytes=10)
    assert budget.pending_write_bytes == 10
    assert reservation.settle(4) == 4
    assert budget.pending_write_bytes == 0

    reserved_growth = budget.reserve_materialized_write("tile", 5)
    assert reserved_growth == 5
    budget.working_cache.adjust(
        3 - reserved_growth, settled_write_bytes=reserved_growth
    )

    assert lease_checks == ["checked", "checked"]
    assert budget.managed_bytes == 7
    assert budget.pending_write_bytes == 0
    assert budget.working_cache.managed_bytes == 3
    assert budget.working_cache.pending_write_bytes == 0


def test_working_cache_is_separate_from_retained_inputs_and_both_are_charged(tmp_path):
    working = tmp_path / "working"
    retained = tmp_path / "retained"
    working.mkdir()
    retained.mkdir()
    (working / "tile").write_bytes(b"x" * 20)
    (retained / "probability").write_bytes(b"x" * 500)
    budget = _budget(
        tmp_path,
        managed_roots=(working, retained),
        working_roots=(working,),
        storage_preflight={
            "storage_tuning_schema_version": 2,
            "working_cache_budget_bytes": 100,
            "v33_managed_artifact_ceiling_bytes": 1_000,
            "deferred_temporary_reserve_bytes": 200,
        },
    )
    assert budget.managed_budget_bytes == 1_300
    assert budget.managed_bytes == 520
    assert budget.working_cache.managed_bytes == 20
    score = budget.working_cache.reserve("score", write_bytes=30)
    shared = budget.reserve_write("partition_probability", 200)
    assert budget.managed_bytes == 750
    assert budget.working_cache.managed_bytes == 50
    shared.settle(120)
    score.settle(25)
    budget.working_cache.released(20)
    assert budget.managed_bytes == 645
    assert budget.working_cache.managed_bytes == 25
    assert budget.working_cache.peak_managed_bytes == 50
    assert budget.peak_managed_bytes == 750
    assert budget.pending_write_bytes == budget.working_cache.pending_write_bytes == 0
    with pytest.raises(RuntimeError, match="already settled"):
        score.settle(25)
    assert budget.managed_bytes == 645


def test_concurrent_working_writes_cannot_spend_the_shared_allowance(tmp_path):
    import threading

    budget = _budget(
        tmp_path,
        storage_preflight={
            "storage_tuning_schema_version": 2,
            "working_cache_budget_bytes": 100,
            "v33_managed_artifact_ceiling_bytes": 1_000,
        },
    )
    barrier = threading.Barrier(2)

    def write():
        reservation = None
        try:
            reservation = budget.working_cache.reserve(
                "concurrent score", write_bytes=60
            )
        except StorageReserveError as error:
            assert error.reason == "managed_budget"
            assert error.managed_budget_bytes == 100
        barrier.wait(timeout=5)
        if reservation is not None:
            reservation.release()
        return reservation is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: write(), range(2)))
    assert sum(results) == 1
    assert budget.working_cache.peak_managed_bytes == 60
    assert budget.managed_bytes == budget.working_cache.managed_bytes == 0
    assert budget.pending_write_bytes == budget.working_cache.pending_write_bytes == 0


def test_working_write_still_checks_total_ceiling_and_disk_reserve(tmp_path):
    from types import SimpleNamespace

    budget = _budget(
        tmp_path,
        storage_preflight={"working_cache_budget_bytes": 100},
    )
    shared = budget.reserve_write("retained input", 90)
    with pytest.raises(StorageReserveError) as total:
        budget.working_cache.reserve("score", write_bytes=11)
    assert total.value.reason == "managed_budget"
    assert budget.managed_bytes == 90
    assert budget.working_cache.managed_bytes == 0
    shared.release()
    budget._disk_usage = lambda _: SimpleNamespace(free=5)
    with pytest.raises(StorageReserveError) as disk:
        budget.working_cache.reserve("score", write_bytes=6)
    assert disk.value.reason == "filesystem_reserve"
    assert budget.managed_bytes == budget.working_cache.managed_bytes == 0
    assert budget.pending_write_bytes == budget.working_cache.pending_write_bytes == 0


def test_ready_updates_and_base_checks_do_not_invert_locks(tmp_path):
    budget = _budget(
        tmp_path,
        storage_preflight={
            "storage_tuning_schema_version": 2,
            "estimated_permanent_bytes": 6,
        },
        stream_ids=("model:a",),
        partitions=(
            {
                "partition_id": "partition-a",
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            },
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        checks = executor.submit(
            lambda: [budget.check("probe", write_bytes=0) for _ in range(100)]
        )
        updates = executor.submit(
            lambda: [
                budget.mark_permanent_ready("model:a", "partition-a", "core_mask")
                for _ in range(100)
            ]
        )
        checks.result(timeout=2)
        updates.result(timeout=2)

    assert budget.remaining_permanent_bytes() == 4
