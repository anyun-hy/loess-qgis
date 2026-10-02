"""Validate frozen Run policy and produce its spatial control-plan inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from labeling_tool.runs.run_build_contract import (
    V3_POLICY_ID,
    V33_POLICY_ID,
    FrozenRunPlan,
    RunBuilderV5Error,
)
from labeling_tool.runs.spatial_planner import plan_spatial_units
from labeling_tool.shared.contracts.run_spec import CLASS_ORDER, sha256_file
from labeling_tool.shared.planning.work_package_planner import (
    plan_work_packages,
    unit_confidence_reserve,
)


def prepare_v5_run_plan(
    *,
    run_dir: Path,
    tile_rows: int,
    tile_cols: int,
    overlap: int,
    scaling: Mapping[str, Any],
    boundary_fitting: Mapping[str, Any],
    fragmentation_regularization: Mapping[str, Any] | None,
    range_selection: Mapping[str, Any] | None,
    storage_report: Mapping[str, Any],
    checkpoint: Callable[[float, str], None],
) -> FrozenRunPlan:
    """Validate policy and plan spatial/package data for a frozen Run.

    The vector-range digest is deliberately read here because it belongs to
    range freezing.  The caller supplies ``checkpoint`` so the 8% cancellation
    gate remains between spatial/package planning and V3.3 reserve validation.
    """

    scaling_value = dict(scaling)
    boundary_value = dict(boundary_fitting)
    if not isinstance(boundary_value.get("enabled"), bool):
        raise RunBuilderV5Error("boundary_fitting.enabled must be true or false")
    boundary_value.setdefault("mode", "divider_cubic_bspline_adaptive_v2")
    if str(boundary_value.get("mode") or "") != "divider_cubic_bspline_adaptive_v2":
        raise RunBuilderV5Error(
            "boundary_fitting.mode must equal divider_cubic_bspline_adaptive_v2"
        )

    fragmentation_value = dict(fragmentation_regularization or {})
    fragmentation_value.setdefault("enabled", True)
    fragmentation_value.setdefault("policy_id", V3_POLICY_ID)
    fragmentation_value.setdefault(
        "policy_version",
        (
            "v33_production_20260826"
            if fragmentation_value["policy_id"] == V33_POLICY_ID
            else "semantic_optimized_200_v3_core_bounded_v1"
        ),
    )
    fragmentation_value.setdefault("baseline_policy_id", V3_POLICY_ID)
    fragmentation_value.setdefault(
        "baseline_policy_version", "semantic_optimized_200_v3_core_bounded_v1"
    )
    fragmentation_value.setdefault("buffer_pixels", 256)
    fragmentation_value.setdefault("max_workers", 4)
    if not isinstance(fragmentation_value.get("enabled"), bool):
        raise RunBuilderV5Error(
            "fragmentation_regularization.enabled must be true or false"
        )
    if fragmentation_value.get("policy_id") not in {V3_POLICY_ID, V33_POLICY_ID}:
        raise RunBuilderV5Error("fragmentation_regularization.policy_id is unsupported")
    if fragmentation_value.get("baseline_policy_id") != V3_POLICY_ID:
        raise RunBuilderV5Error(
            "fragmentation_regularization.baseline_policy_id must equal " + V3_POLICY_ID
        )
    if int(fragmentation_value.get("buffer_pixels") or 0) != 256:
        raise RunBuilderV5Error(
            "fragmentation_regularization.buffer_pixels must equal 256"
        )
    if not 1 <= int(fragmentation_value.get("max_workers") or 0) <= 4:
        raise RunBuilderV5Error(
            "fragmentation_regularization.max_workers must be between 1 and 4"
        )
    v33_enabled = bool(
        fragmentation_value["enabled"]
        and fragmentation_value["policy_id"] == V33_POLICY_ID
    )
    if v33_enabled:
        if fragmentation_value.get("publication") != "authoritative_fusion_core":
            raise RunBuilderV5Error(
                "V3.3 publication must equal authoritative_fusion_core"
            )
        for key in ("policy_sha256", "executor_sha256"):
            digest = str(fragmentation_value.get(key) or "").lower()
            if len(digest) != 64:
                raise RunBuilderV5Error(
                    f"fragmentation_regularization.{key} is required for V3.3"
                )
            try:
                int(digest, 16)
            except ValueError as error:
                raise RunBuilderV5Error(
                    f"fragmentation_regularization.{key} is invalid"
                ) from error
            fragmentation_value[key] = digest
    if fragmentation_value["enabled"] and int(scaling_value["partition_halo_px"]) < int(
        fragmentation_value["buffer_pixels"]
    ):
        raise RunBuilderV5Error(
            "partition_halo_px must be at least "
            "fragmentation_regularization.buffer_pixels"
        )

    range_value = dict(range_selection or {})
    range_mode = str(range_value.get("mode") or "extent")
    if range_mode not in {"extent", "vector_tile_intersection"}:
        raise RunBuilderV5Error(f"unsupported range selection mode: {range_mode}")
    range_value["mode"] = range_mode
    if range_mode == "extent":
        range_value["clip_outputs"] = True
    if range_mode == "vector_tile_intersection":
        if range_value.get("clip_outputs") is not True:
            raise RunBuilderV5Error(
                "vector Tile selection must clip outputs to the exact vector boundary"
            )
        source_value = str(
            range_value.get("vector_source") or range_value.get("vector_path") or ""
        )
        snapshot_path = Path(source_value.split("|", 1)[0]).expanduser().resolve()
        try:
            snapshot_path.relative_to(run_dir)
        except ValueError as error:
            raise RunBuilderV5Error(
                "vector range source must be a run-local frozen snapshot"
            ) from error
        if not snapshot_path.is_file() or snapshot_path.suffix.lower() != ".gpkg":
            raise RunBuilderV5Error(
                "vector range snapshot is missing or is not a GeoPackage"
            )
        snapshot_sha256 = sha256_file(snapshot_path)
        supplied_sha256 = str(range_value.get("vector_sha256") or "")
        if supplied_sha256 and supplied_sha256 != snapshot_sha256:
            raise RunBuilderV5Error("vector range snapshot changed before run creation")
        range_value.update(
            {
                "vector_source": str(snapshot_path),
                "vector_path": str(snapshot_path),
                "vector_sha256": snapshot_sha256,
                "clip_outputs": True,
            }
        )
    grid_count = int(tile_rows) * int(tile_cols)
    selected_count = int(range_value.get("selected_tile_count", grid_count))
    excluded_count = int(range_value.get("excluded_tile_count", 0))
    # The caller's spatial preflight, Package plan, and storage reservation all
    # depend on this value. Do not enlarge it here, or the frozen plan would
    # disagree with its preflight.
    if (
        selected_count < 1
        or excluded_count < 0
        or selected_count + excluded_count != grid_count
    ):
        raise RunBuilderV5Error(
            "range Tile counts must be positive and cover the declared grid"
        )

    spatial_plan = plan_spatial_units(
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        tile_size=512,
        overlap=int(overlap),
        partition_tile_rows=int(scaling_value["partition_tile_rows"]),
        partition_tile_cols=int(scaling_value["partition_tile_cols"]),
        seam_band_px=int(scaling_value["seam_band_px"]),
        halo_px=int(scaling_value["partition_halo_px"]),
    )
    package_plan = plan_work_packages(
        spatial_plan,
        package_tile_limit=int(storage_report["package_tile_limit"]),
        estimated_bytes_per_tile=int(storage_report["working_bytes_per_tile"]),
    )
    checkpoint(8, "空间单元与 Work Package 规划完成")

    package_by_partition = package_plan["package_by_partition"]
    partitions = tuple(
        {
            **partition,
            "package_id": package_by_partition[partition["partition_id"]],
        }
        for partition in spatial_plan["partitions"]
    )
    storage_value = dict(storage_report)
    if v33_enabled:
        confidence_reserve = unit_confidence_reserve(spatial_plan)
        retained_input_bytes = sum(
            (
                (
                    int(partition["halo_window"]["x1"])
                    - int(partition["halo_window"]["x0"])
                )
                * (
                    int(partition["halo_window"]["y1"])
                    - int(partition["halo_window"]["y0"])
                )
                * len(CLASS_ORDER)
                * 2
                + (
                    int(partition["core_window"]["x1"])
                    - int(partition["core_window"]["x0"])
                )
                * (
                    int(partition["core_window"]["y1"])
                    - int(partition["core_window"]["y0"])
                )
                * 4
                + 3 * 64 * 1024
            )
            for partition in partitions
        )
        storage_value["v33_managed_artifact_ceiling_bytes"] = retained_input_bytes
        storage_value["v33_managed_artifact_ceiling_basis"] = (
            "maximum_all_probability_halos_uint16_plus_v3_context_and_"
            "baseline_cores_int16_not_an_admission_requirement"
        )
        if int(storage_value.get("storage_tuning_schema_version") or 0) >= 2:
            frozen_confidence_reserve = int(
                storage_value.get("deferred_temporary_reserve_bytes") or 0
            )
            if frozen_confidence_reserve != confidence_reserve["reserve_bytes"]:
                raise RunBuilderV5Error(
                    "V3.3 Unit confidence reserve does not match spatial plan"
                )
        storage_value.update(
            {
                "v33_storage_mode": "streamed_unit_confidence_v1",
                "v33_unit_confidence_budget_bytes": confidence_reserve["reserve_bytes"],
                "v33_unit_confidence_payload_bytes": confidence_reserve[
                    "payload_bytes"
                ],
                "v33_unit_confidence_file_overhead_bytes": confidence_reserve[
                    "file_overhead_bytes"
                ],
                "v33_unit_confidence_unit_count": confidence_reserve["unit_count"],
                "v33_retained_input_admission_mode": (
                    "runtime_backpressure_until_confidence_and_v33_release"
                ),
                "v33_admission_reserve_bytes": confidence_reserve["reserve_bytes"],
            }
        )

    return FrozenRunPlan(
        scaling=scaling_value,
        boundary_fitting=boundary_value,
        fragmentation_regularization=fragmentation_value,
        range_selection=range_value,
        selected_tile_count=selected_count,
        excluded_tile_count=excluded_count,
        spatial_plan=spatial_plan,
        package_plan=package_plan,
        partitions=partitions,
        storage_report=storage_value,
        v33_enabled=v33_enabled,
    )
