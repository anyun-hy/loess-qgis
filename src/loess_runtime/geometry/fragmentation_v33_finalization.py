"""Durable V3.3 global validation and authoritative publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, cast

import rasterio  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]

from labeling_tool.shared.contracts.run_spec import CLASS_ORDER, load_json, sha256_file
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.fragmentation_global_connectivity import (
    GlobalConnectivityError,
    audit_partitioned_connectivity,
    connectivity_hard_gate,
)
from loess_runtime.geometry.fragmentation_v33_artifact_io import (
    ArtifactIndex,
    ready_artifact_index,
    verified_artifact_path,
    write_atomic_json,
)
from loess_runtime.geometry.fragmentation_v33_candidate import runtime_policy
from loess_runtime.geometry.fragmentation_v33_contract import (
    FragmentationV33WorkPackageError,
    intersect_windows,
    normalize_window,
    physical_metrics,
)

__all__ = [
    "FinalizeHeartbeat",
    "PreparedV33Finalization",
    "V33FinalizeStore",
    "finalize_authoritative_v33",
    "prepare_v33_finalization",
]


@dataclass(frozen=True)
class PreparedV33Finalization:
    """Validated domain inputs gathered before the durable exception boundary."""

    partitions: tuple[Mapping[str, Any], ...]
    artifacts: ArtifactIndex
    core_area: int
    global_area: int
    overlap_pair_count: int


class FinalizeHeartbeat(Protocol):
    def progress(self, current: int) -> None: ...

    def fence(self) -> None: ...

    def stop_for_publication(self) -> None: ...


class V33FinalizeStore(Protocol):
    @property
    def artifacts(self) -> ArtifactRepository: ...

    def complete_fragmentation_v33_finalize(
        self,
        job_id: int,
        lease_token: str,
        outputs: Sequence[Mapping[str, Any]],
        *,
        report_path: str | Path,
        report_byte_count: int,
        report_sha256: str,
    ) -> bool: ...

    def owner_publication(
        self,
        run_id: str,
        run_dir: str | Path,
    ) -> AbstractContextManager[V33FinalizeStore]: ...

    def fragmentation_v33_attempt_commit(
        self,
        job_id: int,
        lease_token: str,
    ) -> AbstractContextManager[V33FinalizeStore]: ...


def prepare_v33_finalization(
    artifacts: ArtifactRepository,
    run_id: str,
    stream_id: str,
    partitions: Sequence[Mapping[str, Any]],
) -> PreparedV33Finalization:
    """Validate Core ownership and freeze the ready Artifact inventory."""

    frozen_partitions = tuple(partitions)
    global_window = {
        "x0": min(int(item["core_window"]["x0"]) for item in frozen_partitions),
        "y0": min(int(item["core_window"]["y0"]) for item in frozen_partitions),
        "x1": max(int(item["core_window"]["x1"]) for item in frozen_partitions),
        "y1": max(int(item["core_window"]["y1"]) for item in frozen_partitions),
    }
    core_area = sum(
        (int(item["core_window"]["x1"]) - int(item["core_window"]["x0"]))
        * (int(item["core_window"]["y1"]) - int(item["core_window"]["y0"]))
        for item in frozen_partitions
    )
    global_area = (global_window["x1"] - global_window["x0"]) * (
        global_window["y1"] - global_window["y0"]
    )
    overlap_pairs = 0
    for index, first in enumerate(frozen_partitions):
        for second in frozen_partitions[index + 1 :]:
            if (
                intersect_windows(
                    normalize_window(first["core_window"]),
                    normalize_window(second["core_window"]),
                )
                is not None
            ):
                overlap_pairs += 1
    if core_area != global_area or overlap_pairs:
        raise FragmentationV33WorkPackageError(
            "Partition Core ownership has a geometric gap or overlap"
        )
    artifact_index = ready_artifact_index(artifacts, run_id, stream_id)
    return PreparedV33Finalization(
        partitions=frozen_partitions,
        artifacts=artifact_index,
        core_area=core_area,
        global_area=global_area,
        overlap_pair_count=overlap_pairs,
    )


def _prepare_atomic_copy(
    source: Path,
    destination: Path,
    staging_key: str,
    *,
    raster_tags: Mapping[str, str] | None = None,
) -> Path:
    """Build a same-directory publication file without replacing canonical data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{staging_key}.tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if raster_tags is not None:
            with rasterio.open(temporary, "r+") as raster:
                raster.update_tags(**dict(raster_tags))
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_prepared(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _release_staged_outputs(
    store: V33FinalizeStore,
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    partition_ids: list[str],
) -> dict[str, int]:
    """Best-effort release after the authority barrier is already committed."""

    released = 0
    released_bytes = 0
    for partition_id in partition_ids:
        for kind in ("v33_staged_mask", "v33_staged_audit"):
            artifact = artifacts.get((partition_id, kind))
            if artifact is None:
                continue
            claimed = store.artifacts.claim_artifact_cleanup(
                int(artifact["artifact_id"])
            )
            if claimed is None:
                continue
            path = Path(str(claimed.get("path") or ""))
            success = False
            try:
                path.unlink(missing_ok=True)
                success = not path.exists()
            finally:
                store.artifacts.finish_artifact_cleanup(
                    int(claimed["artifact_id"]), success=success
                )
            if success:
                released += 1
                released_bytes += int(claimed.get("byte_count") or 0)
    return {"artifact_count": released, "byte_count": released_bytes}


def finalize_authoritative_v33(
    spec: Mapping[str, Any],
    store: V33FinalizeStore,
    job: Mapping[str, Any],
    prepared: PreparedV33Finalization,
    heartbeat: FinalizeHeartbeat,
    canonical_root: Path,
) -> dict[str, Any]:
    """Validate staged outputs and cross the single authority barrier."""

    run_id = str(spec["run_id"])
    partitions = prepared.partitions
    artifacts = prepared.artifacts
    summaries: list[dict[str, Any]] = []
    authoritative_outputs: list[dict[str, Any]] = []
    staged_records: list[dict[str, Any]] = []
    for index, partition in enumerate(partitions, start=1):
        partition_id = str(partition["partition_id"])
        staged_mask = verified_artifact_path(
            artifacts.get((partition_id, "v33_staged_mask")),
            kind="v33_staged_mask",
            partition_id=partition_id,
        )
        staged_audit = verified_artifact_path(
            artifacts.get((partition_id, "v33_staged_audit")),
            kind="v33_staged_audit",
            partition_id=partition_id,
        )
        audit = load_json(staged_audit)
        acceptance = dict(audit.get("acceptance") or {})
        if any(
            int(acceptance.get(key, -1)) != 0
            for key in (
                "gap_pixels",
                "overlap_pixels",
                "outside_pixels",
                "invalid_pixels",
                "protected_source_loss_pixel_count",
                "probability_nonfinite_pixels",
                "probability_negative_pixels",
                "probability_zero_sum_pixels",
                "probability_bad_sum_pixels",
            )
        ):
            raise FragmentationV33WorkPackageError(
                f"{partition_id}: staged V3.3 Core did not pass acceptance"
            )
        baseline_path = verified_artifact_path(
            artifacts.get((partition_id, "v3_baseline_core")),
            kind="v3_baseline_core",
            partition_id=partition_id,
        )
        core = normalize_window(partition["core_window"])
        staged_records.append(
            {
                "partition": partition,
                "partition_id": partition_id,
                "staged_mask": staged_mask,
                "staged_audit": staged_audit,
                "baseline_path": baseline_path,
                "audit": audit,
                "acceptance": acceptance,
                "pixel_area_m2": physical_metrics(
                    Affine(*[float(value) for value in spec["raster"]["transform"]]),
                    str(spec["raster"]["crs"]),
                    core,
                )["pixel_area_m2"],
            }
        )
        candidate = dict(audit.get("candidate") or {})
        summaries.append(
            {
                "partition_id": partition_id,
                "output_mask_sha256": "",
                "changed_pixel_count": int(candidate.get("changed_pixel_count", 0)),
                "baseline_dynamic_fragments": int(
                    (candidate.get("baseline") or {}).get(
                        "dynamic_fragments_4_connected", 0
                    )
                ),
                "candidate_dynamic_fragments": int(
                    (candidate.get("result") or {}).get(
                        "dynamic_fragments_4_connected", 0
                    )
                ),
                "acceptance": acceptance,
            }
        )
        heartbeat.progress(index)
        heartbeat.fence()

    transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
    crs = str(spec["raster"]["crs"])
    selected_policy = runtime_policy()
    dynamic_thresholds = {
        int(code): float(
            selected_policy.class_policies[int(code)].dynamic_fragmentation_m2
        )
        for code in CLASS_ORDER
    }

    def audit_records(
        path_key: str, *, encoding_from_suffix: bool, offset: int
    ) -> dict[str, Any]:
        records = [
            {
                "partition_id": item["partition_id"],
                "core_window": item["partition"]["core_window"],
                "path": item[path_key],
                "encoding": (
                    "class_codes"
                    if encoding_from_suffix
                    and Path(item[path_key]).suffix.lower() == ".npy"
                    else "indices"
                ),
                "pixel_area_m2": item["pixel_area_m2"],
            }
            for item in staged_records
        ]

        def progress(current: int) -> None:
            heartbeat.progress(offset + current)
            heartbeat.fence()

        return cast(
            dict[str, Any],
            audit_partitioned_connectivity(
                records,
                class_codes=CLASS_ORDER,
                dynamic_thresholds_m2=dynamic_thresholds,
                expected_crs=crs,
                global_transform=transform,
                progress=progress,
            ),
        )

    try:
        baseline_connectivity = audit_records(
            "baseline_path", encoding_from_suffix=True, offset=len(partitions)
        )
        candidate_connectivity = audit_records(
            "staged_mask", encoding_from_suffix=False, offset=len(partitions) * 2
        )
    except GlobalConnectivityError as error:
        raise FragmentationV33WorkPackageError(
            f"V3.3 global 4-connected audit failed: {error}"
        ) from error
    connectivity_gate = connectivity_hard_gate(
        baseline_connectivity, candidate_connectivity
    )
    if not connectivity_gate["passed"]:
        raise FragmentationV33WorkPackageError(
            "V3.3 global 4-connected component/fragment hard gate failed"
        )

    acceptance_keys = (
        "gap_pixels",
        "overlap_pixels",
        "outside_pixels",
        "invalid_pixels",
        "protected_source_loss_pixel_count",
        "probability_nonfinite_pixels",
        "probability_negative_pixels",
        "probability_zero_sum_pixels",
        "probability_bad_sum_pixels",
    )
    acceptance_totals = {
        key: sum(int(item["acceptance"].get(key, 0)) for item in summaries)
        for key in acceptance_keys
    }
    acceptance_totals.update(
        {
            "partition_core_area": int(prepared.core_area),
            "global_core_area": int(prepared.global_area),
            "core_overlap_pair_count": int(prepared.overlap_pair_count),
            "argmax_tie_pixels": sum(
                int(item["acceptance"].get("argmax_tie_pixels", 0))
                for item in summaries
            ),
            "near_tie_pixels": sum(
                int(item["acceptance"].get("near_tie_pixels", 0)) for item in summaries
            ),
            "changed_pixel_count": sum(
                int(item["changed_pixel_count"]) for item in summaries
            ),
            "partition_local_baseline_dynamic_fragments": sum(
                int(item["baseline_dynamic_fragments"]) for item in summaries
            ),
            "partition_local_result_dynamic_fragments": sum(
                int(item["candidate_dynamic_fragments"]) for item in summaries
            ),
            "baseline_components_4_connected": int(
                baseline_connectivity["components_4_connected"]
            ),
            "result_components_4_connected": int(
                candidate_connectivity["components_4_connected"]
            ),
            "baseline_dynamic_fragments": int(
                baseline_connectivity["dynamic_fragments_4_connected"]
            ),
            "result_dynamic_fragments": int(
                candidate_connectivity["dynamic_fragments_4_connected"]
            ),
        }
    )
    if any(acceptance_totals[key] for key in acceptance_keys):
        raise FragmentationV33WorkPackageError("V3.3 global raster acceptance failed")

    publication_key = (
        f"job{int(job['job_id'])}."
        f"{hashlib.sha256(str(job['lease_token']).encode()).hexdigest()}"
    )
    report_path = canonical_root / "fragmentation_v33_report.json"
    prepared_replacements: list[tuple[Path, Path]] = []
    prepared_report = report_path.with_name(
        f".{report_path.name}.{publication_key}.publication"
    )
    try:
        for summary, item in zip(summaries, staged_records):
            partition_id = str(item["partition_id"])
            mask_path = canonical_root / "raster_parts" / f"{partition_id}_mask.tif"
            audit_path = (
                canonical_root / "fragmentation_v33_audits" / f"{partition_id}.json"
            )
            prepared_mask = _prepare_atomic_copy(
                Path(item["staged_mask"]),
                mask_path,
                publication_key,
                raster_tags={
                    "classification_authority": (
                        "fragmentation_v33_authoritative_fusion_core_v1"
                    ),
                    "production_replacement": "true",
                },
            )
            prepared_replacements.append((prepared_mask, mask_path))
            mask_sha256 = sha256_file(prepared_mask)
            canonical_audit = dict(item["audit"])
            canonical_audit.update(
                {
                    "publication": "authoritative_fusion_core",
                    "production_replacement": True,
                    "output_mask_sha256": mask_sha256,
                }
            )
            canonical_audit.pop("audit_sha256", None)
            canonical_audit["audit_sha256"] = hashlib.sha256(
                json.dumps(
                    canonical_audit,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            prepared_audit = audit_path.with_name(
                f".{audit_path.name}.{publication_key}.publication"
            )
            write_atomic_json(prepared_audit, canonical_audit)
            prepared_replacements.append((prepared_audit, audit_path))
            audit_sha256 = sha256_file(prepared_audit)
            summary["output_mask_sha256"] = mask_sha256
            authoritative_outputs.append(
                {
                    "partition_id": partition_id,
                    "mask_path": mask_path,
                    "mask_byte_count": prepared_mask.stat().st_size,
                    "mask_sha256": mask_sha256,
                    "audit_path": audit_path,
                    "audit_byte_count": prepared_audit.stat().st_size,
                    "audit_sha256": audit_sha256,
                }
            )
            heartbeat.fence()
        report = {
            "schema_version": 2,
            "status": "ready",
            "validation_status": "passed",
            "publication": "authoritative_fusion_core",
            "publication_barrier": (
                "all_v33_partition_jobs_ready_and_global_4_connected_gate_passed"
            ),
            "production_replacement": True,
            "run_id": run_id,
            "stream_id": str(job["stream_id"]),
            "partition_count": len(summaries),
            "acceptance": acceptance_totals,
            "global_connectivity_4_connected": {
                "baseline": baseline_connectivity,
                "candidate": candidate_connectivity,
                "hard_gate": connectivity_gate,
            },
            "partitions": summaries,
        }
        write_atomic_json(prepared_report, report)
        report_byte_count = prepared_report.stat().st_size
        report_sha256 = sha256_file(prepared_report)
        heartbeat.stop_for_publication()
        # Long copies and hashing are complete.  Lock the Run publication file,
        # then lock and validate the exact finalize lease before the first
        # canonical replacement; both scopes share one database transaction.
        with store.owner_publication(run_id, Path(str(spec["run_dir"]))) as owner:
            with owner.fragmentation_v33_attempt_commit(
                int(job["job_id"]), str(job["lease_token"])
            ) as publication:
                for prepared_path, canonical_path in prepared_replacements:
                    _publish_prepared(prepared_path, canonical_path)
                _publish_prepared(prepared_report, report_path)
                if not publication.complete_fragmentation_v33_finalize(
                    int(job["job_id"]),
                    str(job["lease_token"]),
                    authoritative_outputs,
                    report_path=report_path,
                    report_byte_count=report_byte_count,
                    report_sha256=report_sha256,
                ):
                    raise FragmentationV33WorkPackageError(
                        "V3.3 finalize lease expired before publish"
                    )
    finally:
        for prepared_path, _canonical_path in prepared_replacements:
            prepared_path.unlink(missing_ok=True)
        prepared_report.unlink(missing_ok=True)
    _release_staged_outputs(
        store,
        artifacts,
        [str(item["partition_id"]) for item in summaries],
    )
    return report
