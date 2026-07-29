"""Pure construction of the semantic pipeline process queue."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .result_catalog import stream_by_id
from .run_spec import (
    BOUNDARY_REGULARIZATION_DEFAULT,
    DEFAULT_TILE_OVERLAP,
    FORMAL_VECTORIZATION_METHOD,
    MOSAIC_STRATEGY,
    RAW_VECTORIZATION_METHOD,
)


def _resume_arg(resume: bool) -> list[str]:
    return ["--resume"] if resume else []


def build_pipeline_steps(
    run_spec: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    resume: bool = False,
    ready_stream_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return serial QProcess steps with explicit stream dependencies."""
    run_spec_path = str(catalog["run_spec"])
    class_map = str(run_spec["class_mapping_snapshot"])
    tile = run_spec.get("tile") or {}
    overlap = str(tile.get("overlap", DEFAULT_TILE_OVERLAP))
    mosaic_strategy = str(tile.get("mosaic_strategy") or "")
    if mosaic_strategy != MOSAIC_STRATEGY:
        raise ValueError(
            f"run_spec tile.mosaic_strategy must equal {MOSAIC_STRATEGY!r}, got {mosaic_strategy!r}"
        )
    vectorization = run_spec.get("vectorization") or {}
    vector_method = str(vectorization.get("method") or "")
    if vector_method != FORMAL_VECTORIZATION_METHOD:
        raise ValueError(
            f"run_spec vectorization.method must equal {FORMAL_VECTORIZATION_METHOD!r}, got {vector_method!r}"
        )
    if str(vectorization.get("raw_method") or "") != RAW_VECTORIZATION_METHOD:
        raise ValueError("run_spec vectorization.raw_method is not the hard-mask audit contract")
    regularization = dict(BOUNDARY_REGULARIZATION_DEFAULT)
    regularization.update(dict(run_spec.get("boundary_regularization") or {}))
    if (
        regularization.get("enabled") is not True
        or regularization.get("mode") != FORMAL_VECTORIZATION_METHOD
        or regularization.get("preserve_outer_boundary") is not True
        or regularization.get("natural_smoothing") is not False
        or float(regularization.get("probability_smoothing_sigma", -1)) != 0.0
    ):
        raise ValueError("run_spec boundary_regularization is not the mandatory subpixel contract")
    device = str((run_spec.get("runtime") or {}).get("effective_device") or "cpu")
    ready = set(ready_stream_ids)
    steps: list[dict[str, Any]] = []

    # All models infer first, which guarantees fusion sees a complete collection
    # of per-model manifests and keeps model loading to one process per model.
    for model in run_spec.get("models") or []:
        model_id = str(model["model_id"])
        stream_id = f"model:{model_id}"
        if stream_id in ready:
            continue
        steps.append({
            "label": f"model_batch:{model_id}",
            "stage": "running_models",
            "stream_id": stream_id,
            "script": "run_semantic_batch.sh",
            "args": [
                "--run-spec", run_spec_path,
                "--model-id", model_id,
                "--device", device,
                *_resume_arg(resume),
            ],
            "continue_on_failure": True,
        })

    for model in run_spec.get("models") or []:
        model_id = str(model["model_id"])
        stream_id = f"model:{model_id}"
        if stream_id in ready:
            continue
        stream = stream_by_id(catalog, stream_id)
        paths = stream["paths"]
        steps.extend([
            {
                "label": f"mosaic:{stream_id}",
                "stage": "building_model_streams",
                "stream_id": stream_id,
                "requires_stream": stream_id,
                "script": "run_mosaic.sh",
                "args": [
                    "--mask_dir", paths["tile_mask_dir"],
                    "--conf_dir", paths["tile_confidence_dir"],
                    "--score_dir", paths["tile_score_dir"],
                    "--output_mask", paths["mask_mosaic"],
                    "--output_conf", paths["confidence_mosaic"],
                    "--output_probabilities", paths["probability_mosaic"],
                    "--overlap", overlap,
                    "--strategy", mosaic_strategy,
                ],
            },
            {
                "label": f"polygonize:{stream_id}",
                "stage": "building_model_streams",
                "stream_id": stream_id,
                "requires_stream": stream_id,
                "script": "run_polygonize.sh",
                "args": [
                    "--mask", paths["mask_mosaic"],
                    "--confidence", paths["confidence_mosaic"],
                    "--output", paths["semantic_polygons_raw"],
                    "--run-id", str(run_spec["run_id"]),
                    "--stream-id", stream_id,
                    "--result-kind", "model",
                    "--model-id", model_id,
                    "--model-version", str(model.get("version") or ""),
                    "--class-map", class_map,
                ],
            },
            {
                "label": f"subpixel_vectorize:{stream_id}",
                "stage": "regularizing_model_streams",
                "stream_id": stream_id,
                "requires_stream": stream_id,
                "script": "run_subpixel_vectorize.sh",
                "args": [
                    "--probabilities", paths["probability_mosaic"],
                    "--mask", paths["mask_mosaic"],
                    "--confidence", paths["confidence_mosaic"],
                    "--raw", paths["semantic_polygons_raw"],
                    "--output", paths["semantic_polygons"],
                    "--report", paths["boundary_regularization_report"],
                    "--run-id", str(run_spec["run_id"]),
                    "--stream-id", stream_id,
                    "--result-kind", "model",
                    "--model-id", model_id,
                    "--model-version", str(model.get("version") or ""),
                    "--class-map", class_map,
                    "--interpolation-strength", str(regularization["interpolation_strength"]),
                    "--coverage-tolerance-px", str(regularization["coverage_tolerance_px"]),
                    "--max-deviation-px", str(regularization["max_deviation_px"]),
                    "--stripe-rows", str(regularization["stripe_rows"]),
                ],
            },
            {
                "label": f"difference:{stream_id}",
                "stage": "applying_difference",
                "stream_id": stream_id,
                "requires_stream": stream_id,
                "python_action": "difference",
            },
        ])

    fusion = run_spec.get("fusion")
    if fusion:
        profile_id = str(fusion["profile_id"])
        stream_id = f"fusion:{profile_id}"
        if stream_id not in ready:
            stream = stream_by_id(catalog, stream_id)
            paths = stream["paths"]
            required = [f"model:{model_id}" for model_id in fusion.get("required_model_ids") or []]
            steps.extend([
                {
                    "label": f"fusion_batch:{profile_id}",
                    "stage": "running_fusion",
                    "stream_id": stream_id,
                    "requires_streams": required,
                    "script": "run_fusion.sh",
                    "args": [
                        "--run-spec", run_spec_path,
                        "--profile", str(fusion["snapshot_path"]),
                        "--device", device,
                        *_resume_arg(resume),
                    ],
                    "continue_on_failure": True,
                },
                {
                    "label": f"mosaic:{stream_id}",
                    "stage": "building_fusion_stream",
                    "stream_id": stream_id,
                    "requires_stream": stream_id,
                    "script": "run_mosaic.sh",
                    "args": [
                        "--mask_dir", paths["tile_mask_dir"],
                        "--conf_dir", paths["tile_confidence_dir"],
                        "--score_dir", paths["tile_score_dir"],
                        "--output_mask", paths["mask_mosaic"],
                        "--output_conf", paths["confidence_mosaic"],
                        "--output_probabilities", paths["probability_mosaic"],
                        "--overlap", overlap,
                        "--strategy", mosaic_strategy,
                    ],
                },
                {
                    "label": f"polygonize:{stream_id}",
                    "stage": "building_fusion_stream",
                    "stream_id": stream_id,
                    "requires_stream": stream_id,
                    "script": "run_polygonize.sh",
                    "args": [
                        "--mask", paths["mask_mosaic"],
                        "--confidence", paths["confidence_mosaic"],
                        "--output", paths["semantic_polygons_raw"],
                        "--run-id", str(run_spec["run_id"]),
                        "--stream-id", stream_id,
                        "--result-kind", "fusion",
                        "--fusion-profile-id", profile_id,
                        "--model-version", profile_id,
                        "--class-map", class_map,
                    ],
                },
                {
                    "label": f"subpixel_vectorize:{stream_id}",
                    "stage": "regularizing_fusion_stream",
                    "stream_id": stream_id,
                    "requires_stream": stream_id,
                    "script": "run_subpixel_vectorize.sh",
                    "args": [
                        "--probabilities", paths["probability_mosaic"],
                        "--mask", paths["mask_mosaic"],
                        "--confidence", paths["confidence_mosaic"],
                        "--raw", paths["semantic_polygons_raw"],
                        "--output", paths["semantic_polygons"],
                        "--report", paths["boundary_regularization_report"],
                        "--run-id", str(run_spec["run_id"]),
                        "--stream-id", stream_id,
                        "--result-kind", "fusion",
                        "--fusion-profile-id", profile_id,
                        "--model-version", profile_id,
                        "--class-map", class_map,
                        "--interpolation-strength", str(regularization["interpolation_strength"]),
                        "--coverage-tolerance-px", str(regularization["coverage_tolerance_px"]),
                        "--max-deviation-px", str(regularization["max_deviation_px"]),
                        "--stripe-rows", str(regularization["stripe_rows"]),
                    ],
                },
                {
                    "label": f"difference:{stream_id}",
                    "stage": "applying_difference",
                    "stream_id": stream_id,
                    "requires_stream": stream_id,
                    "python_action": "difference",
                },
            ])
    return steps
