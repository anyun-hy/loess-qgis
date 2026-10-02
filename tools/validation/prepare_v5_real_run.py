#!/usr/bin/env python3
"""Prepare one bounded real-image V5 Run through production planning code."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_GRID_ROWS = 2
DEFAULT_GRID_COLS = 2
TILE_SIZE = 512
OVERLAP = 192
STRIDE = TILE_SIZE - OVERLAP
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Extent:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def xMinimum(self) -> float:
        return self.xmin

    def yMinimum(self) -> float:
        return self.ymin

    def xMaximum(self) -> float:
        return self.xmax

    def yMaximum(self) -> float:
        return self.ymax

    def as_dict(self) -> dict[str, float]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }


class _Crs:
    def __init__(self, auth_id: str) -> None:
        self._auth_id = str(auth_id)

    def authid(self) -> str:
        return self._auth_id


class RasterLayer:
    def __init__(self, path: Path, crs: str, resolution_x: float, resolution_y: float):
        self._path = path
        self._crs = _Crs(crs)
        self._resolution_x = float(resolution_x)
        self._resolution_y = float(resolution_y)

    def source(self) -> str:
        return str(self._path)

    def crs(self) -> _Crs:
        return self._crs

    def rasterUnitsPerPixelX(self) -> float:
        return self._resolution_x

    def rasterUnitsPerPixelY(self) -> float:
        return self._resolution_y


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicit source-pixel grid V5 PostgreSQL Run "
            "from deployed production code, real assets, and a real GeoTIFF"
        )
    )
    parser.add_argument("--plugin-parent", required=True)
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--source-raster", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-dsn", required=True)
    parser.add_argument("--state-schema", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--fusion-profile-id")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--grid-rows",
        type=_positive_int,
        default=DEFAULT_GRID_ROWS,
        help=f"number of source Tile rows (default: {DEFAULT_GRID_ROWS})",
    )
    parser.add_argument(
        "--grid-cols",
        type=_positive_int,
        default=DEFAULT_GRID_COLS,
        help=f"number of source Tile columns (default: {DEFAULT_GRID_COLS})",
    )
    parser.add_argument(
        "--row-offset",
        type=_nonnegative_int,
        help="zero-based source pixel row; omitted centers the grid vertically",
    )
    parser.add_argument(
        "--col-offset",
        type=_nonnegative_int,
        help="zero-based source pixel column; omitted centers the grid horizontally",
    )
    parser.add_argument(
        "--score-cache-budget-gb",
        type=_positive_finite_float,
        help=(
            "explicit Run-only score cache budget passed through the production "
            "storage planner; omitted preserves the deployed effective setting"
        ),
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} is missing or is a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PreparationError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must contain a JSON object")
    return value


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _bootstrap(plugin_parent: Path, scripts_dir: Path) -> dict[str, str]:
    plugin_root = plugin_parent / "labeling_tool"
    runtime_root = scripts_dir / "loess_runtime"
    if not plugin_root.is_dir() or plugin_root.is_symlink():
        raise PreparationError(f"deployed plugin is missing: {plugin_root}")
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise PreparationError(f"deployed inference runtime is missing: {runtime_root}")
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(plugin_parent))
    import labeling_tool
    import loess_runtime

    plugin_origin = Path(labeling_tool.__file__).resolve()
    runtime_origin = Path(loess_runtime.__file__).resolve()
    if not _under(plugin_origin, plugin_root.resolve()):
        raise PreparationError(
            f"labeling_tool was imported outside --plugin-parent: {plugin_origin}"
        )
    if not _under(runtime_origin, runtime_root.resolve()):
        raise PreparationError(
            f"loess_runtime was imported outside --scripts-dir: {runtime_origin}"
        )
    return {
        "plugin_origin": str(plugin_origin),
        "runtime_origin": str(runtime_origin),
    }


def _source_bundle(
    plugin_root: Path, scripts_dir: Path, expected_digest: str
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(expected_digest):
        raise PreparationError("--expected-source-bundle-sha256 must be lowercase SHA256")
    project_manifest = _json_object(scripts_dir.parent / "project_manifest.json", "project manifest")
    plugin_manifest = _json_object(plugin_root / "deployment_manifest.json", "plugin manifest")
    project_source = project_manifest.get("source")
    plugin_source = plugin_manifest.get("source")
    if not isinstance(project_source, Mapping) or not isinstance(plugin_source, Mapping):
        raise PreparationError("deployment manifests are missing source identity")
    project_digest = str(project_source.get("source_bundle_sha256") or "")
    plugin_digest = str(plugin_source.get("source_bundle_sha256") or "")
    if project_digest != expected_digest or plugin_digest != expected_digest:
        raise PreparationError("deployed project/plugin source bundle does not match the expected SHA256")
    return {
        "source_bundle_sha256": expected_digest,
        "git_sha": str(project_manifest.get("git_sha") or ""),
        "source_kind": str(project_source.get("kind") or ""),
        "git_dirty": project_source.get("git_dirty"),
    }


def _validated_environment(
    path: Path,
    scripts_dir: Path,
    requested_profile_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], str, dict[str, Any]]:
    from labeling_tool.shared.contracts.run_spec import sha256_file
    from loess_runtime.system.environment_report import environment_fingerprint

    report = _json_object(path, "environment report")
    if report.get("schema_version") != 1:
        raise PreparationError("environment report schema_version must equal 1")
    status = str(report.get("status") or "")
    if status not in {"ready", "warning"}:
        raise PreparationError(f"environment report is not runnable: {status or '<missing>'}")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise PreparationError("environment report checks are missing")
    failed_checks = [
        str(item.get("id") or "<unknown>")
        for item in checks
        if isinstance(item, Mapping) and str(item.get("status") or "") == "error"
    ]
    if failed_checks:
        raise PreparationError("environment report contains error checks: " + ", ".join(failed_checks))
    current_fingerprint = environment_fingerprint(scripts_dir)
    if str(report.get("config_fingerprint") or "") != current_fingerprint:
        raise PreparationError("environment report fingerprint does not match the deployed project")
    effective = report.get("effective")
    if not isinstance(effective, dict) or effective.get("schema_version") != 2:
        raise PreparationError("environment report is missing effective Schema v2 configuration")
    device = str((effective.get("runtime") or {}).get("effective_device") or "")
    if re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise PreparationError(f"effective runtime device must be CUDA, got {device or '<missing>'}")

    profiles = [item for item in effective.get("fusion_profiles") or [] if isinstance(item, Mapping)]
    runnable = [
        item
        for item in profiles
        if bool(item.get("enabled"))
        and bool(item.get("available"))
        and bool(item.get("trusted"))
        and str(item.get("status") or "") == "approved"
    ]
    if requested_profile_id:
        runnable = [item for item in runnable if item.get("profile_id") == requested_profile_id]
    if len(runnable) != 1:
        raise PreparationError(
            "expected exactly one selected enabled/available/trusted approved Fusion profile, "
            f"got {len(runnable)}"
        )
    profile = dict(runnable[0])
    profile_id = str(profile.get("profile_id") or "")
    profile_path = Path(str(profile.get("file_path") or "")).expanduser().resolve()
    expected_profile_sha = str(profile.get("sha256") or "")
    recorded_profile_sha = str(profile.get("file_sha256") or "")
    if (
        not SHA256_RE.fullmatch(expected_profile_sha)
        or recorded_profile_sha != expected_profile_sha
        or not profile_path.is_file()
        or profile_path.is_symlink()
        or sha256_file(profile_path) != expected_profile_sha
    ):
        raise PreparationError(f"Fusion profile file/hash is not frozen: {profile_id}")

    required_ids = tuple(str(value) for value in profile.get("required_model_ids") or ())
    if not required_ids or len(set(required_ids)) != len(required_ids):
        raise PreparationError("Fusion profile required_model_ids are missing or duplicated")
    models_by_id = {
        str(item.get("model_id") or ""): item
        for item in effective.get("semantic_models") or []
        if isinstance(item, Mapping)
    }
    model_evidence = []
    for model_id in required_ids:
        model = models_by_id.get(model_id)
        if model is None or not bool(model.get("enabled")):
            raise PreparationError(f"required semantic model is missing or disabled: {model_id}")
        artifact = Path(str(model.get("artifact_path") or "")).expanduser().resolve()
        expected_sha = str(model.get("sha256") or "")
        if (
            not SHA256_RE.fullmatch(expected_sha)
            or not artifact.is_file()
            or artifact.is_symlink()
            or sha256_file(artifact) != expected_sha
        ):
            raise PreparationError(f"semantic model file/hash is not frozen: {model_id}")
        model_evidence.append(
            {"model_id": model_id, "artifact_path": str(artifact), "sha256": expected_sha}
        )
    return report, effective, required_ids, profile_id, {
        "report_status": status,
        "config_fingerprint": current_fingerprint,
        "effective_device": device,
        "models": model_evidence,
        "fusion_profile": {
            "profile_id": profile_id,
            "file_path": str(profile_path),
            "sha256": expected_profile_sha,
        },
    }


def _effective_for_run(
    effective: Mapping[str, Any], score_cache_budget_gb: float | None
) -> dict[str, Any]:
    """Copy effective configuration and apply one explicit Run-only disk budget."""

    value = dict(effective)
    scaling = dict(effective.get("scaling") or {})
    if score_cache_budget_gb is not None:
        budget = float(score_cache_budget_gb)
        if not math.isfinite(budget) or budget <= 0:
            raise PreparationError("score cache budget must be finite and positive")
        scaling["score_cache_budget_gb"] = budget
    value["scaling"] = scaling
    return value


def _grid(
    source: Path,
    *,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
    row_offset: int | None = None,
    col_offset: int | None = None,
) -> tuple[RasterLayer, Extent, tuple[dict[str, Any], ...], dict[str, int]]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as error:
        raise PreparationError("rasterio is required in the qgis Conda environment") from error

    try:
        with rasterio.open(source) as dataset:
            rows = int(grid_rows)
            cols = int(grid_cols)
            source_width = int(dataset.width)
            source_height = int(dataset.height)
            if rows < 1 or cols < 1:
                raise PreparationError("grid rows and columns must be at least 1")
            if row_offset is not None and int(row_offset) < 0:
                raise PreparationError("row offset must be non-negative")
            if col_offset is not None and int(col_offset) < 0:
                raise PreparationError("column offset must be non-negative")
            if dataset.count < 3:
                raise PreparationError("source raster must contain at least three bands")
            if dataset.crs is None:
                raise PreparationError("source raster CRS is missing")
            transform = dataset.transform
            if (
                float(transform.a) <= 0
                or float(transform.e) >= 0
                or abs(float(transform.b)) > 1.0e-12
                or abs(float(transform.d)) > 1.0e-12
            ):
                raise PreparationError("source raster must use a north-up, unrotated affine transform")
            window_width = TILE_SIZE + (cols - 1) * STRIDE
            window_height = TILE_SIZE + (rows - 1) * STRIDE
            if source_width < window_width or source_height < window_height:
                raise PreparationError(
                    "source raster is smaller than the requested grid window: "
                    f"source={source_width}x{source_height}, "
                    f"required={window_width}x{window_height}"
                )
            col_off = (
                (source_width - window_width) // 2
                if col_offset is None
                else int(col_offset)
            )
            row_off = (
                (source_height - window_height) // 2
                if row_offset is None
                else int(row_offset)
            )
            if col_off + window_width > source_width:
                raise PreparationError(
                    "requested source pixel column window exceeds raster width: "
                    f"col_off={col_off}, width={window_width}, source_width={source_width}"
                )
            if row_off + window_height > source_height:
                raise PreparationError(
                    "requested source pixel row window exceeds raster height: "
                    f"row_off={row_off}, height={window_height}, source_height={source_height}"
                )
            tiles = []
            for row in range(rows):
                for col in range(cols):
                    window = Window(
                        col_off + col * STRIDE,
                        row_off + row * STRIDE,
                        TILE_SIZE,
                        TILE_SIZE,
                    )
                    left, bottom, right, top = rasterio.windows.bounds(window, transform)
                    tiles.append(
                        {
                            "row": row,
                            "col": col,
                            "bounds": Extent(float(left), float(bottom), float(right), float(top)),
                        }
                    )
            full = Window(col_off, row_off, window_width, window_height)
            left, bottom, right, top = rasterio.windows.bounds(full, transform)
            extent = Extent(float(left), float(bottom), float(right), float(top))
            layer = RasterLayer(
                source,
                dataset.crs.to_string(),
                abs(float(transform.a)),
                abs(float(transform.e)),
            )
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(f"source raster cannot be inspected: {error}") from error
    return layer, extent, tuple(tiles), {
        "row_off": int(row_off),
        "col_off": int(col_off),
        "row_end": int(row_off + window_height),
        "col_end": int(col_off + window_width),
        "width": int(window_width),
        "height": int(window_height),
        "source_width": source_width,
        "source_height": source_height,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    plugin_parent = Path(args.plugin_parent).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve()
    plugin_root = plugin_parent / "labeling_tool"
    imports = _bootstrap(plugin_parent, scripts_dir)

    from labeling_tool.runs.deployment_contract import verify_project_runtime
    from labeling_tool.runs.run_builder_v5 import create_v5_run
    from labeling_tool.runs.run_planning import build_run_builder_kwargs
    from labeling_tool.shared.contracts.run_spec import (
        atomic_write_json,
        reserve_run_directory,
    )
    from labeling_tool.shared.state.postgres_state import (
        DEFAULT_POSTGRES_SCHEMA,
        is_postgres_location,
        validate_schema,
    )
    from loess_runtime.inference.tile_cache_probe import measure_tile_cache

    deployment = verify_project_runtime(scripts_dir, plugin_root=plugin_root)
    if deployment.get("status") != "ready":
        raise PreparationError(
            "deployed plugin/project contract failed: "
            + str(deployment.get("message") or "unknown error")
        )
    bundle = _source_bundle(
        plugin_root,
        scripts_dir,
        str(args.expected_source_bundle_sha256).strip().lower(),
    )
    dsn = str(args.state_dsn).strip()
    if not is_postgres_location(dsn):
        raise PreparationError("--state-dsn must be an explicit PostgreSQL DSN")
    schema = validate_schema(args.state_schema)
    if schema == DEFAULT_POSTGRES_SCHEMA:
        raise PreparationError("validation Run must not use the default production schema")
    os.environ["LOESS_STATE_DB_DSN"] = dsn
    os.environ["LOESS_STATE_DB_SCHEMA"] = schema

    environment_report, effective, model_ids, profile_id, asset_evidence = _validated_environment(
        Path(args.environment_report).expanduser().resolve(),
        scripts_dir,
        args.fusion_profile_id,
    )
    effective_for_run = _effective_for_run(
        effective,
        args.score_cache_budget_gb,
    )
    source = Path(args.source_raster).expanduser().resolve()
    if not source.is_file():
        raise PreparationError(f"source raster is missing: {source}")
    output = Path(args.output_root).expanduser()
    if output.is_symlink():
        raise PreparationError(f"output root cannot be a symlink: {output}")
    output = output.resolve()
    accepted_target = output / "accepted_labels.gpkg"
    if accepted_target.exists() or accepted_target.is_symlink():
        raise PreparationError(
            "validation output already contains accepted_labels.gpkg; "
            "an empty accepted-label audit cannot be asserted"
        )
    layer, extent, grid_tiles, source_window = _grid(
        source,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        row_offset=args.row_offset,
        col_offset=args.col_offset,
    )
    output.mkdir(parents=True, exist_ok=True)
    sample = grid_tiles[0]
    tile_cache_sample = measure_tile_cache(
        source,
        output,
        {
            "tile_id": "0_0",
            "row_no": 0,
            "col_no": 0,
            "bounds": sample["bounds"].as_dict(),
        },
    )
    if tile_cache_sample.get("status") != "passed":
        raise PreparationError("production Tile cache probe did not pass")
    expected_sample_window = {
        "x0": source_window["col_off"],
        "y0": source_window["row_off"],
        "x1": source_window["col_off"] + TILE_SIZE,
        "y1": source_window["row_off"] + TILE_SIZE,
    }
    actual_sample_window = {
        key: int((tile_cache_sample.get("sample_source_window") or {}).get(key, -1))
        for key in expected_sample_window
    }
    if actual_sample_window != expected_sample_window:
        raise PreparationError(
            "production Tile cache probe resolved a different source window: "
            f"{actual_sample_window}"
        )

    run_id, run_dir = reserve_run_directory(output, args.run_id)
    accepted_validation = {
        "status": "passed",
        "feature_count": 0,
        "overlap_pair_count": 0,
        "overlap_tolerance": max(
            layer.rasterUnitsPerPixelX() * layer.rasterUnitsPerPixelY() * 1.0e-6,
            1.0e-18,
        ),
        "crs": layer.crs().authid(),
        "source": "not_present",
    }
    kwargs = build_run_builder_kwargs(
        scripts_dir=str(scripts_dir),
        output_root=str(output),
        accepted_target_gpkg=str(accepted_target),
        raster_layer=layer,
        requested_extent=extent,
        processing_extent=extent,
        grid_tiles=grid_tiles,
        active_tiles=grid_tiles,
        range_selection={
            "mode": "extent",
            "selected_tile_count": args.grid_rows * args.grid_cols,
            "excluded_tile_count": 0,
            "clip_outputs": True,
        },
        effective_config=effective_for_run,
        environment_report=environment_report,
        accepted_validation=accepted_validation,
        skip_accepted=False,
        selected_model_ids=model_ids,
        fusion_profile_id=profile_id,
        boundary_smoothing_enabled=bool(
            (effective.get("boundary_fitting") or {}).get("enabled", True)
        ),
        overlap=OVERLAP,
        run_id=run_id,
        run_dir=str(run_dir),
        accepted_snapshot="",
        skipped_tiles=(),
        tile_cache_sample=tile_cache_sample,
    )
    kwargs["state_database"] = dsn
    spec, spec_path, _database = create_v5_run(**kwargs)
    if (
        spec.get("schema_version") != 2
        or spec.get("state_backend") != "postgresql"
        or spec.get("state_schema") != schema
        or str((spec.get("runtime") or {}).get("effective_device") or "")
        != asset_evidence["effective_device"]
        or str((spec.get("deployment_identity") or {}).get("source_bundle_sha256") or "")
        != bundle["source_bundle_sha256"]
        or str((spec.get("deployment_identity") or {}).get("status") or "")
        != "manifest_recorded"
    ):
        raise PreparationError("created Run does not preserve the requested V5 deployment/runtime identity")
    if args.score_cache_budget_gb is not None:
        frozen_scaling = spec.get("scaling") or {}
        frozen_storage = spec.get("storage_preflight") or {}
        if (
            str(frozen_scaling.get("score_cache_budget_mode") or "") != "explicit"
            or not math.isclose(
                float(frozen_scaling.get("score_cache_budget_gb") or 0),
                float(args.score_cache_budget_gb),
                rel_tol=0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(frozen_storage.get("configured_score_cache_budget_gb") or 0),
                float(args.score_cache_budget_gb),
                rel_tol=0,
                abs_tol=1.0e-12,
            )
        ):
            raise PreparationError(
                "created Run did not freeze the explicit score cache budget"
            )

    report = {
        "schema_version": 1,
        "kind": "v5_real_run_preparation",
        "status": "prepared",
        "success": True,
        "run_id": str(spec["run_id"]),
        "run_spec": str(spec_path),
        "run_spec_content_sha256": str(spec.get("run_spec_content_sha256") or ""),
        "state_backend": "postgresql",
        "state_schema": schema,
        "source_raster": str(source),
        "source_file_identity": dict((spec.get("raster") or {}).get("file_identity") or {}),
        "source_window": source_window,
        "grid": {
            "rows": args.grid_rows,
            "cols": args.grid_cols,
            "tile_count": args.grid_rows * args.grid_cols,
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "requested_row_offset": args.row_offset,
            "requested_col_offset": args.col_offset,
        },
        "score_cache_budget": {
            "requested_gb": args.score_cache_budget_gb,
            "mode": str((spec.get("scaling") or {}).get("score_cache_budget_mode") or ""),
            "resolved_gb": (spec.get("scaling") or {}).get("score_cache_budget_gb"),
            "package_tile_limit": (spec.get("storage_preflight") or {}).get(
                "package_tile_limit"
            ),
            "package_count": (spec.get("spatial_plan_summary") or {}).get(
                "package_count"
            ),
        },
        "processing_extent": extent.as_dict(),
        "tile_cache_probe": {
            "status": tile_cache_sample["status"],
            "measurement_method": tile_cache_sample["measurement_method"],
            "materialized_cache_bytes": tile_cache_sample["materialized_cache_bytes"],
            "sample_source_window": tile_cache_sample["sample_source_window"],
        },
        "deployment": bundle,
        "imports": imports,
        "assets": asset_evidence,
    }
    evidence_path = run_dir / "logs" / "real_validation_preparation.json"
    report["evidence_path"] = str(evidence_path)
    atomic_write_json(evidence_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = prepare(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "kind": "v5_real_run_preparation",
            "status": "error",
            "success": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        exit_code = 2
    else:
        exit_code = 0
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
