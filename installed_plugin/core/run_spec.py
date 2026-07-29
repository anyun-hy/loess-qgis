"""Create immutable run specifications for semantic inference jobs.

The module has no QGIS imports.  Extents are accepted through a small
duck-typed interface so run creation can be tested outside QGIS.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_SPEC_SCHEMA_VERSION = 1
DEFAULT_TILE_OVERLAP = 192
MOSAIC_STRATEGY = "cosine_probability_blend"
RAW_VECTORIZATION_METHOD = "rasterio_features_shapes"
FORMAL_VECTORIZATION_METHOD = "multiclass_subpixel_probability_v1"
BOUNDARY_REGULARIZATION_DEFAULT = {
    "enabled": True,
    "mode": FORMAL_VECTORIZATION_METHOD,
    "interpolation_strength": 1.0,
    "probability_smoothing_sigma": 0.0,
    "coverage_tolerance_px": 1.0,
    "max_deviation_px": 1.5,
    "stripe_rows": 128,
    "qsdk_noninferiority_margin_px": 0.5,
    "preserve_outer_boundary": True,
    "natural_smoothing": False,
}
CLASS_ORDER = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
CLASS_NAMES = {
    12: "水浇地",
    13: "旱地",
    21: "果园",
    31: "有林地",
    32: "灌木林地",
    33: "其他林地",
    43: "其他草地",
    51: "城镇建设用地",
    52: "农村建设用地",
    53: "人为扰动用地",
    54: "其他建设用地",
    61: "农村道路",
    62: "其他交通用地",
    71: "河湖库塘",
}
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
RESERVATION_FILE = ".run-reservation"
FUSION_STRATEGIES = {
    "equal_probability_average",
    "calibrated_global_weighted",
    "calibrated_class_weighted",
    "linear_1x1",
}
APPROVAL_CRITERION = "fusion.test_miou > exported_swin_baseline.test_miou"


class RunSpecError(ValueError):
    pass


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def new_run_id(now: _datetime.datetime | None = None, token: str | None = None) -> str:
    moment = now or _datetime.datetime.now()
    suffix = token or secrets.token_hex(3)
    if not re.fullmatch(r"[a-zA-Z0-9]{4,16}", suffix):
        raise RunSpecError("run id token must contain 4-16 letters or digits")
    return f"{moment.strftime('%Y%m%d_%H%M%S')}_{suffix.lower()}"


def reserve_run_directory(
    output_root: os.PathLike[str] | str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """Reserve an isolated run directory before QGIS starts extracting tiles."""
    output = Path(output_root).expanduser().resolve()
    identifier = run_id or new_run_id()
    if not re.fullmatch(r"\d{8}_\d{6}_[a-z0-9]{4,16}", identifier):
        raise RunSpecError(f"invalid run_id: {identifier!r}")
    run_dir = output / "runs" / identifier
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunSpecError(f"run directory already exists; refusing to overwrite: {run_dir}") from exc
    for relative in (
        "models", "fusion", "classes", "refinement/sam3", "final", "logs", "tmp/tiles",
        "tmp/streams", "tmp/work_packages", "tmp/probability_parts",
        "tmp/unit_outputs", "tmp/failed_jobs",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / RESERVATION_FILE, {
        "run_id": identifier,
        "created_at": _datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    return identifier, run_dir


def _extent_dict(extent: Any) -> dict[str, float]:
    if isinstance(extent, Mapping):
        aliases = {
            "xmin": ("xmin", "x_min"),
            "ymin": ("ymin", "y_min"),
            "xmax": ("xmax", "x_max"),
            "ymax": ("ymax", "y_max"),
        }
        result = {}
        for target, names in aliases.items():
            value = next((extent[name] for name in names if name in extent), None)
            if value is None:
                raise RunSpecError(f"extent is missing {target}")
            result[target] = float(value)
    elif isinstance(extent, Sequence) and not isinstance(extent, (str, bytes)):
        if len(extent) != 4:
            raise RunSpecError("extent sequence must contain xmin,ymin,xmax,ymax")
        result = dict(zip(("xmin", "ymin", "xmax", "ymax"), map(float, extent)))
    else:
        try:
            result = {
                "xmin": float(extent.xMinimum()),
                "ymin": float(extent.yMinimum()),
                "xmax": float(extent.xMaximum()),
                "ymax": float(extent.yMaximum()),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise RunSpecError("unsupported extent value") from exc
    if result["xmin"] >= result["xmax"] or result["ymin"] >= result["ymax"]:
        raise RunSpecError("extent must have positive width and height")
    return result


def _model_dict(model: Any) -> dict[str, Any]:
    def get(name: str, default: Any = "") -> Any:
        if isinstance(model, Mapping):
            return model.get(name, default)
        return getattr(model, name, default)

    model_id = str(get("model_id")).strip()
    if not SAFE_ID_RE.fullmatch(model_id):
        raise RunSpecError(f"invalid model_id: {model_id!r}")
    artifact_path = Path(str(get("artifact_path"))).expanduser().resolve()
    if not artifact_path.is_file():
        raise RunSpecError(f"model artifact does not exist: {artifact_path}")
    expected_sha = str(get("sha256")).lower()
    actual_sha = sha256_file(artifact_path)
    if expected_sha != actual_sha:
        raise RunSpecError(
            f"model artifact SHA256 mismatch for {model_id}: expected {expected_sha}, got {actual_sha}"
        )
    return {
        "model_id": model_id,
        "display_name": str(get("display_name", model_id) or model_id),
        "version": str(get("version") or ""),
        "artifact": str(get("artifact", artifact_path.name) or artifact_path.name),
        "artifact_path": str(artifact_path),
        "sha256": actual_sha,
    }


def _tile_dict(tile: Mapping[str, Any]) -> dict[str, Any]:
    row = int(tile.get("row", 0))
    col = int(tile.get("col", 0))
    tile_id = str(tile.get("tile_id") or f"{row}_{col}")
    if not re.fullmatch(r"-?\d+_-?\d+", tile_id):
        raise RunSpecError(f"invalid tile_id: {tile_id!r}")
    tile_path = Path(str(tile.get("path") or tile.get("tile_path") or "")).expanduser().resolve()
    if not tile_path.is_file():
        raise RunSpecError(f"tile does not exist: {tile_path}")
    result = {
        "tile_id": tile_id,
        "row": row,
        "col": col,
        "path": str(tile_path),
        "sha256": sha256_file(tile_path),
        "width": int(tile.get("width", 512)),
        "height": int(tile.get("height", 512)),
    }
    if result["width"] != 512 or result["height"] != 512:
        raise RunSpecError(f"tile {tile_id} must be declared as 512x512")
    bounds = tile.get("bounds")
    if bounds is not None:
        result["bounds"] = _extent_dict(bounds)
    return result


def _validate_fusion_for_run(profile: Mapping[str, Any], models: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check runtime-critical profile fields again before freezing a run."""
    errors = []
    if profile.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if profile.get("strategy") not in FUSION_STRATEGIES:
        errors.append("strategy is unsupported")
    if profile.get("class_order") != CLASS_ORDER:
        errors.append("class_order must equal the configured 14-class order")
    expected_input = {"height": 512, "width": 512, "channels": 3, "dtype": "float32"}
    if profile.get("input") != expected_input:
        errors.append("input contract must be 3x512x512 float32")
    approval = profile.get("approval") or {}
    if profile.get("status") != "approved" or approval.get("passed") is not True:
        errors.append("profile must be approved")
    if approval.get("criterion") != APPROVAL_CRITERION:
        errors.append("approval criterion is unsupported")

    selected = {str(item["model_id"]): item for item in models}
    entries = profile.get("models") or []
    ids = [str(item.get("model_id") or "") for item in entries if isinstance(item, Mapping)]
    if not entries or len(ids) != len(entries) or len(set(ids)) != len(ids):
        errors.append("models must contain unique model records")
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model_id = str(entry.get("model_id") or "")
        model = selected.get(model_id)
        if model is None:
            errors.append(f"model is not selected: {model_id}")
            continue
        if entry.get("artifact") != model.get("artifact") or entry.get("sha256") != model.get("sha256"):
            errors.append(f"model artifact identity does not match registry: {model_id}")
        temperature = entry.get("temperature")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            errors.append(f"model temperature must be finite and positive: {model_id}")

    weights = profile.get("weights") or []
    if len(weights) != len(CLASS_ORDER):
        errors.append("weights must contain 14 rows")
    else:
        for index, row in enumerate(weights):
            valid = (
                isinstance(row, list)
                and len(row) == len(entries)
                and all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in row)
                and math.isclose(sum(float(value) for value in row), 1.0, abs_tol=1e-6, rel_tol=0)
            )
            if not valid:
                errors.append(f"weights row {index} is invalid")
    return errors


def create_run_spec(
    *,
    output_root: os.PathLike[str] | str,
    raster_path: os.PathLike[str] | str,
    raster_crs: str,
    requested_extent: Any,
    processing_extent: Any,
    tiles: Sequence[Mapping[str, Any]],
    skipped_tiles: Sequence[Mapping[str, Any]] = (),
    models: Sequence[Any],
    effective_device: str,
    keep_score_cache: bool = False,
    overlap: int = DEFAULT_TILE_OVERLAP,
    boundary_regularization: Mapping[str, Any] | None = None,
    accepted_gpkg: os.PathLike[str] | str = "",
    skip_accepted: bool = True,
    fusion_profile_path: os.PathLike[str] | str | None = None,
    config_fingerprint: str = "",
    run_id: str | None = None,
    reserved_run_dir: os.PathLike[str] | str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Create one new run directory and atomically persist its specification."""
    if not models:
        raise RunSpecError("at least one model is required")
    if not tiles:
        raise RunSpecError("at least one extracted tile is required")
    if int(overlap) <= 0 or int(overlap) >= 512:
        raise RunSpecError("overlap must be between 1 and 511 for probability blending")
    regularization = dict(BOUNDARY_REGULARIZATION_DEFAULT)
    if boundary_regularization is not None:
        regularization.update(dict(boundary_regularization))
    if regularization.get("enabled") is not True:
        raise RunSpecError("boundary regularization must be enabled")
    if regularization.get("mode") != FORMAL_VECTORIZATION_METHOD:
        raise RunSpecError(
            f"boundary regularization mode must equal {FORMAL_VECTORIZATION_METHOD}"
        )
    if regularization.get("preserve_outer_boundary") is not True:
        raise RunSpecError("boundary regularization must preserve the outer boundary")
    if regularization.get("natural_smoothing") is not False:
        raise RunSpecError("natural smoothing is not supported")
    for key in (
        "interpolation_strength",
        "coverage_tolerance_px",
        "max_deviation_px",
        "qsdk_noninferiority_margin_px",
    ):
        value = float(regularization.get(key, 0))
        if not math.isfinite(value) or value <= 0:
            raise RunSpecError(f"boundary regularization {key} must be finite and positive")
        regularization[key] = value
    if not math.isclose(regularization["interpolation_strength"], 1.0, abs_tol=1e-12):
        raise RunSpecError("formal interpolation_strength must equal 1.0")
    if not math.isclose(regularization["coverage_tolerance_px"], 1.0, abs_tol=1e-12):
        raise RunSpecError("formal coverage_tolerance_px must equal 1.0")
    if float(regularization.get("probability_smoothing_sigma", -1)) != 0.0:
        raise RunSpecError("formal probability_smoothing_sigma must equal 0.0")
    regularization["stripe_rows"] = int(regularization.get("stripe_rows", 0))
    if regularization["stripe_rows"] < 1:
        raise RunSpecError("boundary regularization stripe_rows must be positive")
    output = Path(output_root).expanduser().resolve()
    if reserved_run_dir is None:
        identifier, run_dir = reserve_run_directory(output, run_id)
    else:
        run_dir = Path(reserved_run_dir).expanduser().resolve()
        identifier = run_id or run_dir.name
        expected = output / "runs" / identifier
        if run_dir != expected:
            raise RunSpecError(f"reserved run directory must equal {expected}")
        reservation = run_dir / RESERVATION_FILE
        if not reservation.is_file() or (run_dir / "run_spec.json").exists():
            raise RunSpecError("reserved run directory is missing its unused reservation marker")

    model_values = [_model_dict(model) for model in models]
    if len({item["model_id"] for item in model_values}) != len(model_values):
        raise RunSpecError("model ids must be unique")
    tile_values = [_tile_dict(tile) for tile in tiles]
    if len({item["tile_id"] for item in tile_values}) != len(tile_values):
        raise RunSpecError("tile ids must be unique")

    fusion = None
    if fusion_profile_path:
        profile_source = Path(fusion_profile_path).expanduser().resolve()
        if not profile_source.is_file():
            raise RunSpecError(f"fusion profile does not exist: {profile_source}")
        with open(profile_source, "r", encoding="utf-8") as handle:
            profile = json.load(handle)
        validation_errors = _validate_fusion_for_run(profile, model_values)
        if validation_errors:
            raise RunSpecError("invalid fusion profile: " + "; ".join(validation_errors))
        profile_id = str(profile.get("profile_id") or "")
        if not SAFE_ID_RE.fullmatch(profile_id):
            raise RunSpecError(f"invalid fusion profile_id: {profile_id!r}")
        required_ids = [str(item.get("model_id")) for item in profile.get("models") or []]
        missing = sorted(set(required_ids) - {item["model_id"] for item in model_values})
        if missing:
            raise RunSpecError(f"fusion profile models are not selected: {missing}")
        snapshot_path = run_dir / "fusion_profile_snapshot.json"
        atomic_write_json(snapshot_path, profile)
        fusion = {
            "profile_id": profile_id,
            "strategy": str(profile.get("strategy") or ""),
            "source_path": str(profile_source),
            "snapshot_path": str(snapshot_path),
            "sha256": sha256_file(snapshot_path),
            "required_model_ids": required_ids,
        }

    class_snapshot = {
        "class_mapping": {str(code): CLASS_NAMES[code] for code in CLASS_ORDER},
        "index_to_code": {str(index): code for index, code in enumerate(CLASS_ORDER)},
        "background_index": -1,
    }
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(class_path, class_snapshot)
    config_snapshot = {
        "schema_version": 2,
        "config_fingerprint": str(config_fingerprint or ""),
        "models": model_values,
        "fusion": fusion,
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
        },
        "mosaic": {
            "strategy": MOSAIC_STRATEGY,
            "overlap": int(overlap),
        },
        "vectorization": {
            "method": FORMAL_VECTORIZATION_METHOD,
            "raw_method": RAW_VECTORIZATION_METHOD,
        },
        "boundary_regularization": regularization,
    }
    atomic_write_json(run_dir / "config_snapshot.json", config_snapshot)

    raster = Path(raster_path).expanduser().resolve()
    spec = {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "run_id": identifier,
        "created_at": _datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_root": str(output),
        "raster": {"path": str(raster), "crs": str(raster_crs)},
        "requested_extent": _extent_dict(requested_extent),
        "processing_extent": _extent_dict(processing_extent),
        "tile": {
            "width": 512,
            "height": 512,
            "overlap": int(overlap),
            "mosaic_strategy": MOSAIC_STRATEGY,
        },
        "vectorization": {
            "method": FORMAL_VECTORIZATION_METHOD,
            "raw_method": RAW_VECTORIZATION_METHOD,
        },
        "boundary_regularization": regularization,
        "accepted_gpkg": str(Path(accepted_gpkg).expanduser().resolve()) if accepted_gpkg else "",
        "skip_accepted": bool(skip_accepted),
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
        },
        "models": model_values,
        "fusion": fusion,
        "class_mapping_snapshot": str(class_path),
        "tiles": tile_values,
        "skipped_tiles": [
            {
                "tile_id": str(tile.get("tile_id") or f"{int(tile.get('row', 0))}_{int(tile.get('col', 0))}"),
                "row": int(tile.get("row", 0)),
                "col": int(tile.get("col", 0)),
                "reason": str(tile.get("skip_reason") or "fully_accepted"),
            }
            for tile in skipped_tiles
        ],
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    try:
        (run_dir / RESERVATION_FILE).unlink()
    except FileNotFoundError:
        pass
    return spec, spec_path


def load_run_spec(path: os.PathLike[str] | str) -> dict[str, Any]:
    source = Path(path).resolve()
    with open(source, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != RUN_SPEC_SCHEMA_VERSION:
        raise RunSpecError("run_spec schema_version must equal 1")
    if Path(value.get("run_dir") or "").resolve() != source.parent:
        raise RunSpecError("run_spec run_dir does not match its location")
    return value
