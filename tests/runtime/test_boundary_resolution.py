import copy
import math

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates

from labeling_tool.shared.contracts.boundary_resolution import (
    normalize_resolution_policy,
)
from loess_runtime.geometry.boundary_fitting.resolution import (
    resolve_boundary_parameters,
)
from loess_runtime.geometry.boundary_fitting.unit_runtime import _smoothing_config
from loess_runtime.geometry.polyline_smoother import smooth_polyline


def _boundary():
    return {
        "smoothing_factor": 1.0,
        "curve_sampling_spacing_px": 0.5,
        "max_chord_error_px": 0.25,
        "max_segment_arc_length_px": 8.0,
        "max_deviation_px": 1.0,
        "resolution_adaptation": normalize_resolution_policy(),
    }


def _resolve(resolution, *, boundary=None, crs="EPSG:32649", transform=None, extent=None):
    return resolve_boundary_parameters(
        _boundary() if boundary is None else boundary,
        raster={"crs": crs, "transform": transform or [resolution, 0, 0, 0, -resolution, 1000]},
        processing_extent=extent or {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000},
    )


@pytest.mark.parametrize("resolution", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_reference_distances_adapt_without_relaxing_pixel_error_caps(resolution):
    config, report = _resolve(resolution)
    assert config["max_deviation_px"] == min(1.0, 2.0 / resolution)
    assert config["max_segment_arc_length_px"] * resolution == 16.0
    assert config["max_chord_error_px"] <= 0.25
    assert config["max_chord_error_px"] * resolution <= 0.5
    assert config["curve_sampling_spacing_px"] <= 0.5
    assert config["smoothing_factor"] * resolution <= 2.0
    assert report["max_deviation_m"] <= 2.0


def test_reference_resolution_preserves_existing_parameters_and_frozen_input():
    boundary = _boundary()
    original = copy.deepcopy(boundary)
    config, report = _resolve(2.0, boundary=boundary)
    assert config == original
    assert boundary == original
    assert report["scale_factor"] == 1.0


@pytest.mark.parametrize("boundary", [{"max_deviation_px": 0.7}, {
    "max_deviation_px": 0.7, "resolution_adaptation": {"enabled": False},
}])
def test_old_or_disabled_runs_do_not_require_or_reinterpret_raster_units(boundary):
    resolved, report = resolve_boundary_parameters(boundary, raster={}, processing_extent={})
    assert resolved == boundary
    assert report is None or report == {"enabled": False}


@pytest.mark.parametrize("matrix", [
    [[0, -4], [0.25, 0]],  # rotated non-square pixels
    [[2, 1], [0, -1]],  # shear: longest axis alone underestimates worst direction
])
def test_non_square_and_sheared_pixels_use_the_worst_direction(matrix):
    matrix = np.asarray(matrix, dtype=float)
    resolved, report = _resolve(1, transform=[*matrix[0], 0, *matrix[1], 1000])
    expected = np.linalg.svd(matrix, compute_uv=False)[0]
    assert report["maximum_pixel_scale_m"] == pytest.approx(expected)
    directions = np.column_stack((np.cos(np.linspace(0, 2*math.pi, 1000)), np.sin(np.linspace(0, 2*math.pi, 1000))))
    moved = directions @ matrix.T * resolved["max_deviation_px"]
    assert np.linalg.norm(moved, axis=1).max() <= 2 + 1e-10


def test_projected_feet_are_converted_to_meters():
    crs = CRS.from_epsg(2277)
    _, report = _resolve(10, crs=crs.to_wkt())
    assert report["maximum_pixel_scale_m"] == pytest.approx(10 * crs.linear_units_factor[1])
    assert report["max_deviation_m"] == pytest.approx(2.0)


def test_geographic_degrees_use_ellipsoid_distances():
    resolved, report = _resolve(
        0.00001, crs="EPSG:4326", transform=[0.00001, 0, 110, 0, -0.00001, 40.1],
        extent={"xmin": 110, "xmax": 110.1, "ymin": 39.9, "ymax": 40.1},
    )
    assert 1.1 < report["maximum_pixel_scale_m"] < 1.12
    assert report["metric_basis"] == "geographic_ground_scale_bound"
    assert resolved["max_deviation_px"] == 1.0
    # On this UTM central meridian, undo the known 0.9996 scale to check
    # against an independent PROJ coordinate transformation.
    _, northings = transform_coordinates("EPSG:4326", "EPSG:32649", [111, 111], [40, 40.00001])
    assert abs(northings[1] - northings[0]) / 0.9996 <= report["maximum_pixel_scale_m"]


def test_web_mercator_corrects_its_latitude_scale():
    _, report = _resolve(
        2.0, crs="EPSG:3857", transform=[2, 0, 12000000, 0, -2, 4550000],
        extent={"xmin": 12000000, "xmax": 12001000, "ymin": 4540000, "ymax": 4550000},
    )
    assert report["metric_basis"] == "web_mercator_ground_scale_bound"
    assert 1.5 < report["maximum_pixel_scale_m"] < 1.7


@pytest.mark.parametrize("resolution", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_same_physical_outline_obeys_both_limits_at_multiple_resolutions(resolution):
    # Controlled continuous geometry in metres, not a claim about predictions
    # made from real imagery at these resolutions.
    x = np.linspace(0, 128, 513)
    source_m = np.column_stack((x, 4 * np.sin(x / 15)))
    resolved, _ = _resolve(resolution)
    result = smooth_polyline(source_m / resolution, _smoothing_config(resolved))
    assert result.status == "smoothed"
    assert result.max_deviation_upper_bound <= resolved["max_deviation_px"]
    assert result.max_deviation_upper_bound * resolution <= 2.0
    assert result.output_point_count < len(source_m)
    assert result.max_segment_arc_length * resolution <= 16 + 1e-9


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True])
def test_invalid_reference_resolution_is_rejected(invalid):
    with pytest.raises(ValueError, match="reference_resolution_m"):
        normalize_resolution_policy({"reference_resolution_m": invalid})


def test_invalid_affine_and_angular_extent_are_rejected():
    with pytest.raises(ValueError, match="zero pixel area"):
        _resolve(1, transform=[1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="pole"):
        _resolve(1, crs="EPSG:4326")
