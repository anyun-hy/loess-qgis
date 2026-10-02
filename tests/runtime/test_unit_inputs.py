from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import loess_runtime.geometry.boundary_fitting.unit_inputs as unit_inputs
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.boundary_fitting.unit_errors import UnitRuntimeError
from loess_runtime.geometry.boundary_fitting.unit_inputs import (
    load_unit_fit_inputs,
    read_unit_probabilities,
)

RUN_ID = "run-1"
STREAM_ID = "fusion:test"


class FakeArtifacts:
    def __init__(
        self,
        values: Mapping[tuple[str, str], Path],
        events: list[tuple[str, str, str]] | None = None,
    ) -> None:
        self._values = dict(values)
        self._events = events

    def artifact_for_stream_unit(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        kind: str,
        *,
        status: str = "ready",
    ) -> dict[str, Any] | None:
        assert run_id == RUN_ID
        assert stream_id == STREAM_ID
        assert status == "ready"
        if self._events is not None:
            self._events.append(("artifact", unit_id, kind))
        path = self._values.get((unit_id, kind))
        return {"path": str(path)} if path is not None else None


def _artifacts(
    values: Mapping[tuple[str, str], Path],
    events: list[tuple[str, str, str]] | None = None,
) -> ArtifactRepository:
    return cast(ArtifactRepository, FakeArtifacts(values, events))


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    dtype: str,
    nodata: int | float | None = None,
    scales: tuple[float, ...] | None = None,
) -> None:
    bands = values if values.ndim == 3 else values[None, :, :]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=dtype,
        crs="EPSG:3857",
        transform=from_origin(0, bands.shape[1], 1, 1),
        nodata=nodata,
    ) as destination:
        destination.write(bands.astype(dtype))
        if scales is not None:
            destination.scales = scales


def _unit(
    dependency_ids: list[str],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "unit_id": "unit-1",
        "dependency_ids": dependency_ids,
        "pixel_window": {"x0": 0, "y0": 0, "x1": width, "y1": height},
    }


def _lookup(
    partitions: Mapping[str, Mapping[str, Any]],
    events: list[tuple[str, str, str]] | None = None,
):
    def read_partition(run_id: str, partition_id: str):
        assert run_id == RUN_ID
        if events is not None:
            events.append(("partition", partition_id, ""))
        return partitions.get(partition_id)

    return read_partition


def test_full_inputs_preserve_dependency_query_order_and_values(tmp_path):
    first_probability = np.zeros((14, 2, 4), dtype=np.int16)
    second_probability = np.zeros((14, 2, 4), dtype=np.int16)
    first_probability[0] = 3
    second_probability[1] = 1
    probability_paths = [
        tmp_path / "p0_probability.tif",
        tmp_path / "p1_probability.tif",
    ]
    for path, values in zip(
        probability_paths,
        (first_probability, second_probability),
    ):
        _write_raster(path, values, dtype="int16", scales=(1.0,) * 14)
    core_paths = [tmp_path / "p0_core.tif", tmp_path / "p1_core.tif"]
    _write_raster(core_paths[0], np.zeros((2, 2), dtype=np.int16), dtype="int16")
    _write_raster(core_paths[1], np.ones((2, 2), dtype=np.int16), dtype="int16")
    partitions = {
        "p0": {
            "halo_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 2},
            "core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
        },
        "p1": {
            "halo_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 2},
            "core_window": {"x0": 2, "y0": 0, "x1": 4, "y1": 2},
        },
    }
    events: list[tuple[str, str, str]] = []
    artifacts = _artifacts(
        {
            ("p0", "partition_probability"): probability_paths[0],
            ("p1", "partition_probability"): probability_paths[1],
            ("p0", "core_mask"): core_paths[0],
            ("p1", "core_mask"): core_paths[1],
        },
        events,
    )

    result = load_unit_fit_inputs(
        artifacts,
        _lookup(partitions, events),
        RUN_ID,
        STREAM_ID,
        _unit(["p0", "p1"], width=4, height=2),
        compact_confidence=False,
    )

    assert events == [
        ("partition", "p0", ""),
        ("artifact", "p0", "partition_probability"),
        ("partition", "p1", ""),
        ("artifact", "p1", "partition_probability"),
        ("partition", "p0", ""),
        ("artifact", "p0", "core_mask"),
        ("partition", "p1", ""),
        ("artifact", "p1", "core_mask"),
    ]
    assert result.labels.dtype == np.int16
    assert result.labels.tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert result.valid_mask.dtype == np.bool_
    assert np.all(result.valid_mask)
    assert result.confidence.dtype == np.float32
    assert np.all(result.confidence == np.float32(0.75))


def test_full_inputs_release_probability_bands_before_reading_core(monkeypatch):
    probability_reference: weakref.ReferenceType[np.ndarray] | None = None

    def read_probabilities(*_args):
        nonlocal probability_reference
        probabilities = np.zeros((14, 2, 3), dtype=np.float32)
        probabilities[0] = 0.75
        probabilities[1] = 0.25
        probability_reference = weakref.ref(probabilities)
        return probabilities, np.ones((2, 3), dtype=bool)

    def read_labels(*_args):
        gc.collect()
        assert probability_reference is not None
        assert probability_reference() is None
        return (
            np.zeros((2, 3), dtype=np.int16),
            np.ones((2, 3), dtype=bool),
        )

    monkeypatch.setattr(unit_inputs, "read_unit_probabilities", read_probabilities)
    monkeypatch.setattr(
        unit_inputs,
        "_read_unit_authoritative_labels",
        read_labels,
    )

    result = load_unit_fit_inputs(
        cast(ArtifactRepository, object()),
        lambda _run_id, _partition_id: None,
        RUN_ID,
        STREAM_ID,
        _unit(["p0"], width=3, height=2),
        compact_confidence=False,
    )

    assert np.all(result.confidence == np.float32(0.75))


@pytest.mark.parametrize(
    ("band_count", "scales", "message"),
    [
        (13, (1.0,) * 13, "unexpected shape"),
        (14, (1.0,) * 13 + (0.0,), "scale metadata is invalid"),
    ],
)
def test_probability_raster_rejects_band_and_scale_contracts(
    tmp_path,
    band_count,
    scales,
    message,
):
    path = tmp_path / "probability.tif"
    _write_raster(
        path,
        np.ones((band_count, 2, 3), dtype=np.int16),
        dtype="int16",
        scales=scales,
    )
    partitions = {
        "p0": {
            "halo_window": {"x0": 0, "y0": 0, "x1": 3, "y1": 2},
        }
    }

    with pytest.raises(UnitRuntimeError, match=message):
        read_unit_probabilities(
            _artifacts({("p0", "partition_probability"): path}),
            _lookup(partitions),
            RUN_ID,
            STREAM_ID,
            _unit(["p0"], width=3, height=2),
        )


def test_probability_dependency_requires_partition_artifact_and_halo(tmp_path):
    path = tmp_path / "probability.tif"
    _write_raster(
        path,
        np.ones((14, 2, 2), dtype=np.int16),
        dtype="int16",
        scales=(1.0,) * 14,
    )
    unit = _unit(["p0"], width=2, height=2)
    with pytest.raises(UnitRuntimeError, match="no Partition dependencies"):
        read_unit_probabilities(
            _artifacts({}),
            _lookup({}),
            RUN_ID,
            STREAM_ID,
            _unit([], width=2, height=2),
        )

    with pytest.raises(UnitRuntimeError, match="dependency is missing"):
        read_unit_probabilities(
            _artifacts({}),
            _lookup({}),
            RUN_ID,
            STREAM_ID,
            unit,
        )

    outside = {"p0": {"halo_window": {"x0": 1, "y0": 0, "x1": 3, "y1": 2}}}
    with pytest.raises(UnitRuntimeError, match="outside Partition Halo"):
        read_unit_probabilities(
            _artifacts({("p0", "partition_probability"): path}),
            _lookup(outside),
            RUN_ID,
            STREAM_ID,
            unit,
        )


def test_compact_confidence_and_core_share_the_same_valid_domain(tmp_path):
    confidence_path = tmp_path / "confidence.tif"
    core_path = tmp_path / "core.tif"
    confidence = np.array([[-1.0, 0.8], [0.6, 0.7]], dtype=np.float32)
    labels = np.array([[-1, 1], [2, 3]], dtype=np.int16)
    _write_raster(
        confidence_path,
        confidence,
        dtype="float32",
        nodata=-1.0,
    )
    _write_raster(core_path, labels, dtype="int16", nodata=-1)
    partition = {
        "p0": {
            "core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
        }
    }

    result = load_unit_fit_inputs(
        _artifacts(
            {
                ("unit-1", "unit_confidence"): confidence_path,
                ("p0", "core_mask"): core_path,
            }
        ),
        _lookup(partition),
        RUN_ID,
        STREAM_ID,
        _unit(["p0"], width=2, height=2),
        compact_confidence=True,
    )

    assert np.array_equal(result.labels, labels)
    assert np.array_equal(result.confidence, confidence)
    assert result.valid_mask.tolist() == [[False, True], [True, True]]


@pytest.mark.parametrize(
    ("confidence", "dtype", "message"),
    [
        (np.ones((2, 2), dtype=np.int16), "int16", "contract is invalid"),
        (np.array([[0.1, 1.1], [0.2, 0.3]], dtype=np.float32), "float32", "outside"),
        (np.ones((1, 2), dtype=np.float32), "float32", "unexpected shape"),
    ],
)
def test_compact_confidence_rejects_dtype_range_and_shape(
    tmp_path,
    confidence,
    dtype,
    message,
):
    confidence_path = tmp_path / "confidence.tif"
    _write_raster(confidence_path, confidence, dtype=dtype)
    with pytest.raises(UnitRuntimeError, match=message):
        load_unit_fit_inputs(
            _artifacts({("unit-1", "unit_confidence"): confidence_path}),
            _lookup({}),
            RUN_ID,
            STREAM_ID,
            _unit([], width=2, height=2),
            compact_confidence=True,
        )


def test_authoritative_core_rejects_conflicting_overlap(tmp_path):
    confidence_path = tmp_path / "confidence.tif"
    first_path = tmp_path / "first.tif"
    second_path = tmp_path / "second.tif"
    _write_raster(
        confidence_path,
        np.full((2, 2), 0.8, dtype=np.float32),
        dtype="float32",
    )
    _write_raster(first_path, np.zeros((2, 2), dtype=np.int16), dtype="int16")
    _write_raster(second_path, np.ones((2, 2), dtype=np.int16), dtype="int16")
    partitions = {
        key: {"core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}}
        for key in ("p0", "p1")
    }
    same = load_unit_fit_inputs(
        _artifacts(
            {
                ("unit-1", "unit_confidence"): confidence_path,
                ("p0", "core_mask"): first_path,
                ("p1", "core_mask"): first_path,
            }
        ),
        _lookup(partitions),
        RUN_ID,
        STREAM_ID,
        _unit(["p0", "p1"], width=2, height=2),
        compact_confidence=True,
    )
    assert np.all(same.labels == 0)

    with pytest.raises(UnitRuntimeError, match="overlapping.*disagree"):
        load_unit_fit_inputs(
            _artifacts(
                {
                    ("unit-1", "unit_confidence"): confidence_path,
                    ("p0", "core_mask"): first_path,
                    ("p1", "core_mask"): second_path,
                }
            ),
            _lookup(partitions),
            RUN_ID,
            STREAM_ID,
            _unit(["p0", "p1"], width=2, height=2),
            compact_confidence=True,
        )


def test_authoritative_core_rejects_gap_and_probability_coverage_mismatch(tmp_path):
    confidence_path = tmp_path / "confidence.tif"
    gap_core_path = tmp_path / "gap_core.tif"
    full_core_path = tmp_path / "full_core.tif"
    _write_raster(
        confidence_path,
        np.array([[-1.0, 0.8], [0.7, 0.6]], dtype=np.float32),
        dtype="float32",
        nodata=-1.0,
    )
    _write_raster(gap_core_path, np.ones((2, 1), dtype=np.int16), dtype="int16")
    _write_raster(full_core_path, np.ones((2, 2), dtype=np.int16), dtype="int16")
    artifacts = _artifacts(
        {
            ("unit-1", "unit_confidence"): confidence_path,
            ("p0", "core_mask"): gap_core_path,
        }
    )
    with pytest.raises(UnitRuntimeError, match="coverage gap"):
        load_unit_fit_inputs(
            artifacts,
            _lookup({"p0": {"core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 2}}}),
            RUN_ID,
            STREAM_ID,
            _unit(["p0"], width=2, height=2),
            compact_confidence=True,
        )

    with pytest.raises(UnitRuntimeError, match="probability coverage is absent"):
        load_unit_fit_inputs(
            _artifacts(
                {
                    ("unit-1", "unit_confidence"): confidence_path,
                    ("p0", "core_mask"): full_core_path,
                }
            ),
            _lookup({"p0": {"core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}}}),
            RUN_ID,
            STREAM_ID,
            _unit(["p0"], width=2, height=2),
            compact_confidence=True,
        )
