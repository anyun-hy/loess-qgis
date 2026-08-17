import json

from labeling_tool.core import result_catalog
from labeling_tool.core.run_spec import sha256_file


def _write_run(root, run_id, *, workspace=False, status="ready"):
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    stream = {
        "stream_id": "fusion:l2_fusion_v1",
        "kind": "fusion",
        "status": "ready",
    }
    (run_dir / "run_spec.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "run_id": run_id,
            "run_spec": str(run_dir / "run_spec.json"),
            "status": status,
            "streams": [stream],
        }),
        encoding="utf-8",
    )
    if workspace:
        (run_dir / "classes").mkdir()
        (run_dir / "classes" / "workspace.json").write_text("{}", encoding="utf-8")


def test_discover_ready_results_prefers_resumable_workspace(tmp_path, monkeypatch):
    _write_run(tmp_path, "20260716_100000_newer")
    _write_run(tmp_path, "20260715_100000_workspace", workspace=True)
    _write_run(tmp_path, "20260717_100000_failed", status="failed")
    monkeypatch.setattr(
        result_catalog,
        "valid_ready_stream_ids",
        lambda catalog: tuple(item["stream_id"] for item in catalog["streams"]),
    )

    discovered = result_catalog.discover_ready_results(tmp_path)

    assert [result["run_id"] for result, _spec in discovered] == [
        "20260715_100000_workspace",
        "20260716_100000_newer",
    ]
    assert discovered[0][0]["success"] is True
    assert discovered[0][0]["ready_streams"][0]["kind"] == "fusion"


def test_vrt_statistics_metadata_does_not_invalidate_an_output(tmp_path):
    path = tmp_path / "confidence_mosaic.vrt"
    original = (
        b'<VRTDataset rasterXSize="1" rasterYSize="1">\n'
        b'  <VRTRasterBand dataType="Float32" band="1">\n'
        b'    <ColorInterp>Gray</ColorInterp>\n'
        b'  </VRTRasterBand>\n'
        b'</VRTDataset>\n'
    )
    path.write_bytes(original)
    expected = result_catalog.artifact_sha256(path)

    path.write_bytes(original.replace(
        b"    <ColorInterp>Gray</ColorInterp>",
        b"    <Metadata>\n"
        b'      <MDI key="STATISTICS_MINIMUM">0.2</MDI>\n'
        b'      <MDI key="STATISTICS_MAXIMUM">1</MDI>\n'
        b"    </Metadata>\n"
        b"    <ColorInterp>Gray</ColorInterp>",
    ))

    assert result_catalog.artifact_sha256(path) == expected


def test_vrt_non_statistics_metadata_remains_part_of_integrity_hash(tmp_path):
    path = tmp_path / "mask_mosaic.vrt"
    path.write_text("<VRTDataset/>\n", encoding="utf-8")
    original = result_catalog.artifact_sha256(path)
    path.write_text(
        '<VRTDataset>\n  <Metadata>\n'
        '    <MDI key="CLASS_ORDER">12,13</MDI>\n'
        '  </Metadata>\n</VRTDataset>\n',
        encoding="utf-8",
    )
    assert result_catalog.artifact_sha256(path) != original


def test_v2_ready_stream_requires_derived_review_checksum(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(
        json.dumps({"schema_version": 2, "models": []}), encoding="utf-8"
    )
    paths = {}
    for name in (
        "mask_mosaic",
        "confidence_mosaic",
        "semantic_polygons_raw",
        "semantic_polygons",
        "fitted_edges",
    ):
        path = run_dir / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = str(path)
    report_path = run_dir / "boundary_fitting_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "validation": {"passed": True},
                "input_sha256": sha256_file(paths["semantic_polygons_raw"]),
                "output_sha256": sha256_file(paths["semantic_polygons"]),
            }
        ),
        encoding="utf-8",
    )
    paths["boundary_fitting_report"] = str(report_path)
    review = run_dir / "derived_review.gpkg"
    review.write_bytes(b"derived-review")
    checksums = {
        key: result_catalog.artifact_sha256(path) for key, path in paths.items()
    }
    checksums["review_polygons"] = result_catalog.artifact_sha256(review)
    catalog = {
        "schema_version": 2,
        "run_spec": str(spec_path),
        "run_spec_sha256": sha256_file(spec_path),
        "streams": [
            {
                "stream_id": "fusion:test",
                "status": "ready",
                "boundary_fitting_status": "passed",
                "paths": paths,
                "review_polygons": str(review),
                "output_sha256": checksums,
            }
        ],
    }

    assert result_catalog.valid_ready_stream_ids(catalog) == ("fusion:test",)
    review.write_bytes(b"tampered")
    assert result_catalog.valid_ready_stream_ids(catalog) == ()
