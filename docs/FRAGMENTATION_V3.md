# Fragmentation V3 production stage

V3 is a bounded, class-aware postprocess for Fusion output. It is deliberately
separate from GPU inference: it consumes committed partition mask/confidence
rasters and writes a derived review result below the Run directory.

## Data contract

- Source data is read-only: `fusion/<profile>/raster_parts/*_mask.tif` and
  `*_confidence.tif`.
- Derived data is written to
  `postprocess/semantic_optimized_200_v3/fusion/<profile>/`.
- Original `semantic_polygons.gpkg`, boundary-fitting outputs, `classes/`,
  weights, inputs, and accepted labels are not replaced.
- The Run manifest receives `review_polygons` only after all partitions, the
  final GeoPackage integrity check, class-presence checks, protected-class
  checks, and SHA256 checks pass.

The frozen policy is `semantic_optimized_200_v3_core_bounded_v1`: a 200 m2
ceiling for eligible non-protected classes, 256-pixel context, semantic target
compatibility, a 0.65 confidence guard, 8% source/target class budgets, minimum
remaining class area, and elongated-feature preservation. Classes 12, 33, 61,
62, and 71 are protected from source reassignment.

## Completed Run resume

```bash
./inference_scripts/run_fragmentation_postprocess.sh \
  --run-spec /path/to/run/run_spec.json \
  --stream-id fusion:l2_fusion_v1 \
  --workers 4 \
  --buffer-pixels 256 \
  --activate-review
```

Partition masks, partition vectors, and reports are resumable. A passed final
manifest is reused only when the Run spec, source mask/confidence inventory,
report, and final GeoPackage still match their recorded fingerprints.

## Future Runs

`fragmentation_regularization` in `inference_scripts/config.yaml` freezes the
same policy into new `run_spec.json` files. The v5 runner invokes the stage
after stream assembly and before scale acceptance. The runner then records the
validated derived GeoPackage as the Fusion `review_polygons` source.

If an older `classes/` workspace exists, the plugin never replaces it silently.
It offers a V3 rebuild only when all 14 class records are unmodified and
unconfirmed, no SAM session is active, the workspace is unlocked, and
`edit_history.jsonl` is empty. The user must confirm the rebuild in the QGIS
dialog.
