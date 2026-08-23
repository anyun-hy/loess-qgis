# Fragmentation V3 production stage

V3 is the bounded, class-aware authoritative-raster stage for new Fusion Runs.
Each Work Package applies the frozen policy to a Fusion probability Halo and
publishes only its non-overlapping cleaned Core mask. Core, Seam, and Junction
units read those masks as their sole class source before the one vectorization
and common-boundary fitting pass.

## Data contract

- Fusion Halo probabilities are read-only and remain the source for confidence
  statistics.
- `fusion/<profile>/raster_parts/*_mask.tif` is the cleaned authoritative Core
  classification; its GeoTIFF tags record the policy, version, actual context,
  and changed-pixel count.
- Spatial units do not run `argmax` or fragmentation repair.
- `semantic_polygons.gpkg` is assembled once and becomes `review_polygons`.
- The v5 runner does not invoke the historical postprocess script.

The frozen policy is `semantic_optimized_200_v3_core_bounded_v1`: a 200 m2
ceiling for eligible non-protected classes, 256-pixel context, semantic target
compatibility, a 0.65 confidence guard, 8% source/target class budgets, minimum
remaining class area, and elongated-feature preservation. Classes 12, 33, 61,
62, and 71 are protected from source reassignment.

## Historical completed-Run repair

The standalone command is retained only for Runs completed before the
authoritative-raster pipeline. It deliberately writes a derived result below
`run_dir/postprocess` and does not participate in new v5 Runs.

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

## New Runs

`fragmentation_regularization` in `inference_scripts/config.yaml` freezes the
same policy into new `run_spec.json` files. The configured Partition Halo must
be at least the policy buffer. Work Packages publish cleaned Core masks before
publishing the probability Artifact that releases dependent spatial units.
After one vectorization/fitting/assembly pass, the runner records the formal
GeoPackage as the review source and continues directly to scale acceptance.

If an older `classes/` workspace exists, the plugin never replaces it silently.
It offers a V3 rebuild only when all 14 class records are unmodified and
unconfirmed, no SAM session is active, the workspace is unlocked, and
`edit_history.jsonl` is empty. The user must confirm the rebuild in the QGIS
dialog.
