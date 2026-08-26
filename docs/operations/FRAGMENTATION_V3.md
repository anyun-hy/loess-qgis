# Fragmentation V3.3 production stage

V3.3 is the class-aware authoritative-raster stage for new Fusion Runs. Each
Work Package first applies frozen V3 to its Fusion probability Halo and stores
the V3 baseline Core, context, and probability. After every Work Package is
ready, one V3.3 job adjudicates the frozen owner Cores and publishes the final
non-overlapping Core masks. Core, Seam, and Junction units read only those
V3.3 masks before the one vectorization and common-boundary fitting pass.

## Data contract

- Fusion Halo probabilities are read-only and remain the source for confidence
  statistics.
- `fusion/<profile>/raster_parts/*_mask.tif` is the V3.3 authoritative Core
  classification; its GeoTIFF tags record the policy, version, production
  replacement state, actual context, and changed-pixel count.
- Spatial units do not run `argmax` or fragmentation repair.
- `semantic_polygons.gpkg` is assembled once and becomes `review_polygons`.
- The v5 runner does not invoke the historical postprocess script.

The production policy is `fragmentation_v33_configurable_absorption_v1`; its
complete class permissions, area ceilings, enclosure routing, rarity order,
bridge rules, budgets, conflict order, and hard gates are frozen in
`fragmentation_policy/policies/v33.yaml`. The first-stage baseline remains
`semantic_optimized_200_v3_core_bounded_v1`.

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

`fragmentation_regularization` in `inference_scripts/config.yaml` selects V3.3
and records V3 as its baseline in each new `run_spec.json`. The configured
Partition Halo must be at least the policy buffer. Work Packages publish the
frozen inputs; the singleton V3.3 job then atomically publishes one mask and
one audit per Core plus the global report. Dependent Fusion spatial units stay
blocked until that report is ready. After one vectorization/fitting/assembly
pass, the runner records the formal GeoPackage as the review source and
continues directly to scale acceptance.

To roll back a newly created Run, explicitly set `policy_id` to
`semantic_optimized_200_v3`. A running V3.3 Run never falls back silently.

If an older `classes/` workspace exists, the plugin never replaces it silently.
It offers a V3 rebuild only when all 14 class records are unmodified and
unconfirmed, no SAM session is active, the workspace is unlocked, and
`edit_history.jsonl` is empty. The user must confirm the rebuild in the QGIS
dialog.
