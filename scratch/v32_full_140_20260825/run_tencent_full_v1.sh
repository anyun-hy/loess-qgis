#!/usr/bin/env bash
set -euo pipefail

v32_run_root=/home/example/Desktop/loess-project/scratch/v32_full_140_20260825/run_20260825_v32_v1
v32_code_root=${v32_run_root}/code_snapshot
v32_output_root=${v32_run_root}/out/full_140_v32
v32_evaluation_root=${v32_run_root}/out/comparison
v31a_manifest=/home/example/Desktop/loess-project/scratch/v31a_full_140_20260824/run_20260824_full_v1/out/full_140/run_manifest.json
v31b_manifest=/home/example/Desktop/loess-project/scratch/v31b_full_140_20260825/run_20260825_b_v2/out/full_140_b/run_manifest.json

resume_args=()
if [[ ${1:-} == --resume ]]; then
  resume_args=(--resume)
fi

conda run -n qgis python \
  "${v32_code_root}/scratch/v32_full_140_20260825/run_v32_from_v3.py" \
  --v31a-manifest "${v31a_manifest}" \
  --output-root "${v32_output_root}" \
  --workers 2 \
  "${resume_args[@]}"

conda run -n qgis python \
  "${v32_code_root}/scratch/v32_full_140_20260825/evaluate_v32_against_b.py" \
  --b-manifest "${v31b_manifest}" \
  --v32-manifest "${v32_output_root}/run_manifest.json" \
  --output-root "${v32_evaluation_root}"
