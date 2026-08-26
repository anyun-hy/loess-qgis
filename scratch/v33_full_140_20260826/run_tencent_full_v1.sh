#!/usr/bin/env bash
set -euo pipefail

v33_run_root=/home/example/Desktop/loess-project/scratch/v33_full_140_20260826/run_20260826_v33_v1
v33_code_root=${v33_run_root}/code_snapshot
v33_output_root=${v33_run_root}/out/full_140_v33
v33_evaluation_root=${v33_run_root}/out/comparison
v31a_manifest=/home/example/Desktop/loess-project/scratch/v31a_full_140_20260824/run_20260824_full_v1/out/full_140/run_manifest.json
v31b_manifest=/home/example/Desktop/loess-project/scratch/v31b_full_140_20260825/run_20260825_b_v2/out/full_140_b/run_manifest.json
v32_manifest=/home/example/Desktop/loess-project/scratch/v32_full_140_20260825/run_20260825_v32_v1/out/full_140_v32/run_manifest.json

resume_args=()
if [[ ${1:-} == --resume ]]; then
  resume_args=(--resume)
fi

conda run -n qgis python \
  "${v33_code_root}/scratch/v33_full_140_20260826/run_v33_from_v3.py" \
  --v31a-manifest "${v31a_manifest}" \
  --output-root "${v33_output_root}" \
  --workers 2 \
  "${resume_args[@]}"

conda run -n qgis python \
  "${v33_code_root}/scratch/v33_full_140_20260826/evaluate_v33.py" \
  --b-manifest "${v31b_manifest}" \
  --v32-manifest "${v32_manifest}" \
  --v33-manifest "${v33_output_root}/run_manifest.json" \
  --output-root "${v33_evaluation_root}"
