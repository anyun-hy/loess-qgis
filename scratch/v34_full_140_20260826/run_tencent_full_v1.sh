#!/usr/bin/env bash
set -euo pipefail

v34_run_root=/home/example/Desktop/loess-project/scratch/v34_full_140_20260826/run_20260826_v34_v1
v34_code_root=${v34_run_root}/code_snapshot
v34_output_root=${v34_run_root}/out/full_140_v34
v34_evaluation_root=${v34_run_root}/out/comparison
v33_manifest=/home/example/Desktop/loess-project/scratch/v33_full_140_20260826/run_20260826_v33_v1/out/full_140_v33/run_manifest.json

resume_args=()
if [[ ${1:-} == --resume ]]; then
  resume_args=(--resume)
fi

conda run -n qgis python \
  "${v34_code_root}/scratch/v34_full_140_20260826/run_v34_from_v33.py" \
  --v33-manifest "${v33_manifest}" \
  --output-root "${v34_output_root}" \
  --workers 2 \
  "${resume_args[@]}"

conda run -n qgis python \
  "${v34_code_root}/scratch/v34_full_140_20260826/evaluate_v34.py" \
  --v33-manifest "${v33_manifest}" \
  --v34-manifest "${v34_output_root}/run_manifest.json" \
  --output-root "${v34_evaluation_root}"
