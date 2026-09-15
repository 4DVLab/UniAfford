#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python code/utils/rendering/compare_prediction_csv_renders.py \
  results/GEAL-gen/validation_samples.csv \
  "results/GREAT replicate/prediction_id_mapping.csv" \
  results/IAGNet/prediction_id_mapping.csv \
  --dataset-root datasets/9_GEAL \
  --output-dir outputs/3d-compare \
  --modality point \
  --method-names Ours,GREAT,IAGNet \
  --gt-threshold-3d 0.5 \
  --thresholds-3d 0.25 0.5 0.5 \
  --point-backend fast \
  --num-workers 16 \
  --skip-existing
