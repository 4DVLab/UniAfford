python utils/rendering/compare_prediction_csv_renders.py \
  run_a/validation_samples.csv \
  run_b/validation_samples.csv\
  --dataset-root path/to/dataset \
  --output-dir outputs/compare_ab \
  --method-names A,B \
  --thresholds-2d 0.35,0.50 \
  --thresholds-3d 0.40,0.55 \
  --modality both