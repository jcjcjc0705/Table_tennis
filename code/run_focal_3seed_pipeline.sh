#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) Train 3 focal-loss models (seeds 42/123/456)
# 2) Ensemble inference (argmax)
# 3) Class-aware inference with adaptive alpha

# Python selection priority:
# 1) explicit PYTHON_BIN env var
# 2) active conda env python
# 3) python3 on PATH
# 4) /usr/bin/python3 fallback
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="/usr/bin/python3"
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "==> Detected conda env: ${CONDA_DEFAULT_ENV}"
fi
echo "==> Using Python: ${PYTHON_BIN}"
TRAIN="${TRAIN:-data/train.csv}"
TEST="${TEST:-data/test.csv}"
SAMPLE="${SAMPLE:-data/sample_submission.csv}"
ENCODER_OUT="${ENCODER_OUT:-output/encoders/encoder.pkl}"
OUT_DIR_MODELS="${OUT_DIR_MODELS:-output/models}"
OUT_DIR_SUBS="${OUT_DIR_SUBS:-output/submissions}"

HIDDEN="${HIDDEN:-192}"
LAYERS="${LAYERS:-3}"
DROP="${DROP:-0.3}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"
WARMUP="${WARMUP:-5}"

FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${FOCAL_ALPHA:-1.0}"

ALPHA_BASE="${ALPHA_BASE:-0.4}"
ALPHA_DEV_THRESHOLD="${ALPHA_DEV_THRESHOLD:-0.20}"
ALPHA_MAX_BOOST="${ALPHA_MAX_BOOST:-0.30}"

SEEDS=(42 123 456)

mkdir -p "$OUT_DIR_MODELS" "$OUT_DIR_SUBS" "$(dirname "$ENCODER_OUT")"

echo "==> Training focal-loss models"
for seed in "${SEEDS[@]}"; do
  MODEL_PATH="$OUT_DIR_MODELS/model.pt.focal_seed${seed}"
  RAW_OUT="$OUT_DIR_SUBS/submission_focal_seed${seed}_raw.csv"

  echo "---- seed=$seed"
  "$PYTHON_BIN" code/shuttleNet_code.py \
    --train "$TRAIN" \
    --test "$TEST" \
    --sample "$SAMPLE" \
    --encoder_out "$ENCODER_OUT" \
    --model_out "$MODEL_PATH" \
    --out "$RAW_OUT" \
    --hidden "$HIDDEN" \
    --layers "$LAYERS" \
    --drop "$DROP" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --warmup "$WARMUP" \
    --seed "$seed" \
    --use_focal_action \
    --focal_gamma "$FOCAL_GAMMA" \
    --focal_alpha "$FOCAL_ALPHA"
done

MODEL_LIST="$OUT_DIR_MODELS/model.pt.focal_seed42,$OUT_DIR_MODELS/model.pt.focal_seed123,$OUT_DIR_MODELS/model.pt.focal_seed456"

echo "==> Ensemble argmax inference"
"$PYTHON_BIN" code/shuttleNet_code.py \
  --mode infer \
  --test "$TEST" \
  --sample "$SAMPLE" \
  --encoder_out "$ENCODER_OUT" \
  --model_out "$MODEL_LIST" \
  --out "$OUT_DIR_SUBS/submission_focal_3seed_argmax.csv" \
  --hidden "$HIDDEN" \
  --layers "$LAYERS" \
  --drop "$DROP"

echo "==> Ensemble class-aware adaptive inference"
"$PYTHON_BIN" code/class_aware_infer.py \
  --train "$TRAIN" \
  --test "$TEST" \
  --sample "$SAMPLE" \
  --encoder_out "$ENCODER_OUT" \
  --model_out "$MODEL_LIST" \
  --out "$OUT_DIR_SUBS/submission_focal_3seed_classaware_adapt.csv" \
  --alpha "$ALPHA_BASE" \
  --adaptive_alpha \
  --alpha_deviation_threshold "$ALPHA_DEV_THRESHOLD" \
  --alpha_max_boost "$ALPHA_MAX_BOOST" \
  --hidden "$HIDDEN" \
  --layers "$LAYERS"

echo "==> Done"
echo "Argmax submission:      $OUT_DIR_SUBS/submission_focal_3seed_argmax.csv"
echo "Class-aware submission: $OUT_DIR_SUBS/submission_focal_3seed_classaware_adapt.csv"
