#!/usr/bin/env bash
# scripts/run_evaluation.sh
#
# Master post-training evaluation pipeline for multires Krasnoyarsk model.
# Run on the cluster after training completes.
#
# Usage:
#   bash scripts/run_evaluation.sh <experiment_dir> [data_dir]
#
# Examples:
#   bash scripts/run_evaluation.sh experiments/multires_nores_nofreeze
#   bash scripts/run_evaluation.sh experiments/multires_nores_freeze6 data/datasets/multires_krsk_19f
#
# What it does:
#   1. Inference → predictions.pt
#   2. Conference scatter maps (Truth/Prediction/Error)
#   3. WRF comparison table (if WRF data available)
#   4. Data Assimilation experiments (Nudging + OI)
#   5. Summary

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────
EXP_DIR="${1:?Usage: $0 <experiment_dir> [data_dir]}"
DATA_DIR="${2:-data/datasets/multires_krsk_19f}"
WRF_JSON="aaaa/wrf_krasnoyarsk/wrf_d03_jan2023.json"

# ─── Config ───────────────────────────────────────────────────────────
REGION="55.5 56.5 92 94"   # Krasnoyarsk city bbox
AR_STEPS=4
MAX_SAMPLES=200
PRED_FILE="${EXP_DIR}/predictions.pt"

echo "================================================================"
echo "  Post-Training Evaluation Pipeline"
echo "================================================================"
echo "  Experiment:  ${EXP_DIR}"
echo "  Dataset:     ${DATA_DIR}"
echo "  Region:      ${REGION}"
echo "  AR steps:    ${AR_STEPS}"
echo "================================================================"
echo ""

# ─── Step 1: Inference ────────────────────────────────────────────────
echo "━━━ STEP 1: Inference ━━━"
if [[ -f "${PRED_FILE}" ]]; then
    echo "  predictions.pt already exists, skipping inference."
    echo "  (Delete ${PRED_FILE} to re-run)"
else
    echo "  Running predict.py..."
    python scripts/predict.py "${EXP_DIR}" \
        --data-dir "${DATA_DIR}" \
        --region ${REGION} \
        --ar-steps ${AR_STEPS} \
        --per-channel \
        --no-residual \
        --max-samples ${MAX_SAMPLES}
fi
echo ""

# ─── Step 2: Conference visualizations ───────────────────────────────
echo "━━━ STEP 2: Conference visualizations ━━━"
PLOT_DIR="${EXP_DIR}/plots"
echo "  Generating scatter maps → ${PLOT_DIR}"
python scripts/plot_region_multires.py "${PRED_FILE}" \
    --data-dir "${DATA_DIR}" \
    --region ${REGION} \
    --out-dir "${PLOT_DIR}" \
    --vars t2m 10u 10v \
    --marker-size 18
echo ""

# Also generate plots for all horizons, all surface vars
echo "  Generating full variable set..."
python scripts/plot_region_multires.py "${PRED_FILE}" \
    --data-dir "${DATA_DIR}" \
    --region ${REGION} \
    --out-dir "${PLOT_DIR}/all_vars" \
    --vars t2m 10u 10v sp t@850 u@850 v@850 z@500 \
    --marker-size 18
echo ""

# ─── Step 3: WRF comparison ──────────────────────────────────────────
echo "━━━ STEP 3: WRF (ДВОРФ) comparison ━━━"
if [[ -f "${WRF_JSON}" ]]; then
    echo "  WRF data found: ${WRF_JSON}"
    echo "  NOTE: WRF covers Jan 20-21 2023. For correct comparison,"
    echo "        predictions.pt must be from a Jan 2023 test set."
    python scripts/compare_wrf.py \
        --predictions "${PRED_FILE}" \
        --data-dir "${DATA_DIR}" \
        --experiment-dir "${EXP_DIR}" \
        --ar-steps ${AR_STEPS}
else
    echo "  WRF data not found at ${WRF_JSON}, skipping."
    echo "  (Place wrf_d03_jan2023.json or pass --wrf-path to compare_wrf.py manually)"
fi
echo ""

# ─── Step 4: Data Assimilation (DA) ──────────────────────────────────
echo "━━━ STEP 4: Data Assimilation experiments ━━━"
DA_DIR="${EXP_DIR}/da_results"
mkdir -p "${DA_DIR}"

# 4a. Baseline (no DA) — already in predictions.pt, just note it
echo "  [Baseline] See STEP 1 results (no DA)"
echo ""

# 4b. Nudging (offline mode — works with any rollout)
echo "  [Nudging] Running offline nudging (α=0.25)..."
python scripts/predict.py "${EXP_DIR}" \
    --data-dir "${DATA_DIR}" \
    --region ${REGION} \
    --ar-steps ${AR_STEPS} \
    --per-channel \
    --no-residual \
    --max-samples ${MAX_SAMPLES} \
    --assim-method nudging \
    --nudging-mode offline \
    --nudging-alpha 0.25 \
    --save "${DA_DIR}/predictions_nudging_025.pt" 2>&1 | tail -30
echo ""

echo "  [Nudging] Running offline nudging (α=0.5)..."
python scripts/predict.py "${EXP_DIR}" \
    --data-dir "${DATA_DIR}" \
    --region ${REGION} \
    --ar-steps ${AR_STEPS} \
    --per-channel \
    --no-residual \
    --max-samples ${MAX_SAMPLES} \
    --assim-method nudging \
    --nudging-mode offline \
    --nudging-alpha 0.5 \
    --save "${DA_DIR}/predictions_nudging_050.pt" 2>&1 | tail -30
echo ""

# 4c. OI (Optimal Interpolation)
echo "  [OI] Running Optimal Interpolation..."
python scripts/predict.py "${EXP_DIR}" \
    --data-dir "${DATA_DIR}" \
    --region ${REGION} \
    --ar-steps ${AR_STEPS} \
    --per-channel \
    --no-residual \
    --max-samples ${MAX_SAMPLES} \
    --assim-method oi \
    --oi-sigma-b 0.8 \
    --oi-sigma-o 0.5 \
    --oi-corr-len 10000 \
    --save "${DA_DIR}/predictions_oi.pt" 2>&1 | tail -30
echo ""

# 4d. DA visualizations
echo "  [DA plots] Generating comparison maps..."
for da_file in "${DA_DIR}"/predictions_*.pt; do
    if [[ -f "${da_file}" ]]; then
        da_name=$(basename "${da_file}" .pt | sed 's/predictions_//')
        python scripts/plot_region_multires.py "${da_file}" \
            --data-dir "${DATA_DIR}" \
            --region ${REGION} \
            --out-dir "${DA_DIR}/plots_${da_name}" \
            --vars t2m 10u 10v \
            --marker-size 18
    fi
done
echo ""

# ─── Step 5: Summary ─────────────────────────────────────────────────
echo "━━━ STEP 5: Summary ━━━"
echo ""
echo "Artifacts produced:"
echo "  ${PRED_FILE}              — baseline predictions"
echo "  ${PLOT_DIR}/              — conference scatter maps"
echo "  ${DA_DIR}/                — DA experiment predictions"
echo "  ${DA_DIR}/plots_*/        — DA comparison maps"
echo ""
echo "Next steps:"
echo "  1. Review plots in ${PLOT_DIR}/ for conference slides"
echo "  2. Compare baseline vs DA RMSE improvements"
echo "  3. For WRF comparison on Jan 2023 event:"
echo "     - Build multires Jan 2023 dataset (build_multires_dataset.py on Jan 2023 data)"
echo "     - Re-run this pipeline with that dataset"
echo "================================================================"
echo "  DONE"
echo "================================================================"
