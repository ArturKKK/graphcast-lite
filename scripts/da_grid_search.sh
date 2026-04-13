#!/bin/bash
# DA Grid Search: Nudging + OI on freeze6 multires model
# Run from /workdir/graphcast-lite with LD_LIBRARY_PATH set

set -e
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
cd /workdir/graphcast-lite

PY=".venv/bin/python"
EXP="experiments/multires_nores_freeze6"
RESULTS="/data/predictions/da_results.txt"

echo "=== DA Grid Search ===" | tee "$RESULTS"
echo "Date: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ── NUDGING ──
for SPARSITY in 0.01 0.1; do
  for ALPHA in 0.01 0.05 0.1 0.3; do
    echo "=== NUDGING sparsity=$SPARSITY alpha=$ALPHA ===" | tee -a "$RESULTS"
    $PY scripts/predict.py $EXP \
      --no-residual --prune-mesh --ar-steps 4 --per-channel \
      --max-samples 200 --no-save \
      --assim-method nudging --nudging-alpha $ALPHA \
      --obs-sparsity $SPARSITY --obs-roi-only \
      2>&1 | grep -E "(Skill|RMSE|ACC|Region|Per-horizon|^\s+\+)" | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
  done
done

# ── OI ──
for SPARSITY in 0.01 0.1; do
  for CORR_LEN in 5000 10000 50000; do
    for SIGMA_O in 0.3 0.5; do
      echo "=== OI sparsity=$SPARSITY corr_len=$CORR_LEN sigma_o=$SIGMA_O ===" | tee -a "$RESULTS"
      $PY scripts/predict.py $EXP \
        --no-residual --prune-mesh --ar-steps 4 --per-channel \
        --max-samples 200 --no-save \
        --assim-method oi --oi-corr-len $CORR_LEN --oi-sigma-o $SIGMA_O \
        --obs-sparsity $SPARSITY --obs-roi-only \
        2>&1 | grep -E "(Skill|RMSE|ACC|Region|Per-horizon|^\s+\+)" | tee -a "$RESULTS"
      echo "" | tee -a "$RESULTS"
    done
  done
done

echo "=== ALL DA DONE at $(date) ===" | tee -a "$RESULTS"
