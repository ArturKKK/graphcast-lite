#!/usr/bin/env bash
# DA experiments on multires (133K nodes) dataset
# Model: multires_nores_freeze6
set -euo pipefail

VENV=/data/venv/bin/activate
source "$VENV"
cd /workdir/graphcast-lite

EXP=experiments/multires_nores_freeze6
DATA=/data/datasets/multires_krsk_19f
OUT=/data/da_multires_results
mkdir -p "$OUT"

COMMON="--data-dir $DATA --no-residual --region 50 60 83 98 --prune-mesh --ar-steps 4 --obs-seed 42 --obs-roi-only --max-samples 200"

run_exp() {
    local name="$1"; shift
    local logfile="$OUT/${name}.log"
    echo "$(date '+%H:%M:%S') >>> $name"
    python scripts/predict.py $EXP $COMMON "$@" > "$logfile" 2>&1
    # Extract key metrics
    grep -E "Full ACC|Region ACC|City ACC|RMSE" "$logfile" | tail -20 >> "$OUT/summary.txt"
    echo "--- $name ---" >> "$OUT/summary.txt"
    echo "$(date '+%H:%M:%S') <<< $name done"
}

echo "=== DA Multires Experiments ===" > "$OUT/summary.txt"
date >> "$OUT/summary.txt"

# ─── BASELINE ───
echo "$(date '+%H:%M:%S') ═══ BASELINE ═══"
run_exp "baseline" --assim-method none

# ─── OI 10% ───
echo "$(date '+%H:%M:%S') ═══ OI 10% ═══"
for corr in 50 100 150 300; do
    for sigma in 0.3 0.5 1.0; do
        run_exp "oi10_c${corr}_s${sigma}" \
            --assim-method oi --obs-sparsity 0.1 \
            --oi-corr-len "$corr" --oi-sigma-o "$sigma"
    done
done

# ─── OI 1% ───
echo "$(date '+%H:%M:%S') ═══ OI 1% ═══"
for corr in 50 100 150 300; do
    for sigma in 0.3 0.5 1.0; do
        run_exp "oi1_c${corr}_s${sigma}" \
            --assim-method oi --obs-sparsity 0.01 \
            --oi-corr-len "$corr" --oi-sigma-o "$sigma"
    done
done

# ─── NUDGING 10% ───
echo "$(date '+%H:%M:%S') ═══ NUDGING 10% ═══"
for alpha in 0.1 0.3 0.5 0.7; do
    run_exp "nudg10_a${alpha}" \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha"
done

# ─── NUDGING 1% ───
echo "$(date '+%H:%M:%S') ═══ NUDGING 1% ═══"
for alpha in 0.1 0.3 0.5 0.7; do
    run_exp "nudg1_a${alpha}" \
        --assim-method nudging --obs-sparsity 0.01 \
        --nudging-alpha "$alpha"
done

echo ""
echo "$(date '+%H:%M:%S') ═══ ALL DONE ═══"
echo "Results in $OUT"
cat "$OUT/summary.txt"
