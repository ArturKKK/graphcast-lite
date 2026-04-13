#!/bin/bash
# DA experiments v3: extended corr_len for 1% + MOS/IDW parameter sweep
# 13 April 2026
set -e
cd /workdir/graphcast-lite
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
PY=".venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
BASE_FLAGS="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42"

run_oi() {
    local sparsity=$1 corr=$2 sigma=$3 tag=$4
    local log="/data/v3_oi_s${tag}_c${corr}_sig${sigma}.log"
    echo "=== OI sparsity=$sparsity corr=$corr sigma=$sigma → $log ==="
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma > "$log" 2>&1
    tail -20 "$log"
}

echo "========================================"
echo "Starting DA experiments v3 at $(date)"
echo "========================================"

# === PART 1: Extended corr_len for 1% stations ===
# At 1%, 150km was still improving (75.84%). Try 200, 300, 500km.
echo "--- Part 1: OI 1% extended corr_len ---"
run_oi 0.01 200000 0.5  001   # 200km
run_oi 0.01 300000 0.5  001   # 300km
run_oi 0.01 500000 0.5  001   # 500km

# === PART 2: Extended corr_len for 10% (confirm drop) ===
echo "--- Part 2: OI 10% extended corr_len ---"
run_oi 0.1 200000 0.5  01    # 200km
run_oi 0.1 300000 0.5  01    # 300km

# === PART 3: Larger corr_len with lower sigma (sweet spot?) ===
echo "--- Part 3: OI 1% c=200-300km с σ=0.3 ---"
run_oi 0.01 200000 0.3  001   # 200km σ=0.3
run_oi 0.01 300000 0.3  001   # 300km σ=0.3

echo "========================================"
echo "DA experiments v3 (Part 1-3) completed at $(date)"
echo "========================================"
