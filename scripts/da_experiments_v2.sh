#!/bin/bash
# Comprehensive DA experiments for diploma — v2 (corrected)
# ALL experiments use --no-residual (freeze6 model)
# Runs sequentially, one at a time, logging to /data/v2_*.log

set -e
cd /workdir/graphcast-lite
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
PY=".venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
BASE_FLAGS="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42"

run_oi() {
    local sparsity=$1 corr=$2 sigma=$3 tag=$4
    local log="/data/v2_oi_s${tag}_c${corr}_sig${sigma}.log"
    echo "=== OI sparsity=$sparsity corr=$corr sigma=$sigma → $log ==="
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma > "$log" 2>&1
    grep "Skill:" "$log" | head -1
    grep "+06h:" "$log" | tail -1
}

run_nudge() {
    local sparsity=$1 alpha=$2 mode=$3 tag=$4
    local log="/data/v2_nudge_s${tag}_a${alpha}_${mode}.log"
    echo "=== Nudge sparsity=$sparsity alpha=$alpha mode=$mode → $log ==="
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method nudging \
        --nudging-alpha $alpha --nudging-mode $mode > "$log" 2>&1
    grep "Skill:" "$log" | head -1
    grep "+06h:" "$log" | tail -1
}

run_oi_channels() {
    local sparsity=$1 corr=$2 sigma=$3 channels=$4 label=$5
    local log="/data/v2_oi_ch_${label}_s${sparsity}.log"
    echo "=== OI channels=$channels sparsity=$sparsity → $log ==="
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma \
        --obs-channels "$channels" > "$log" 2>&1
    grep "Skill:" "$log" | head -1
    grep "+06h:" "$log" | tail -1
}

echo "========================================"
echo "Starting DA experiments v2 at $(date)"
echo "========================================"

# === PART 1: Sanity check (20 samples, no DA) ===
echo "--- Sanity check ---"
$PY scripts/predict.py $EXP --max-samples 20 --ar-steps 4 --per-channel --no-residual > /data/v2_sanity.log 2>&1
grep "Skill:" /data/v2_sanity.log | head -1

# === PART 2: OI — sparsity 10% (s=0.1) — corr sweep ===
run_oi 0.1 10000  0.5  01    # corr=10km σ=0.5 (reproduce previous best)
run_oi 0.1 25000  0.5  01    # corr=25km
run_oi 0.1 50000  0.5  01    # corr=50km
run_oi 0.1 100000 0.5  01    # corr=100km
run_oi 0.1 150000 0.5  01    # corr=150km

# === PART 3: OI — sparsity 10% — sigma sweep (best corr from P2) ===
run_oi 0.1 10000  0.3  01    # reproduce previous
run_oi 0.1 10000  1.0  01    # higher sigma
run_oi 0.1 50000  0.3  01    # corr=50km lower sigma
run_oi 0.1 50000  1.0  01    # corr=50km higher sigma
run_oi 0.1 100000 0.3  01    # corr=100km lower sigma
run_oi 0.1 100000 1.0  01    # corr=100km higher sigma

# === PART 4: OI — sparsity 1% (s=0.01) ===
run_oi 0.01 10000  0.5  001
run_oi 0.01 50000  0.5  001
run_oi 0.01 100000 0.5  001
run_oi 0.01 150000 0.5  001

# === PART 5: Nudging sweep — 10% ===
run_nudge 0.1  0.3  sequential  01
run_nudge 0.1  0.5  sequential  01
run_nudge 0.1  0.7  sequential  01
run_nudge 0.1  0.3  offline     01

# === PART 6: Nudging — 1% ===
run_nudge 0.01 0.3  sequential  001
run_nudge 0.01 0.5  sequential  001

# === PART 7: Variable groups (OI best config: corr=10km σ=0.5 s=0.1) ===
# Group 1: temperature only
run_oi_channels 0.1 10000 0.5 "t2m"                    "t2m_only"
# Group 2: temperature + wind (cheap sensors)
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v"            "t_wind"
# Group 3: standard surface station (t2m + wind + pressure)
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl"        "surface"
# Group 4: surface + upper-air temperature
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,t@850,t@500"  "surface_tup"
# Group 5: all dynamic (exclude static z_surf, lsm)
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,tp,sp,tcwv,t@850,u@850,v@850,z@850,q@850,t@500,u@500,v@500,z@500,q@500" "all_dynamic"

echo "========================================"
echo "ALL DA experiments v2 completed at $(date)"
echo "========================================"
