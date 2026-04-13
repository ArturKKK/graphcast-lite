#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE DA + MOS EXPERIMENTS — merge dataset
# All results on merge-mode multires dataset (real-world scenario)
# VM: MLC graphcast_v2-la3ti6, A100-SXM4-80GB
# Date: 14 Apr 2026
# ═══════════════════════════════════════════════════════════════

set -e
cd /workdir/graphcast-lite
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workdir/graphcast-lite
PY=".venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
BASE_FLAGS="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42"
LOG_DIR="/data/merge_results"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/batch_master.log"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MASTER_LOG"; }

run_oi() {
    local sparsity=$1 corr=$2 sigma=$3 tag=$4
    local log_file="$LOG_DIR/oi_s${tag}_c${corr}_sig${sigma}.log"
    log "START OI sparsity=$sparsity corr=$corr sigma=$sigma → $log_file"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma > "$log_file" 2>&1
    # Extract key metrics
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  OI s=$sparsity c=$corr σ=$sigma | $skill | $region"
}

run_nudge() {
    local sparsity=$1 alpha=$2 mode=$3 tag=$4
    local log_file="$LOG_DIR/nudge_s${tag}_a${alpha}_${mode}.log"
    log "START Nudge sparsity=$sparsity alpha=$alpha mode=$mode → $log_file"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method nudging \
        --nudging-alpha $alpha --nudging-mode $mode > "$log_file" 2>&1
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  Nudge s=$sparsity α=$alpha $mode | $skill | $region"
}

run_oi_channels() {
    local sparsity=$1 corr=$2 sigma=$3 channels=$4 label=$5
    local log_file="$LOG_DIR/oi_ch_${label}_s${sparsity}.log"
    log "START OI channels=$channels sparsity=$sparsity → $log_file"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma \
        --obs-channels "$channels" > "$log_file" 2>&1
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  OI ch=$label | $skill | $region"
}

log "═══════════════════════════════════════════════════"
log "COMPREHENSIVE EXPERIMENTS — merge dataset"
log "═══════════════════════════════════════════════════"

# ═══ PART 0: SANITY (baseline, no DA) ═══
log "=== PART 0: Baseline sanity check ==="
$PY scripts/predict.py $EXP --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    > "$LOG_DIR/sanity.log" 2>&1
log "DONE  Sanity: $(grep 'Skill:' $LOG_DIR/sanity.log | head -1)"

# ═══ PART 1: OI — 10% stations — corr_len sweep (σ=0.5) ═══
log "=== PART 1: OI 10% corr_len sweep ==="
run_oi 0.1 10000   0.5  01
run_oi 0.1 25000   0.5  01
run_oi 0.1 50000   0.5  01
run_oi 0.1 100000  0.5  01
run_oi 0.1 150000  0.5  01
run_oi 0.1 200000  0.5  01
run_oi 0.1 300000  0.5  01
run_oi 0.1 500000  0.5  01

# ═══ PART 2: OI — 10% — sigma sweep ═══
log "=== PART 2: OI 10% sigma sweep ==="
run_oi 0.1 10000  0.3  01
run_oi 0.1 10000  1.0  01
run_oi 0.1 50000  0.3  01
run_oi 0.1 50000  1.0  01
run_oi 0.1 100000 0.3  01
run_oi 0.1 100000 1.0  01

# ═══ PART 3: OI — 1% stations — extended corr_len sweep ═══
log "=== PART 3: OI 1% corr_len sweep (extended) ==="
run_oi 0.01 10000   0.5  001
run_oi 0.01 50000   0.5  001
run_oi 0.01 100000  0.5  001
run_oi 0.01 150000  0.5  001
run_oi 0.01 200000  0.5  001
run_oi 0.01 300000  0.5  001
run_oi 0.01 500000  0.5  001

# ═══ PART 4: Nudging — 10% ═══
log "=== PART 4: Nudging 10% ==="
run_nudge 0.1  0.3  sequential  01
run_nudge 0.1  0.5  sequential  01
run_nudge 0.1  0.7  sequential  01
run_nudge 0.1  0.3  offline     01

# ═══ PART 5: Nudging — 1% ═══
log "=== PART 5: Nudging 1% ==="
run_nudge 0.01 0.3  sequential  001
run_nudge 0.01 0.5  sequential  001

# ═══ PART 6: Variable groups (OI c=10km σ=0.5 10%) ═══
log "=== PART 6: Variable groups ==="
run_oi_channels 0.1 10000 0.5 "t2m"                    "t2m_only"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v"            "t_wind"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl"        "surface"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,t@850,t@500"  "surface_tup"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,tp,sp,tcwv,t@850,u@850,v@850,z@850,q@850,t@500,u@500,v@500,z@500,q@500" "all_dynamic"

# ═══ PART 7: MOS/IDW Post-processing Sweep ═══
log "=== PART 7: MOS/IDW sweep ==="

# Check if global and regional datasets exist
GLOBAL_DIR="/data/data/datasets/wb2_512x256_19f_ar"
REGION_DIR="/data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
MOS_MODEL="live_runtime_bundle/learned_mos_t2m_19stations.joblib"

# Fallback: check alternative MOS model paths
if [ ! -f "$MOS_MODEL" ]; then
    MOS_MODEL="live_runtime_bundle/learned_mos_t2m.joblib"
fi

if [ -d "$GLOBAL_DIR" ] && [ -d "$REGION_DIR" ] && [ -f "$MOS_MODEL" ]; then
    $PY scripts/mos_idw_sweep.py \
        --gnn-exp $EXP \
        --global-dir "$GLOBAL_DIR" \
        --region-dir "$REGION_DIR" \
        --learned-mos "$MOS_MODEL" \
        --ar-steps 4 --max-samples 50 \
        > "$LOG_DIR/mos_idw_sweep.log" 2>&1
    log "DONE  MOS/IDW sweep"
else
    log "SKIP  MOS/IDW sweep — missing datasets or MOS model"
    [ ! -d "$GLOBAL_DIR" ] && log "  Missing: $GLOBAL_DIR"
    [ ! -d "$REGION_DIR" ] && log "  Missing: $REGION_DIR"
    [ ! -f "$MOS_MODEL" ] && log "  Missing: $MOS_MODEL"
fi

log "═══════════════════════════════════════════════════"
log "ALL EXPERIMENTS COMPLETED at $(date)"
log "═══════════════════════════════════════════════════"

# ═══ Summary: extract all key metrics ═══
log ""
log "═══ SUMMARY TABLE ═══"
for f in "$LOG_DIR"/*.log; do
    [ "$f" = "$MASTER_LOG" ] && continue
    name=$(basename "$f" .log)
    skill=$(grep "Skill:" "$f" 2>/dev/null | head -1 | sed 's/.*Skill: //')
    region6=$(grep "+06h:" "$f" 2>/dev/null | tail -1)
    if [ -n "$skill" ]; then
        log "  $name | Skill: $skill | $region6"
    fi
done
