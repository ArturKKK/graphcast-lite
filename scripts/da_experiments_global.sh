#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# DA + MOS EXPERIMENTS — GLOBAL 512×256 dataset (CORRECT setup)
#
# nores_freeze6 обучался на 512×256 графе → тестируем на нём же.
# Наблюдения вводим только в ROI (--obs-roi-only --region 50 60 83 98).
# Метрики региона смотрим через --region 50 60 83 98 (294 узла @ 0.7°).
#
# VM: MLC graphcast_v2-vqpgo2, A100-SXM4-80GB
# Date: Apr 2026
# ═══════════════════════════════════════════════════════════════

set -e
cd /workdir/graphcast-lite
export PYTHONPATH=/workdir/graphcast-lite
PY="/data/venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
GLOBAL="/data/datasets/wb2_512x256_19f_ar"
BASE_FLAGS="--data-dir $GLOBAL --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 50 60 83 98 --obs-roi-only --obs-seed 42"
LOG_DIR="/data/da_global_results"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/batch_master.log"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MASTER_LOG"; }

run_oi() {
    local sparsity=$1 corr=$2 sigma=$3 tag=$4
    local log_file="$LOG_DIR/oi_s${tag}_c${corr}_sig${sigma}.log"
    log "START OI sparsity=$sparsity corr=$corr sigma=$sigma → $(basename $log_file)"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma > "$log_file" 2>&1
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  OI s=$sparsity c=$corr σ=$sigma | $skill | region: $region"
}

run_nudge() {
    local sparsity=$1 alpha=$2 mode=$3 tag=$4
    local log_file="$LOG_DIR/nudge_s${tag}_a${alpha}_${mode}.log"
    log "START Nudge sparsity=$sparsity alpha=$alpha mode=$mode"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method nudging \
        --nudging-alpha $alpha --nudging-mode $mode > "$log_file" 2>&1
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  Nudge s=$sparsity α=$alpha $mode | $skill | region: $region"
}

run_oi_channels() {
    local sparsity=$1 corr=$2 sigma=$3 channels=$4 label=$5
    local log_file="$LOG_DIR/oi_ch_${label}_s${sparsity}.log"
    log "START OI channels=$label sparsity=$sparsity"
    $PY scripts/predict.py $EXP $BASE_FLAGS \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma \
        --obs-channels "$channels" > "$log_file" 2>&1
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log "DONE  OI ch=$label | $skill | region: $region"
}

log "═══════════════════════════════════════════════════"
log "DA EXPERIMENTS — GLOBAL 512x256 dataset"
log "ROI: 50-60N x 83-98E | 294 nodes @ 0.7deg"
log "═══════════════════════════════════════════════════"

# ═══ PART 0: Baseline (no DA) ═══
log "=== PART 0: Baseline (no DA) ==="
$PY scripts/predict.py $EXP $BASE_FLAGS \
    > "$LOG_DIR/baseline.log" 2>&1
log "DONE  Baseline: $(grep 'Skill:' $LOG_DIR/baseline.log | head -1)"

# ═══ PART 1: OI 10% станций — sweep по corr_len (σ=0.5) ═══
log "=== PART 1: OI 10% corr_len sweep (sigma=0.5) ==="
run_oi 0.1 10000   0.5  01
run_oi 0.1 25000   0.5  01
run_oi 0.1 50000   0.5  01
run_oi 0.1 100000  0.5  01
run_oi 0.1 150000  0.5  01
run_oi 0.1 200000  0.5  01
run_oi 0.1 300000  0.5  01
run_oi 0.1 500000  0.5  01

# ═══ PART 2: OI 10% — sigma sweep ═══
log "=== PART 2: OI 10% sigma sweep ==="
run_oi 0.1 10000  0.3  01
run_oi 0.1 10000  1.0  01
run_oi 0.1 50000  0.3  01
run_oi 0.1 50000  1.0  01
run_oi 0.1 100000 0.3  01
run_oi 0.1 100000 1.0  01

# ═══ PART 3: OI 1% — sweep по corr_len ═══
log "=== PART 3: OI 1% corr_len sweep ==="
run_oi 0.01 10000   0.5  001
run_oi 0.01 50000   0.5  001
run_oi 0.01 100000  0.5  001
run_oi 0.01 150000  0.5  001
run_oi 0.01 200000  0.5  001
run_oi 0.01 300000  0.5  001
run_oi 0.01 500000  0.5  001

# ═══ PART 4: Nudging 10% ═══
log "=== PART 4: Nudging 10% ==="
run_nudge 0.1  0.3  sequential  01
run_nudge 0.1  0.5  sequential  01
run_nudge 0.1  0.7  sequential  01
run_nudge 0.1  0.3  offline     01

# ═══ PART 5: Nudging 1% ═══
log "=== PART 5: Nudging 1% ==="
run_nudge 0.01 0.3  sequential  001
run_nudge 0.01 0.5  sequential  001

# ═══ PART 6: Variable groups (OI c=10km σ=0.5 10%) ═══
log "=== PART 6: Variable groups ==="
run_oi_channels 0.1 10000 0.5 "t2m"                                                                                         "t2m_only"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v"                                                                                 "t_wind"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl"                                                                             "surface"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,t@850,t@500"                                                                "surface_tup"
run_oi_channels 0.1 10000 0.5 "t2m,10u,10v,msl,tp,sp,tcwv,t@850,u@850,v@850,z@850,q@850,t@500,u@500,v@500,z@500,q@500"    "all_dynamic"

log "═══════════════════════════════════════════════════"
log "ALL EXPERIMENTS COMPLETED at $(date)"
log "═══════════════════════════════════════════════════"

# ═══ Summary ═══
log ""
log "═══ SUMMARY TABLE ==="
log "Experiment | Global Skill | Region +6h Skill | Region +6h ACC"
for f in "$LOG_DIR"/*.log; do
    [ "$f" = "$MASTER_LOG" ] && continue
    name=$(basename "$f" .log)
    gs=$(grep "^Skill:" "$f" 2>/dev/null | head -1 | awk '{print $2}')
    rs=$(grep "+06h:" "$f" 2>/dev/null | tail -1 | grep -o 'skill=[^|]*' | head -1)
    ra=$(grep "+06h:" "$f" 2>/dev/null | tail -1 | grep -o 'ACC=[^ ]*' | head -1)
    log "  $name | $gs | $rs | $ra"
done
