#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VM SETUP: venv + datasets + multires build + DA experiments
# Run inside tmux on VM: tmux new-session -d -s setup "bash /workdir/graphcast-lite/scripts/vm_setup_and_da.sh"
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
LOG="/data/setup_master.log"
mkdir -p /data
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "═══ VM SETUP START ═══"

# ═══ STEP 1: Python venv ═══
if [ ! -f /data/venv/bin/python ]; then
    log "=== STEP 1: Creating venv ==="
    python3 -m venv /data/venv
    source /data/venv/bin/activate
    pip install --upgrade pip
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    pip install torch_geometric==2.5.3 scipy numpy==1.26.4 rtree joblib wandb
    log "DONE venv: $(python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")')"
else
    log "SKIP venv (already exists)"
    source /data/venv/bin/activate
fi

# ═══ STEP 2: Extract global dataset ═══
GLOBAL_DIR="/data/datasets/wb2_512x256_19f_ar"
if [ ! -d "$GLOBAL_DIR" ]; then
    log "=== STEP 2: Extracting global 512x256 dataset ==="
    cd /data/datasets
    if [ -f dataset_512x256.tar.zst ]; then
        zstd -d dataset_512x256.tar.zst -o dataset_512x256.tar --long=31
        tar xf dataset_512x256.tar
        rm -f dataset_512x256.tar  # save space
        # Find extracted directory and rename
        EXTRACTED=$(ls -d */ 2>/dev/null | grep -i "512\|global\|wb2" | head -1)
        if [ -n "$EXTRACTED" ] && [ "$EXTRACTED" != "wb2_512x256_19f_ar/" ]; then
            mv "$EXTRACTED" wb2_512x256_19f_ar
        fi
        log "DONE global: $(ls $GLOBAL_DIR/ | head -5)"
    else
        log "ERROR: no dataset_512x256.tar.zst found"
        exit 1
    fi
else
    log "SKIP global (already exists)"
fi

# Verify global dataset
if [ ! -f "$GLOBAL_DIR/data.npy" ]; then
    # Maybe it was extracted into a subdirectory
    INNER=$(find /data/datasets -maxdepth 3 -name "data.npy" -path "*512*" -o -name "data.npy" -path "*global*" 2>/dev/null | head -1)
    if [ -n "$INNER" ]; then
        INNER_DIR=$(dirname "$INNER")
        log "Found global data at $INNER_DIR, linking to $GLOBAL_DIR"
        ln -sfn "$INNER_DIR" "$GLOBAL_DIR"
    else
        log "ERROR: data.npy not found in $GLOBAL_DIR — check extraction"
        ls -R /data/datasets/ | head -30
        exit 1
    fi
fi
log "Global: $(cat $GLOBAL_DIR/dataset_info.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"T={d[\"n_time\"]}, {d[\"n_lon\"]}x{d[\"n_lat\"]}, feat={d[\"n_feat\"]}")')"

# ═══ STEP 3: Extract regional dataset ═══
REGION_DIR="/data/datasets/region_krsk_61x41_19f_2010-2020_025deg"
if [ ! -d "$REGION_DIR" ]; then
    log "=== STEP 3: Extracting regional dataset ==="
    cd /data/datasets
    tar xzf region_krsk_61x41_19f_2010-2020_025deg.tar.gz
    log "DONE regional: $(cat $REGION_DIR/dataset_info.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"T={d[\"n_time\"]}, {d[\"n_lon\"]}x{d[\"n_lat\"]}, feat={d[\"n_feat\"]}")')"
else
    log "SKIP regional (already exists)"
fi

# ═══ STEP 4: Build multires dataset ═══
MULTIRES_DIR="/data/datasets/multires_krsk_19f"
cd /workdir/graphcast-lite
export PYTHONPATH=/workdir/graphcast-lite

if [ ! -f "$MULTIRES_DIR/data.npy" ]; then
    log "=== STEP 4: Building multires dataset (interpolate mode) ==="
    /data/venv/bin/python -u scripts/build_multires_dataset.py \
        --global-dir "$GLOBAL_DIR" \
        --region-coords "$REGION_DIR/coords.npz" \
        --roi 50 60 83 98 \
        --mode interpolate \
        --out-dir "$MULTIRES_DIR" 2>&1 | tee -a "$LOG"
    log "DONE multires: $(cat $MULTIRES_DIR/dataset_info.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"N={d[\"n_nodes\"]}, T={d[\"n_time\"]}, feat={d[\"n_feat\"]}")')"
else
    log "SKIP multires (already exists)"
fi

# ═══ STEP 5: Install tmux ═══
if ! command -v tmux &>/dev/null; then
    log "Installing tmux..."
    sudo apt-get install -y tmux 2>&1 | tail -2
fi

# ═══ STEP 6: Git pull latest code ═══
log "=== STEP 6: Pulling latest code ==="
cd /workdir/graphcast-lite
git pull 2>&1 | tail -3
log "Code updated"

# ═══ STEP 7: Baseline test on multires ═══
log "=== STEP 7: Baseline on multires ==="
PY="/data/venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
BASE_FLAGS="--data-dir $MULTIRES_DIR --max-samples 200 --ar-steps 4 --per-channel --no-residual"
DA_DIR="/data/da_multires_results"
mkdir -p "$DA_DIR"
MASTER="$DA_DIR/batch_master.log"

log2() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MASTER"; }

log2 "═══════════════════════════════════════════════════"
log2 "DA EXPERIMENTS — MULTIRES 133K dataset"
log2 "═══════════════════════════════════════════════════"

# Baseline без DA, без region (full multires)
log2 "=== Baseline (no DA) ==="
$PY scripts/predict.py $EXP $BASE_FLAGS \
    > "$DA_DIR/baseline_full.log" 2>&1
BL_SKILL=$(grep "Skill:" "$DA_DIR/baseline_full.log" | head -1)
log2 "DONE Baseline full: $BL_SKILL"

# Baseline с region (ROI метрики)
log2 "=== Baseline with ROI metrics ==="
$PY scripts/predict.py $EXP $BASE_FLAGS \
    --region 50 60 83 98 \
    > "$DA_DIR/baseline_roi.log" 2>&1
ROI_SKILL=$(grep "+06h:" "$DA_DIR/baseline_roi.log" | tail -1)
log2 "DONE Baseline ROI: $ROI_SKILL"

# Baseline с city metrics
$PY scripts/predict.py $EXP $BASE_FLAGS \
    --region 55.5 56.5 92 94 \
    > "$DA_DIR/baseline_city.log" 2>&1
CITY_SKILL=$(grep "+06h:" "$DA_DIR/baseline_city.log" | tail -1)
log2 "DONE Baseline City: $CITY_SKILL"

log "=== BASELINE RESULTS ==="
log "Full: $BL_SKILL"
log "ROI:  $ROI_SKILL"
log "City: $CITY_SKILL"

# ═══ STEP 8: DA Experiments on multires ═══
log2 "═══════════════════════════════════════════════════"
log2 "DA EXPERIMENTS — OI on multires"
log2 "═══════════════════════════════════════════════════"

DA_BASE="$BASE_FLAGS --region 50 60 83 98 --obs-roi-only --obs-seed 42"

run_oi() {
    local sparsity=$1 corr=$2 sigma=$3 tag=$4
    local log_file="$DA_DIR/oi_s${tag}_c${corr}_sig${sigma}.log"
    if [ -f "$log_file" ] && grep -q "Skill:" "$log_file" 2>/dev/null; then
        log2 "SKIP  OI s=$sparsity c=$corr σ=$sigma"
        return 0
    fi
    log2 "START OI sparsity=$sparsity corr=$corr sigma=$sigma"
    $PY scripts/predict.py $EXP $DA_BASE \
        --obs-sparsity $sparsity --assim-method oi \
        --oi-corr-len $corr --oi-sigma-o $sigma > "$log_file" 2>&1 || {
        log2 "ERROR OI s=$sparsity c=$corr σ=$sigma"
        return 1
    }
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log2 "DONE  OI s=$sparsity c=$corr σ=$sigma | $skill | $region"
}

run_nudge() {
    local sparsity=$1 alpha=$2 mode=$3 tag=$4
    local log_file="$DA_DIR/nudge_s${tag}_a${alpha}_${mode}.log"
    if [ -f "$log_file" ] && grep -q "Skill:" "$log_file" 2>/dev/null; then
        log2 "SKIP  Nudge s=$sparsity α=$alpha $mode"
        return 0
    fi
    log2 "START Nudge sparsity=$sparsity alpha=$alpha mode=$mode"
    $PY scripts/predict.py $EXP $DA_BASE \
        --obs-sparsity $sparsity --assim-method nudging \
        --nudging-alpha $alpha --nudging-mode $mode > "$log_file" 2>&1 || {
        log2 "ERROR Nudge s=$sparsity α=$alpha $mode"
        return 1
    }
    local skill=$(grep "Skill:" "$log_file" | head -1)
    local region=$(grep "+06h:" "$log_file" | tail -1)
    log2 "DONE  Nudge s=$sparsity α=$alpha $mode | $skill | $region"
}

# Part 1: OI 10% corr_len sweep (σ=0.5)
log2 "=== Part 1: OI 10% corr_len sweep (sigma=0.5) ==="
for C in 10000 25000 50000 100000 150000 200000 300000 500000; do
    run_oi 0.1 $C 0.5 01
done

# Part 2: OI 10% sigma sweep (best corr_lens from global: 100km, 150km)
log2 "=== Part 2: OI 10% sigma sweep ==="
for C in 10000 50000 100000 150000; do
    for S in 0.3 1.0; do
        run_oi 0.1 $C $S 01
    done
done

# Part 3: OI 1%
log2 "=== Part 3: OI 1% sweep ==="
for C in 10000 50000 100000 150000 200000 300000 500000; do
    run_oi 0.01 $C 0.5 001
done

# Part 4: Nudging 10%
log2 "=== Part 4: Nudging 10% ==="
run_nudge 0.1 0.3 sequential 01
run_nudge 0.1 0.5 sequential 01
run_nudge 0.1 0.7 sequential 01
run_nudge 0.1 0.3 offline 01

# Part 5: Nudging 1%
log2 "=== Part 5: Nudging 1% ==="
run_nudge 0.01 0.3 sequential 001
run_nudge 0.01 0.5 sequential 001

log2 "═══════════════════════════════════════════════════"
log2 "ALL DONE at $(date)"
log2 "═══════════════════════════════════════════════════"

# Summary
log2 ""
log2 "═══ SUMMARY ==="
for f in "$DA_DIR"/*.log; do
    [ "$f" = "$MASTER" ] && continue
    name=$(basename "$f" .log)
    skill=$(grep -oP "Skill:\s*\K[\d.]+%" "$f" 2>/dev/null | head -1 || echo "?")
    r6h=$(grep "+06h:" "$f" 2>/dev/null | tail -1 | grep -oP "skill=\K[\d.]+%" | head -1 || echo "?")
    r24h=$(grep "+24h:" "$f" 2>/dev/null | tail -1 | grep -oP "skill=\K[\d.]+%" | head -1 || echo "?")
    log2 "  $name | $skill | +6h: $r6h | +24h: $r24h"
done

log "═══ ALL COMPLETED ═══"
