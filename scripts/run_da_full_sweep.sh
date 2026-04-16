#!/usr/bin/env bash
# Full DA experiment sweep for BOTH interpolate and merge multires datasets
# Model: multires_nores_freeze6
# Covers: OI (full param grid), nudging (seq+offline), variable groups
#
# Timing strategy:
#   - Baseline + best OI configs: 200 samples with --per-channel (full detail)
#   - Parameter sweep: 50 samples (fast screening)
#   - Best configs from screening re-run with 200 samples later
set -euo pipefail

source /data/venv/bin/activate
cd /workdir/graphcast-lite

EXP=experiments/multires_nores_freeze6
SEED=42

run_exp() {
    local tag="$1"; shift   # dataset tag: interp / merge
    local data="$1"; shift  # data dir
    local out="$1"; shift   # output dir
    local name="$1"; shift  # experiment name
    local samples="$1"; shift  # max samples
    local logfile="$out/${name}.log"

    if [[ -f "$logfile" ]]; then
        if grep -q "Overall:" "$logfile" 2>/dev/null; then
            echo "$(date '+%H:%M:%S') SKIP $tag/$name (already done)"
            return 0
        fi
    fi

    echo "$(date '+%H:%M:%S') >>> $tag/$name (N=$samples)"
    local extra_flags=""
    if [[ "$samples" -ge 200 ]]; then
        extra_flags="--per-channel"
    fi
    python scripts/predict.py "$EXP" \
        --data-dir "$data" --no-residual \
        --region 50 60 83 98 --prune-mesh \
        --ar-steps 4 --obs-seed "$SEED" --obs-roi-only \
        --max-samples "$samples" $extra_flags \
        "$@" > "$logfile" 2>&1 || true

    # Extract region metrics to summary
    echo "=== $tag/$name (N=$samples) ===" >> "$out/summary.txt"
    grep -E "Overall:|^\s+\+|Region|City|per-channel" "$logfile" | tail -30 >> "$out/summary.txt" 2>/dev/null
    echo "" >> "$out/summary.txt"
    echo "$(date '+%H:%M:%S') <<< $tag/$name done"
}

# ═══════════════════════════════════════════════════════
# PART 1: INTERPOLATE DATASET
# ═══════════════════════════════════════════════════════
DATA_INTERP=/data/datasets/multires_krsk_19f
OUT_INTERP=/data/da_results_interp
mkdir -p "$OUT_INTERP"

echo "$(date '+%H:%M:%S') ════════════════════════════════════════" | tee -a "$OUT_INTERP/summary.txt"
echo "INTERPOLATE DATASET EXPERIMENTS" | tee -a "$OUT_INTERP/summary.txt"
echo "$(date '+%H:%M:%S') ════════════════════════════════════════" | tee -a "$OUT_INTERP/summary.txt"

# --- Baseline ---
run_exp interp "$DATA_INTERP" "$OUT_INTERP" "baseline" 200 --assim-method none

# --- OI 10%: full corr_len × sigma grid (screening: 50 samples) ---
# NOTE: --oi-corr-len is in METERS. We pass km values multiplied by 1000.
for corr_km in 10 25 50 100 150 200 300 500; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5 1.0; do
        run_exp interp "$DATA_INTERP" "$OUT_INTERP" "oi10_c${corr_km}km_s${sigma}" 50 \
            --assim-method oi --obs-sparsity 0.1 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# --- OI 1%: full corr_len × sigma grid (screening: 50 samples) ---
for corr_km in 10 25 50 100 150 200 300 500; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5 1.0; do
        run_exp interp "$DATA_INTERP" "$OUT_INTERP" "oi1_c${corr_km}km_s${sigma}" 50 \
            --assim-method oi --obs-sparsity 0.01 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# --- Nudging 10%: alpha sweep × sequential (screening) ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp interp "$DATA_INTERP" "$OUT_INTERP" "nudg10_a${alpha}_seq" 50 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode sequential
done

# --- Nudging 10%: alpha sweep × offline ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp interp "$DATA_INTERP" "$OUT_INTERP" "nudg10_a${alpha}_off" 50 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode offline
done

# --- Nudging 1%: alpha sweep × sequential ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp interp "$DATA_INTERP" "$OUT_INTERP" "nudg1_a${alpha}_seq" 50 \
        --assim-method nudging --obs-sparsity 0.01 \
        --nudging-alpha "$alpha" --nudging-mode sequential
done

# --- Nudging 1%: alpha sweep × offline ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp interp "$DATA_INTERP" "$OUT_INTERP" "nudg1_a${alpha}_off" 50 \
        --assim-method nudging --obs-sparsity 0.01 \
        --nudging-alpha "$alpha" --nudging-mode offline
done

# --- Variable group experiments (OI 10%, c=100km, σ=0.5, screening) ---
run_exp interp "$DATA_INTERP" "$OUT_INTERP" "vargroup_t2m_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m"

run_exp interp "$DATA_INTERP" "$OUT_INTERP" "vargroup_twind_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v"

run_exp interp "$DATA_INTERP" "$OUT_INTERP" "vargroup_surface_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v,msl,sp"

run_exp interp "$DATA_INTERP" "$OUT_INTERP" "vargroup_surfupper_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v,msl,sp,t@850,u@850,v@850,t@500,u@500,v@500"

# ═══════════════════════════════════════════════════════
# PART 2: MERGE DATASET
# ═══════════════════════════════════════════════════════
DATA_MERGE=/data/datasets/multires_krsk_19f_merge
OUT_MERGE=/data/da_results_merge
mkdir -p "$OUT_MERGE"

if [[ ! -f "$DATA_MERGE/data.npy" ]]; then
    echo "$(date '+%H:%M:%S') WAITING for merge dataset to build..."
    while [[ ! -f "$DATA_MERGE/data.npy" ]] || ! grep -q "DONE" /data/build_merge.log 2>/dev/null; do
        sleep 30
    done
    echo "$(date '+%H:%M:%S') Merge dataset ready!"
fi

echo "$(date '+%H:%M:%S') ════════════════════════════════════════" | tee -a "$OUT_MERGE/summary.txt"
echo "MERGE DATASET EXPERIMENTS" | tee -a "$OUT_MERGE/summary.txt"
echo "$(date '+%H:%M:%S') ════════════════════════════════════════" | tee -a "$OUT_MERGE/summary.txt"

# --- Baseline ---
run_exp merge "$DATA_MERGE" "$OUT_MERGE" "baseline" 200 --assim-method none

# --- OI 10%: full corr_len × sigma grid (screening: 50 samples) ---
# NOTE: --oi-corr-len is in METERS. We pass km values multiplied by 1000.
for corr_km in 10 25 50 100 150 200 300 500; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5 1.0; do
        run_exp merge "$DATA_MERGE" "$OUT_MERGE" "oi10_c${corr_km}km_s${sigma}" 50 \
            --assim-method oi --obs-sparsity 0.1 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# --- OI 1%: full corr_len × sigma grid (screening: 50 samples) ---
for corr_km in 10 25 50 100 150 200 300 500; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5 1.0; do
        run_exp merge "$DATA_MERGE" "$OUT_MERGE" "oi1_c${corr_km}km_s${sigma}" 50 \
            --assim-method oi --obs-sparsity 0.01 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# --- Nudging 10%: alpha sweep × sequential ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp merge "$DATA_MERGE" "$OUT_MERGE" "nudg10_a${alpha}_seq" 50 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode sequential
done

# --- Nudging 10%: alpha sweep × offline ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp merge "$DATA_MERGE" "$OUT_MERGE" "nudg10_a${alpha}_off" 50 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode offline
done

# --- Nudging 1%: alpha sweep × sequential ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp merge "$DATA_MERGE" "$OUT_MERGE" "nudg1_a${alpha}_seq" 50 \
        --assim-method nudging --obs-sparsity 0.01 \
        --nudging-alpha "$alpha" --nudging-mode sequential
done

# --- Nudging 1%: alpha sweep × offline ---
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    run_exp merge "$DATA_MERGE" "$OUT_MERGE" "nudg1_a${alpha}_off" 50 \
        --assim-method nudging --obs-sparsity 0.01 \
        --nudging-alpha "$alpha" --nudging-mode offline
done

# --- Variable group experiments (OI 10%, c=100km, σ=0.5) ---
run_exp merge "$DATA_MERGE" "$OUT_MERGE" "vargroup_t2m_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m"

run_exp merge "$DATA_MERGE" "$OUT_MERGE" "vargroup_twind_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v"

run_exp merge "$DATA_MERGE" "$OUT_MERGE" "vargroup_surface_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v,msl,sp"

run_exp merge "$DATA_MERGE" "$OUT_MERGE" "vargroup_surfupper_oi10" 50 \
    --assim-method oi --obs-sparsity 0.1 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 \
    --obs-channels "t2m,10u,10v,msl,sp,t@850,u@850,v@850,t@500,u@500,v@500"

# ═══════════════════════════════════════════════════════
# PART 3: RE-RUN BEST CONFIGS WITH 200 SAMPLES
# (Based on prior knowledge of best params from temp_itog1.md)
# ═══════════════════════════════════════════════════════
echo "$(date '+%H:%M:%S') ════════════════════════════════════════"
echo "BEST CONFIGS — 200 SAMPLES WITH PER-CHANNEL"
echo "$(date '+%H:%M:%S') ════════════════════════════════════════"

# Interpolate: Best OI 10% (likely c=100km, σ=0.3 or 0.5)
# NOTE: --oi-corr-len is in METERS, multiply km by 1000
for corr_km in 50 100 150; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5; do
        run_exp interp "$DATA_INTERP" "$OUT_INTERP" "BEST_oi10_c${corr_km}km_s${sigma}" 200 \
            --assim-method oi --obs-sparsity 0.1 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# Interpolate: Best OI 1% (likely c=150-200km, σ=0.3 or 0.5)
for corr_km in 100 150 200; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5; do
        run_exp interp "$DATA_INTERP" "$OUT_INTERP" "BEST_oi1_c${corr_km}km_s${sigma}" 200 \
            --assim-method oi --obs-sparsity 0.01 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# Merge: Best OI 10% 
for corr_km in 50 100 150; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5; do
        run_exp merge "$DATA_MERGE" "$OUT_MERGE" "BEST_oi10_c${corr_km}km_s${sigma}" 200 \
            --assim-method oi --obs-sparsity 0.1 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# Merge: Best OI 1%
for corr_km in 100 150 200 300; do
    corr_m=$((corr_km * 1000))
    for sigma in 0.3 0.5; do
        run_exp merge "$DATA_MERGE" "$OUT_MERGE" "BEST_oi1_c${corr_km}km_s${sigma}" 200 \
            --assim-method oi --obs-sparsity 0.01 \
            --oi-corr-len "$corr_m" --oi-sigma-o "$sigma"
    done
done

# Best nudging configs (200 samples)
for alpha in 0.5 0.7; do
    run_exp interp "$DATA_INTERP" "$OUT_INTERP" "BEST_nudg10_a${alpha}_seq" 200 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode sequential
    run_exp merge "$DATA_MERGE" "$OUT_MERGE" "BEST_nudg10_a${alpha}_seq" 200 \
        --assim-method nudging --obs-sparsity 0.1 \
        --nudging-alpha "$alpha" --nudging-mode sequential
done

# ═══════════════════════════════════════════════════════
# FINAL: Summary
# ═══════════════════════════════════════════════════════
echo ""
echo "$(date '+%H:%M:%S') ╔══════════════════════════════════════╗"
echo "$(date '+%H:%M:%S') ║     ALL EXPERIMENTS COMPLETE         ║"
echo "$(date '+%H:%M:%S') ╚══════════════════════════════════════╝"
echo ""
echo "Interpolate results: $OUT_INTERP/ ($(ls $OUT_INTERP/*.log 2>/dev/null | wc -l) experiments)"
echo "Merge results: $OUT_MERGE/ ($(ls $OUT_MERGE/*.log 2>/dev/null | wc -l) experiments)"
