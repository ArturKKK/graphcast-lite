#!/bin/bash
# Master experiment script — timeforce tests + DA grid
# Run with setsid on VM: setsid /data/run_all.sh </dev/null >/dev/null 2>&1 &
set -e
cd /workdir/graphcast-lite
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
P=.venv/bin/python

# ===================== TIMEFORCE TESTS =====================
echo "=== TF1: --no-residual --prune-mesh ===" && date
$P -u scripts/predict.py experiments/region_krsk_23f_timeforce \
    --max-samples 200 --ar-steps 4 --per-channel \
    --no-residual --prune-mesh > /data/tf_nores_prune.log 2>&1 || true

echo "=== TF2: residual ON --prune-mesh ===" && date
$P -u scripts/predict.py experiments/region_krsk_23f_timeforce \
    --max-samples 200 --ar-steps 4 --per-channel \
    --prune-mesh > /data/tf_res_prune.log 2>&1 || true

echo "=== TF3: --no-residual no-prune ===" && date
$P -u scripts/predict.py experiments/region_krsk_23f_timeforce \
    --max-samples 200 --ar-steps 4 --per-channel \
    --no-residual > /data/tf_nores_noprune.log 2>&1 || true

echo "=== TF4: residual ON no-prune ===" && date
$P -u scripts/predict.py experiments/region_krsk_23f_timeforce \
    --max-samples 200 --ar-steps 4 --per-channel \
    > /data/tf_res_noprune.log 2>&1 || true

echo TIMEFORCE_DONE > /data/tf_done.flag
date

# ===================== DA GRID: OI =====================
# Base flags for freeze6 + OI
BFLAGS="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42 --assim-method oi"

# --- Sparsity 10% (s=0.1), vary corr_len ---
echo "=== OI s=0.1 corr=50km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 50000 --oi-sigma-o 0.5 > /data/da_oi_s01_c50k_sig05.log 2>&1

echo "=== OI s=0.1 corr=100km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/da_oi_s01_c100k_sig05.log 2>&1

echo "=== OI s=0.1 corr=150km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 150000 --oi-sigma-o 0.5 > /data/da_oi_s01_c150k_sig05.log 2>&1

# --- Sparsity 10%, vary sigma ---
echo "=== OI s=0.1 corr=10km sig=1.0 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 10000 --oi-sigma-o 1.0 > /data/da_oi_s01_c10k_sig10.log 2>&1

echo "=== OI s=0.1 corr=10km sig=2.0 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 10000 --oi-sigma-o 2.0 > /data/da_oi_s01_c10k_sig20.log 2>&1

echo "=== OI s=0.1 corr=50km sig=1.0 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.1 --oi-corr-len 50000 --oi-sigma-o 1.0 > /data/da_oi_s01_c50k_sig10.log 2>&1

# --- Sparsity 1% (s=0.01) ---
echo "=== OI s=0.01 corr=10km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.01 --oi-corr-len 10000 --oi-sigma-o 0.5 > /data/da_oi_s001_c10k_sig05.log 2>&1

echo "=== OI s=0.01 corr=50km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.01 --oi-corr-len 50000 --oi-sigma-o 0.5 > /data/da_oi_s001_c50k_sig05.log 2>&1

echo "=== OI s=0.01 corr=10km sig=1.0 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.01 --oi-corr-len 10000 --oi-sigma-o 1.0 > /data/da_oi_s001_c10k_sig10.log 2>&1

echo "=== OI s=0.01 corr=100km sig=0.5 ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $BFLAGS \
    --obs-sparsity 0.01 --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/da_oi_s001_c100k_sig05.log 2>&1

# ===================== DA GRID: NUDGING s=0.01 =====================
NBASE="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42 --assim-method nudging --nudging-mode sequential"

echo "=== Nudge s=0.01 a=0.5 seq ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $NBASE \
    --obs-sparsity 0.01 --nudging-alpha 0.5 > /data/da_nudge_s001_a05_seq.log 2>&1

echo "=== Nudge s=0.01 a=0.3 seq ===" && date
$P -u scripts/predict.py experiments/multires_nores_freeze6 $NBASE \
    --obs-sparsity 0.01 --nudging-alpha 0.3 > /data/da_nudge_s001_a03_seq.log 2>&1

echo "=== ALL DONE ===" && date
echo ALLDONE > /data/all_done.flag
