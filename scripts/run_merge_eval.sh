#!/bin/bash
# =======================================================================
# Полная оценка nores_freeze6 и real_freeze6 на merge-датасете
# VM: graphcast_v2-vqpgo2  |  15 апреля 2026
# =======================================================================
set -euo pipefail

export LD_LIBRARY_PATH=/home/mlcore/conda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workdir/graphcast-lite

cd /workdir/graphcast-lite

OUTDIR=/data/merge_eval_results
mkdir -p "$OUTDIR"
PYTHON=/data/venv/bin/python

# ═══════════════════════════════════════════════════════════════
# STEP 0: Setup venv (if needed)
# ═══════════════════════════════════════════════════════════════
if [ ! -f "$PYTHON" ]; then
    echo "========== STEP 0: Creating venv =========="
    python3 -m venv /data/venv
    /data/venv/bin/pip install --upgrade pip
    /data/venv/bin/pip install numpy==1.26.4 scipy==1.13.1 torch==2.2.2 \
        pydantic==2.7.3 trimesh==4.4.0 torch_geometric==2.5.3 tabulate==0.8.9
    echo "[0] venv ready."
else
    echo "[0] venv already exists."
fi

# ═══════════════════════════════════════════════════════════════
# STEP 1: Распаковка датасетов
# ═══════════════════════════════════════════════════════════════
echo "========== STEP 1: Extracting datasets =========="

# 1a. Regional dataset (1.2 GB, fast)
if [ ! -f /data/datasets/region_krsk_61x41_19f_2010-2020_025deg/data.npy ]; then
    echo "[1a] Extracting regional dataset..."
    cd /data/datasets
    tar xzf region_krsk_61x41_19f_2010-2020_025deg.tar.gz
    echo "[1a] Done."
else
    echo "[1a] Regional dataset already extracted."
fi

# 1b. Global dataset (74 GB, slow)
if [ ! -f /data/datasets/wb2_512x256_19f_ar/data.npy ]; then
    echo "[1b] Extracting global dataset (74GB, will take a while)..."
    cd /data/datasets
    zstd -dc dataset_512x256.tar.zst | tar xf - --strip-components=2
    # archive has data/datasets/wb2_512x256_19f_ar/ → strip 2 gives wb2_512x256_19f_ar/
    echo "[1b] Done."
else
    echo "[1b] Global dataset already extracted."
fi

cd /workdir/graphcast-lite

# Verify
echo "=== Checking datasets ==="
ls -lh /data/datasets/wb2_512x256_19f_ar/data.npy
ls -lh /data/datasets/region_krsk_61x41_19f_2010-2020_025deg/data.npy

# ═══════════════════════════════════════════════════════════════
# STEP 2: Build merge dataset
# ═══════════════════════════════════════════════════════════════
echo "========== STEP 2: Building merge dataset =========="

MERGE_DS=/data/datasets/multires_krsk_19f_merge
if [ ! -f "$MERGE_DS/data.npy" ]; then
    echo "[2] Running build_multires_dataset.py --mode merge..."
    $PYTHON -u scripts/build_multires_dataset.py \
        --global-dir /data/datasets/wb2_512x256_19f_ar \
        --region-dir /data/datasets/region_krsk_61x41_19f_2010-2020_025deg \
        --roi 50 60 83 98 \
        --mode merge \
        --out-dir "$MERGE_DS"
    echo "[2] Done."
else
    echo "[2] Merge dataset already exists."
fi

# Fix symlinks
ln -sfn "$MERGE_DS" data/datasets/multires_krsk_19f_real
ln -sfn "$MERGE_DS" data/datasets/multires_krsk_19f_merge

echo "=== Merge dataset info ==="
cat "$MERGE_DS/dataset_info.json"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 3: Inference — nores_freeze6 on merge
# ═══════════════════════════════════════════════════════════════
echo "========== STEP 3: nores_freeze6 on merge =========="

echo "[3a] nores_freeze6 — full (2501 region + global) on merge"
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 \
    --data-dir "$MERGE_DS" \
    --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    2>&1 | tee "$OUTDIR/nores_freeze6_merge_2501.log"

echo ""
echo "[3b] nores_freeze6 — city 45 nodes on merge"
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 \
    --data-dir "$MERGE_DS" \
    --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    --region 55.5 56.5 92 94 \
    2>&1 | tee "$OUTDIR/nores_freeze6_merge_city45.log"

# ═══════════════════════════════════════════════════════════════
# STEP 4: Inference — real_freeze6 on merge
# ═══════════════════════════════════════════════════════════════
echo "========== STEP 4: real_freeze6 on merge =========="

echo "[4a] real_freeze6 — full (2501 region + global) on merge"
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 \
    --data-dir "$MERGE_DS" \
    --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    2>&1 | tee "$OUTDIR/real_freeze6_merge_2501.log"

echo ""
echo "[4b] real_freeze6 — city 45 nodes on merge"
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 \
    --data-dir "$MERGE_DS" \
    --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    --region 55.5 56.5 92 94 \
    2>&1 | tee "$OUTDIR/real_freeze6_merge_city45.log"

echo ""
echo "================================================================"
echo "ALL DONE! Results in $OUTDIR/"
echo "================================================================"
ls -lh "$OUTDIR/"
