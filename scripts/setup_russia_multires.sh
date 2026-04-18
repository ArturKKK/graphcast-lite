#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# VM SETUP: Build Russia-wide interpolated multires dataset + train
# Run on a NEW MLC VM with the repo and global dataset available.
#
# Pre-requisites on the VM:
#   - /workdir/graphcast-lite  (git clone)
#   - /data/datasets/wb2_512x256_19f_ar  (global dataset)
#   - .venv with torch installed (or /data/venv)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
LOG="/data/setup_russia.log"
mkdir -p /data
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

cd /workdir/graphcast-lite
export PYTHONPATH=/workdir/graphcast-lite

# Detect python
if [ -f .venv/bin/python ]; then
    PY=".venv/bin/python"
elif [ -f /data/venv/bin/python ]; then
    PY="/data/venv/bin/python"
else
    log "ERROR: No python venv found. Create one first."
    exit 1
fi
log "Using python: $PY"

# LD path for CUDA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/mlcore/conda/lib"

GLOBAL_DIR="/data/datasets/wb2_512x256_19f_ar"
RUSSIA_COORDS="/data/datasets/russia_025deg_coords.npz"
MULTIRES_DIR="/data/datasets/multires_russia_19f"

# Russia bounding box (0.25° grid):
# lat: 41.0 - 82.0  (covers from Black Sea to Arctic)
# lon: 19.0 - 180.0  (from Kaliningrad to Kamchatka)
ROI_LAT_MIN=41.0
ROI_LAT_MAX=82.0
ROI_LON_MIN=19.0
ROI_LON_MAX=180.0

# ═══ STEP 1: Generate Russia region coords.npz ═══
if [ ! -f "$RUSSIA_COORDS" ]; then
    log "=== STEP 1: Generating Russia 0.25° coords ==="
    $PY -c "
import numpy as np
lat = np.arange($ROI_LAT_MIN, $ROI_LAT_MAX + 0.01, 0.25)
lon = np.arange($ROI_LON_MIN, $ROI_LON_MAX + 0.01, 0.25)
print(f'Russia grid: lat {lat.min():.1f}-{lat.max():.1f} ({len(lat)} pts), lon {lon.min():.1f}-{lon.max():.1f} ({len(lon)} pts)')
print(f'Regional nodes: {len(lat)} x {len(lon)} = {len(lat)*len(lon)}')
np.savez('$RUSSIA_COORDS', latitude=lat.astype(np.float64), longitude=lon.astype(np.float64))
print('Saved:', '$RUSSIA_COORDS')
"
    log "DONE Russia coords"
else
    log "SKIP Russia coords (exists)"
fi

# ═══ STEP 2: Build multires dataset (interpolate mode) ═══
if [ ! -f "$MULTIRES_DIR/data.npy" ]; then
    log "=== STEP 2: Building Russia multires dataset (interpolate) ==="
    $PY -u scripts/build_multires_dataset.py \
        --global-dir "$GLOBAL_DIR" \
        --region-coords "$RUSSIA_COORDS" \
        --roi $ROI_LAT_MIN $ROI_LAT_MAX $ROI_LON_MIN $ROI_LON_MAX \
        --mode interpolate \
        --out-dir "$MULTIRES_DIR" 2>&1 | tee -a "$LOG"
    log "DONE multires Russia"
else
    log "SKIP multires Russia (exists)"
fi

log "Dataset ready at $MULTIRES_DIR"
log "To start training:"
log "  $PY -u src/main.py experiments/multires_russia_19f --pretrained experiments/wb2_512x256_19f_ar_v2/best_model.pth"
