#!/bin/bash
set -e
LOG=/data/pipeline.log
log(){ echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

# 1. Extract global wb2
if [ ! -f /data/datasets/wb2_512x256_19f_ar/data.npy ]; then
  log "extract wb2_512x256..."
  cd / && zstd -d -c /data/datasets/dataset_512x256.tar.zst | tar -xf - --exclude="._*" 2>>"$LOG"
  log "done wb2_512x256"
else
  log "skip wb2_512x256 (exists)"
fi

# 2. Build russia multires (interpolate)
if [ ! -f /data/datasets/multires_russia_19f/dataset_info.json ]; then
  log "build russia multires..."
  cd /workdir/graphcast-lite && bash scripts/setup_russia_multires.sh 2>&1 | tee -a "$LOG"
  log "done russia multires"
else
  log "skip russia multires (exists)"
fi

# 3. Resume training
log "starting training (--resume)..."
cd /workdir/graphcast-lite
export PYTHONPATH=/workdir/graphcast-lite
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/mlcore/conda/lib"
.venv/bin/python -u src/main.py experiments/multires_russia_19f --resume >> /data/v1_train.log 2>&1
log "training exited rc=$?"
