#!/bin/bash
# Wait for multires Russia build to complete, then resume training.
# Designed to run on MLC VM in background (setsid nohup ... &).
LOG=/data/auto_resume.log
echo "[$(date)] auto_resume started, waiting for build..." >> "$LOG"

while [ ! -f /data/datasets/multires_russia_19f/dataset_info.json ]; do
    sleep 60
    if ! pgrep -f "build_multires_dataset" > /dev/null; then
        if [ ! -f /data/datasets/multires_russia_19f/dataset_info.json ]; then
            echo "[$(date)] BUILD DIED without dataset_info.json — abort" >> "$LOG"
            exit 1
        fi
        break
    fi
done

echo "[$(date)] build finished, sleeping 30s for fs sync..." >> "$LOG"
sleep 30
ls -la /data/datasets/multires_russia_19f/ >> "$LOG"

cd /workdir/graphcast-lite
export PYTHONPATH=/workdir/graphcast-lite
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/mlcore/conda/lib"

echo "[$(date)] starting training with --resume" >> "$LOG"
.venv/bin/python -u src/main.py experiments/multires_russia_19f --resume >> /data/v1_train.log 2>&1
echo "[$(date)] training exited rc=$?" >> "$LOG"
