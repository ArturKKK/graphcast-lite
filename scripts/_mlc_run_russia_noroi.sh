#!/usr/bin/env bash
# Russia multires 19f no-roi pipeline. VM = graphcast_v3-z1w6to.
# Предполагает: multires_russia_19f уже собран (после прошлого _mlc_run_russia.sh).
# Если нет — запустите старый _mlc_run_russia.sh, он создаст /data/datasets/multires_russia_19f.
#
# Pretrained: experiments/wb2_512x256_19f_ar_v2/best_model.pth (global v2)
#   — забран в git через `git checkout origin/main-arthur -- ...` либо force-add.
#
# Запуск из консоли laptop:
#   mlc job exec graphcast_v3-z1w6to -- bash -lc '(setsid nohup bash scripts/_mlc_run_russia_noroi.sh </dev/null >/dev/null 2>&1 &); sleep 3; tail -20 /data/logs/russia_noroi.log'

set -uo pipefail

LOG=/data/logs/russia_noroi.log
mkdir -p /data/logs
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] Russia multires 19f NO-ROI training"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
MULTI_DIR=$DATA/multires_russia_19f
EXP=experiments/multires_russia_19f_noroi
PRETRAINED=experiments/wb2_512x256_19f_ar_v2/best_model.pth

cd "$REPO"
source "$VENV/bin/activate"

# --- sanity ---
if [[ ! -f "$MULTI_DIR/data.npy" ]]; then
  echo "[ERR] $MULTI_DIR/data.npy missing — run scripts/_mlc_run_russia.sh first to build dataset"
  exit 2
fi
if [[ ! -f "$PRETRAINED" ]]; then
  echo "[ERR] $PRETRAINED missing. Pull from git:"
  echo "  cd $REPO && git fetch && git checkout origin/main-arthur -- $PRETRAINED"
  exit 3
fi
if [[ ! -f "$EXP/config.json" ]]; then
  echo "[ERR] $EXP/config.json missing — pull latest repo"
  exit 4
fi

# --- если уже есть checkpoint в exp_dir → --resume, иначе свежий finetune ---
RESUME_FLAG=""
if [[ -f "$EXP/results/checkpoint.pth" ]]; then
  echo "[info] found existing checkpoint — resuming"
  RESUME_FLAG="--resume"
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
nvidia-smi | head -20

python -m src.main "$EXP" \
  --pretrained "$PRETRAINED" \
  $RESUME_FLAG

echo "[done $(date)]"
