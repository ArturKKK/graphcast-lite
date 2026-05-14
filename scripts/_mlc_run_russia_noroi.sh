#!/usr/bin/env bash
# Russia multires 19f no-roi pipeline. VM = graphcast_v3-z1w6to.
# Дообучение от глобальной v2 (19 каналов), без roi_only_loss, AR=3 с первой эпохи, 12 эпох.
#
# Запуск:
#   mlc job exec graphcast_v3-z1w6to -- bash -lc \
#     '(setsid nohup bash scripts/_mlc_run_russia_noroi.sh </dev/null >/dev/null 2>&1 &); sleep 3; tail -20 /data/logs/russia_noroi.log'

set -uo pipefail

LOG=/data/logs/russia_noroi.log
mkdir -p /data/logs
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] Russia multires 19f NO-ROI training (AR=3, 12ep)"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_DIR=$DATA/wb2_512x256_19f_ar
RUS_DIR=$DATA/region_russia_645x165_19f_2010-2021_025deg
MULTI_DIR=$DATA/multires_russia_19f
EXP=experiments/multires_russia_19f_noroi
PRETRAINED=experiments/wb2_512x256_19f_ar_v2/best_model.pth

cd "$REPO"

# ----- 1. venv -----
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
fi
source "$VENV/bin/activate"
echo "[env] python=$(which python) torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# ----- 2. ensure base 19f data.npy -----
if [[ ! -f "$BASE_DIR/data.npy" ]]; then
  if [[ -f "$DATA/dataset_512x256.tar.zst" ]]; then
    echo "[2 $(date +%H:%M:%S)] extracting base 512x256 tar.zst"
    mkdir -p "$BASE_DIR"
    apt-get install -y -q zstd >/dev/null 2>&1 || true
    tar --use-compress-program=unzstd -xf "$DATA/dataset_512x256.tar.zst" -C "$BASE_DIR" --strip-components=1
    find "$BASE_DIR" -name "._*" -delete 2>/dev/null || true
    found=$(find "$BASE_DIR" -maxdepth 4 -name data.npy -type f | head -1)
    if [[ -n "$found" && "$(dirname "$found")" != "$BASE_DIR" ]]; then
      mv "$(dirname "$found")"/* "$BASE_DIR"/ 2>/dev/null || true
      find "$BASE_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    fi
    [[ -f "$BASE_DIR/data.npy" ]] && rm -f "$DATA/dataset_512x256.tar.zst"
  fi
fi
[[ -f "$BASE_DIR/data.npy" ]] || { echo "[ERR] $BASE_DIR/data.npy missing"; exit 2; }

# ----- 3. ensure Russia regional data.npy -----
if [[ ! -f "$RUS_DIR/data.npy" ]]; then
  RUS_TAR="$RUS_DIR/region_russia_645x165_19f_2010-2021_025deg.tar"
  if [[ -f "$RUS_TAR" ]]; then
    echo "[3 $(date +%H:%M:%S)] extracting Russia regional tar"
    tar -xf "$RUS_TAR" -C "$RUS_DIR" --strip-components=1 2>/dev/null \
      || tar -xf "$RUS_TAR" -C "$RUS_DIR"
    find "$RUS_DIR" -name "._*" -delete 2>/dev/null || true
    found=$(find "$RUS_DIR" -maxdepth 4 -name data.npy -type f | head -1)
    if [[ -n "$found" && "$(dirname "$found")" != "$RUS_DIR" ]]; then
      mv "$(dirname "$found")"/* "$RUS_DIR"/ 2>/dev/null || true
      find "$RUS_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    fi
    [[ -f "$RUS_DIR/data.npy" ]] && rm -f "$RUS_TAR"
  fi
fi
[[ -f "$RUS_DIR/data.npy" ]] || { echo "[ERR] $RUS_DIR/data.npy missing"; exit 3; }

# ----- 4. build multires merge dataset -----
if [[ ! -f "$MULTI_DIR/data.npy" ]]; then
  echo "[4 $(date +%H:%M:%S)] building multires merge dataset"
  mkdir -p "$MULTI_DIR"
  python scripts/build_multires_dataset.py \
    --global-dir "$BASE_DIR" \
    --region-dir "$RUS_DIR" \
    --roi 41 82 19 180 \
    --mode merge \
    --out-dir "$MULTI_DIR"
else
  echo "[4] multires already built"
fi
ls -lh "$MULTI_DIR" | head

# ----- 5. pretrained sanity -----
[[ -f "$PRETRAINED" ]] || { echo "[ERR] $PRETRAINED missing — git pull first"; exit 4; }

# ----- 6. launch training -----
RESUME_FLAG=""
[[ -f "$EXP/results/checkpoint.pth" ]] && { echo "[info] resuming"; RESUME_FLAG="--resume"; }

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
nvidia-smi | head -20

python -m src.main "$EXP" \
  --pretrained "$PRETRAINED" \
  $RESUME_FLAG

echo "[done $(date)]"
