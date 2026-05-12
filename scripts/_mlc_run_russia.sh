#!/usr/bin/env bash
# Russia multires fine-tune end-to-end pipeline (lean, archive-cleanup).
# Runs on MLC VM. Targets: /data for big stuff, /workdir/graphcast-lite for code.
set -uo pipefail

LOG=/data/logs/russia_pipeline.log
mkdir -p /data/logs /data/venvs /data/datasets
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] Russia multires pipeline"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_DIR=$DATA/wb2_512x256_19f_ar
RUS_DIR=$DATA/region_russia_645x165_19f_2010-2021_025deg
MULTI_DIR=$DATA/multires_russia_19f

cd "$REPO"

# ----- 1. venv -----
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[1/6 $(date +%H:%M:%S)] creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
  echo "[1/6] venv ready"
else
  echo "[1/6] venv already present"
fi
source "$VENV/bin/activate"
echo "python = $(which python); torch = $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# ----- 2. extract base 512x256 zst (delete archive right after) -----
if [[ ! -f "$BASE_DIR/data.npy" ]]; then
  echo "[2/6 $(date +%H:%M:%S)] extracting base 512x256 zst"
  mkdir -p "$BASE_DIR"
  zstd_bin=$(command -v zstd || echo "")
  if [[ -z "$zstd_bin" ]]; then
    apt-get install -y -q zstd >/dev/null 2>&1 || pip install -q zstandard
    zstd_bin=$(command -v zstd || echo "")
  fi
  # tar.zst is nested as wb2_512x256_19f_ar/wb2_512x256_19f_ar/...
  tar --use-compress-program=unzstd -xf "$DATA/dataset_512x256.tar.zst" -C "$BASE_DIR" --strip-components=1
  # Drop macOS AppleDouble junk
  find "$BASE_DIR" -name "._*" -delete 2>/dev/null || true
  # Locate data.npy and flatten everything up to $BASE_DIR
  found=$(find "$BASE_DIR" -maxdepth 4 -name data.npy -type f | head -1)
  if [[ -n "$found" && "$(dirname "$found")" != "$BASE_DIR" ]]; then
    src=$(dirname "$found")
    echo "[2/6] flattening from $src -> $BASE_DIR"
    mv "$src"/* "$BASE_DIR"/ 2>/dev/null || true
    # remove now-empty intermediate dirs
    find "$BASE_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  fi
  ls -lh "$BASE_DIR" | head
  if [[ -f "$BASE_DIR/data.npy" ]]; then
    rm -f "$DATA/dataset_512x256.tar.zst"
    echo "[2/6] base extracted; archive removed"
  else
    echo "[2/6] ERROR: data.npy not in $BASE_DIR after extract"
    ls -laR "$BASE_DIR" | head -50
    exit 2
  fi
else
  echo "[2/6] base already extracted"
fi

# ----- 3. extract Russia tar (delete archive right after) -----
if [[ ! -f "$RUS_DIR/data.npy" ]]; then
  echo "[3/6 $(date +%H:%M:%S)] extracting Russia tar"
  tar -xf "$RUS_DIR/region_russia_645x165_19f_2010-2021_025deg.tar" -C "$RUS_DIR" --strip-components=1 2>/dev/null \
    || tar -xf "$RUS_DIR/region_russia_645x165_19f_2010-2021_025deg.tar" -C "$RUS_DIR"
  find "$RUS_DIR" -name "._*" -delete 2>/dev/null || true
  found=$(find "$RUS_DIR" -maxdepth 4 -name data.npy -type f | head -1)
  if [[ -n "$found" && "$(dirname "$found")" != "$RUS_DIR" ]]; then
    src=$(dirname "$found")
    echo "[3/6] flattening from $src -> $RUS_DIR"
    mv "$src"/* "$RUS_DIR"/ 2>/dev/null || true
    find "$RUS_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  fi
  ls -lh "$RUS_DIR" | head
  if [[ -f "$RUS_DIR/data.npy" ]]; then
    rm -f "$RUS_DIR/region_russia_645x165_19f_2010-2021_025deg.tar"
    echo "[3/6] Russia extracted; archive removed"
  else
    echo "[3/6] ERROR: data.npy not in $RUS_DIR after extract"
    ls -laR "$RUS_DIR" | head -50
    exit 3
  fi
else
  echo "[3/6] Russia already extracted"
fi

df -h /data | tail -1

# ----- 4. build multires merge dataset -----
if [[ ! -f "$MULTI_DIR/data.npy" ]]; then
  echo "[4/6 $(date +%H:%M:%S)] building multires merge dataset"
  mkdir -p "$MULTI_DIR"
  python scripts/build_multires_dataset.py \
    --global-dir "$BASE_DIR" \
    --region-dir "$RUS_DIR" \
    --roi 41 82 19 180 \
    --mode merge \
    --out-dir "$MULTI_DIR"
  echo "[4/6] merge build finished"
  ls -lh "$MULTI_DIR" | head
else
  echo "[4/6] multires already built"
fi

# ----- 5. cleanup source data.npy to reclaim disk before training -----
echo "[5/6 $(date +%H:%M:%S)] removing source data.npy files to reclaim disk"
rm -f "$BASE_DIR/data.npy" "$RUS_DIR/data.npy" || true
df -h /data | tail -1

# ----- 6. launch training -----
echo "[6/6 $(date +%H:%M:%S)] launching training"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd "$REPO"
nvidia-smi | head -20

python -m src.main experiments/multires_russia_19f_freeze6 \
  --pretrained experiments/multires_russia_19f_freeze6/best_model.pth \
  --resume

echo "[done $(date)]"
