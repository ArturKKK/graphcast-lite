#!/usr/bin/env bash
# v3 GLOBAL pipeline for wb2_512x256_33f_v3.
#
# На VM ожидается:
#   /data/datasets/dataset_512x256.tar.zst                              (~79 GB)  base 19f архив
#   /data/datasets/global_512x256_extra_2010-2021_07deg/data_extra.npy  (~43 GB)  10 plev уже распакованный
#   /data/datasets/global_512x256_extra_2010-2021_07deg/scalers_extra.npz
#   /data/datasets/global_512x256_extra_2010-2021_07deg/dataset_info_extra.json
#
# Шаги:
#   1. setup venv (/data/venvs/graphcast) + git clone репо
#   2. распаковать base 19f tar.zst → /data/datasets/wb2_512x256_19f_ar/ (data.npy)
#   3. python scripts/build_v3_extra_with_time.py → /data/datasets/wb2_512x256_33f_v3/
#        (symlink data.npy + новый data_extra.npy 14ch + scalers/coords/variables/info)
#   4. cleanup tar.zst
#   5. launch python -m src.main experiments/wb2_512x256_33f_ar_v3
#
set -uo pipefail

LOG=/data/logs/v3_global_pipeline.log
mkdir -p /data/logs /data/venvs /data/datasets
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] v3 GLOBAL pipeline (33f = 19 + 10 + 4)"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_DIR=$DATA/wb2_512x256_19f_ar
EXTRA_DIR=$DATA/global_512x256_extra_2010-2021_07deg
OUT_DIR=$DATA/wb2_512x256_33f_v3

# ===== 1. Repo + venv =====
if [[ ! -d "$REPO/.git" ]]; then
  echo "[1a/5 $(date +%H:%M:%S)] cloning repo to $REPO"
  mkdir -p /workdir
  git clone -b main-arthur https://github.com/ArturKKK/graphcast-lite.git "$REPO"
else
  echo "[1a/5] repo present; pulling latest"
  (cd "$REPO" && git pull --rebase --autostash) || echo "  pull failed (continuing)"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[1b/5 $(date +%H:%M:%S)] creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
else
  echo "[1b/5] venv already present"
fi
source "$VENV/bin/activate"
echo "python = $(which python); torch = $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

cd "$REPO"

# ===== 2. Extract base 19f =====
extract_19f_base() {
  local archive="$DATA/dataset_512x256.tar.zst"
  local target="$BASE_DIR"
  if [[ -f "$target/data.npy" ]]; then
    echo "[2/5] base 19f already extracted: $target"
    return 0
  fi
  if [[ ! -f "$archive" ]]; then
    echo "[2/5] ERROR: $archive not found"
    return 2
  fi
  echo "[2/5 $(date +%H:%M:%S)] extracting $(basename "$archive") -> $target"
  mkdir -p "$target"
  tar --use-compress-program=unzstd -xf "$archive" -C "$target" --strip-components=1
  find "$target" -name "._*" -delete 2>/dev/null || true
  local found=$(find "$target" -maxdepth 4 -name data.npy -type f | head -1)
  if [[ -n "$found" && "$(dirname "$found")" != "$target" ]]; then
    local src=$(dirname "$found")
    echo "  flattening from $src -> $target"
    mv "$src"/* "$target"/ 2>/dev/null || true
    find "$target" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  fi
  ls -lh "$target" | head
  if [[ -f "$target/data.npy" ]]; then
    rm -f "$archive"
    echo "  archive removed; $(df -h /data | tail -1)"
  else
    echo "  ERROR: data.npy not in $target after extract"
    return 2
  fi
}
extract_19f_base

# ===== 3. Sanity check extra dir =====
if [[ ! -f "$EXTRA_DIR/data_extra.npy" ]]; then
  echo "[3/5] ERROR: $EXTRA_DIR/data_extra.npy not found"
  exit 2
fi
echo "[3/5] extra dir OK:"
ls -lh "$EXTRA_DIR"

# ===== 4. Build merged 33f dataset =====
if [[ ! -f "$OUT_DIR/dataset_info.json" ]]; then
  echo "[4/5 $(date +%H:%M:%S)] building merged 33f dataset -> $OUT_DIR"
  python scripts/build_v3_extra_with_time.py \
      --base-dir "$BASE_DIR" \
      --extra-dir "$EXTRA_DIR" \
      --out-dir "$OUT_DIR"
else
  echo "[4/5] $OUT_DIR/dataset_info.json already exists, skip build"
fi

df -h /data | tail -1
ls -lh "$OUT_DIR"

# ===== 5. Launch training =====
echo "[5/5 $(date +%H:%M:%S)] launching training"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi | head -20

python -m src.main experiments/wb2_512x256_33f_ar_v3 \
  --pretrained experiments/wb2_512x256_33f_ar_v3/best_model.pth \
  --resume

echo "[done $(date)]"
