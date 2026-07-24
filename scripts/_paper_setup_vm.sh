#!/usr/bin/env bash
# P0 для статьи: подготовка VM (venv + датасеты + merge) БЕЗ обучения.
# Одинаков для обеих VM (A — «ядро AR-4», B — «AR-28 + абляции»).
# Запуск на VM:  setsid nohup bash scripts/_paper_setup_vm.sh </dev/null >/dev/null 2>&1 &
# Лог:           /data/logs/paper_setup.log
set -uo pipefail
LOG=/data/logs/paper_setup.log
mkdir -p /data/logs /data/venvs /data/datasets /data/paper_results
exec >>"$LOG" 2>&1
trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] paper P0 setup"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_DIR=$DATA/wb2_512x256_19f_ar
KRSK_DIR=$DATA/region_krsk_61x41_19f_2010-2020_025deg
MERGE_DIR=$DATA/multires_krsk_19f_merge
cd "$REPO"

# ----- 1. venv -----
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[1/4 $(date +%H:%M:%S)] creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
  echo "[1/4] venv ready"
else
  echo "[1/4] venv already present"
fi
source "$VENV/bin/activate"
echo "python=$(which python); torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# ----- 2. extract global 512x256 (tar.zst) -----
if [[ ! -f "$BASE_DIR/data.npy" ]]; then
  echo "[2/4 $(date +%H:%M:%S)] extracting global 512x256"
  mkdir -p "$BASE_DIR"
  zstd_bin=$(command -v zstd || echo "")
  if [[ -z "$zstd_bin" ]]; then
    apt-get install -y -q zstd >/dev/null 2>&1 || pip install -q zstandard
  fi
  ARC=$(ls "$DATA"/dataset_512x256.tar.zst "$DATA"/wb2_512x256*.tar.zst 2>/dev/null | head -1)
  [[ -z "$ARC" ]] && { echo "[2/4] ERROR: архив global 512x256 не найден в $DATA"; ls -la "$DATA"; exit 2; }
  tar --use-compress-program=unzstd -xf "$ARC" -C "$BASE_DIR" --strip-components=1
  find "$BASE_DIR" -name "._*" -delete 2>/dev/null || true
  found=$(find "$BASE_DIR" -maxdepth 4 -name data.npy -type f | head -1)
  if [[ -n "$found" && "$(dirname "$found")" != "$BASE_DIR" ]]; then
    mv "$(dirname "$found")"/* "$BASE_DIR"/ 2>/dev/null || true
    find "$BASE_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  fi
  [[ -f "$BASE_DIR/data.npy" ]] || { echo "[2/4] ERROR: data.npy не найден после распаковки"; exit 2; }
  echo "[2/4] global extracted (архив НЕ удаляю — место есть)"
else
  echo "[2/4] global already extracted"
fi

# ----- 3. extract Krsk regional 0.25 (tar/tar.gz) -----
if [[ ! -f "$KRSK_DIR/data.npy" ]]; then
  echo "[3/4 $(date +%H:%M:%S)] extracting Krsk regional"
  mkdir -p "$KRSK_DIR"
  ARC=$(ls "$DATA"/region_krsk_61x41*.tar.gz "$DATA"/region_krsk_61x41*.tar "$KRSK_DIR"/region_krsk_61x41*.tar* 2>/dev/null | head -1)
  [[ -z "$ARC" ]] && { echo "[3/4] ERROR: архив Krsk 61x41 не найден"; ls -la "$DATA"; exit 3; }
  tar -xf "$ARC" -C "$KRSK_DIR" --strip-components=1 2>/dev/null || tar -xf "$ARC" -C "$KRSK_DIR"
  find "$KRSK_DIR" -name "._*" -delete 2>/dev/null || true
  found=$(find "$KRSK_DIR" -maxdepth 4 -name data.npy -type f | head -1)
  if [[ -n "$found" && "$(dirname "$found")" != "$KRSK_DIR" ]]; then
    mv "$(dirname "$found")"/* "$KRSK_DIR"/ 2>/dev/null || true
    find "$KRSK_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  fi
  [[ -f "$KRSK_DIR/data.npy" ]] || { echo "[3/4] ERROR: data.npy не найден после распаковки"; exit 3; }
  echo "[3/4] Krsk regional extracted"
else
  echo "[3/4] Krsk regional already extracted"
fi

# ----- 4. build merge dataset (ROI Красноярск 50-60N 83-98E) -----
if [[ ! -f "$MERGE_DIR/data.npy" ]]; then
  echo "[4/4 $(date +%H:%M:%S)] building merge dataset -> $MERGE_DIR"
  python scripts/build_multires_dataset.py \
      --global-dir "$BASE_DIR" \
      --region-dir "$KRSK_DIR" \
      --roi 50 60 83 98 \
      --mode merge \
      --out-dir "$MERGE_DIR"
  [[ -f "$MERGE_DIR/data.npy" ]] || { echo "[4/4] ERROR: merge не собрался"; exit 4; }
  echo "[4/4] merge ready: $(du -sh "$MERGE_DIR" | cut -f1)"
else
  echo "[4/4] merge already present"
fi

echo "============================================================"
echo "[done $(date)] P0 complete. Чекпойнты моделей переносит пользователь:"
echo "  experiments/multires_merge_freeze6_v2/best_model.pth"
echo "  experiments/multires_nores_freeze6/{best_model.pth,checkpoint.pth}"
echo "  experiments/multires_nores_nofreeze/{best_model.pth,checkpoint.pth}"
echo "  experiments/region_krsk_cds_19f/'best_model (18).pth'  (для VM-B)"
echo "  (wb2_512x256_19f_ar_v2/best_model.pth — уже в git)"
echo "============================================================"
