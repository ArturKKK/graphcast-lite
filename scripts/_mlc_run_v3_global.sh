#!/usr/bin/env bash
# v3 GLOBAL multires pipeline: ставит venv, распаковывает архивы из /data/datasets/,
# собирает /data/datasets/wb2_512x256_23f_v3/data.npy и запускает обучение.
#
# Сценарии (выбор через переменную SCENARIO):
#   one_19f          — один архив dataset_512x256.tar.zst (19f) → +time_features → 23f
#   two_time_chunks  — два архива dataset_512x256_part1.tar.zst + _part2.tar.zst (split по времени)
#                      склеить time-axis, потом +time_features
#   two_feat_chunks  — dataset_512x256_19f.tar.zst + dataset_512x256_4f_time.tar.zst
#                      склеить feat-axis (без вычисления time forcing)
#
# Использование:
#   SCENARIO=one_19f bash /data/run_v3_global.sh
#
set -uo pipefail

SCENARIO="${SCENARIO:-one_19f}"
LOG=/data/logs/v3_global_pipeline.log
mkdir -p /data/logs /data/venvs /data/datasets
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] v3 GLOBAL pipeline, scenario=$SCENARIO"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_19F_DIR=$DATA/wb2_512x256_19f_ar          # промежуточный 19f
OUT_DIR=$DATA/wb2_512x256_34f_v3               # финальный 34f
mkdir -p "$OUT_DIR"

cd "$REPO"

# ===== 1. venv =====
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[1/5 $(date +%H:%M:%S)] creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
else
  echo "[1/5] venv already present"
fi
source "$VENV/bin/activate"
echo "python = $(which python); torch = $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# ===== helper: extract + flatten + cleanup =====
extract_and_flatten() {
  local archive="$1"
  local target="$2"
  echo "  extracting $(basename "$archive") -> $target"
  mkdir -p "$target"
  case "$archive" in
    *.tar.zst)
      tar --use-compress-program=unzstd -xf "$archive" -C "$target" --strip-components=1
      ;;
    *.tar.gz)
      tar -xzf "$archive" -C "$target" --strip-components=1
      ;;
    *.tar)
      tar -xf "$archive" -C "$target" --strip-components=1
      ;;
    *)
      echo "  unknown archive type: $archive"; return 1
      ;;
  esac
  find "$target" -name "._*" -delete 2>/dev/null || true
  # flatten if data.npy nested deeper
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
    echo "  archive removed"
  else
    echo "  ERROR: data.npy not in $target after extract"
    return 2
  fi
}

# ===== 2-4: подготовка датасета по сценарию =====
if [[ ! -f "$OUT_DIR/data.npy" ]]; then
  case "$SCENARIO" in

    one_19f)
      echo "[2/5 $(date +%H:%M:%S)] scenario=one_19f"
      if [[ ! -f "$BASE_19F_DIR/data.npy" ]]; then
        extract_and_flatten "$DATA/dataset_512x256.tar.zst" "$BASE_19F_DIR"
      else
        echo "  19f already extracted"
      fi
      echo "[3/5 $(date +%H:%M:%S)] adding time features 19f -> 23f (NB: only 19+4 — needs 30+4 for v3-34f path; see README scenario C)"
      python scripts/add_time_features.py --src "$BASE_19F_DIR" --dst "$OUT_DIR"
      echo "[4/5] cleanup 19f source"
      rm -f "$BASE_19F_DIR/data.npy"
      ;;

    two_time_chunks)
      echo "[2/5 $(date +%H:%M:%S)] scenario=two_time_chunks"
      P1=$DATA/wb2_512x256_19f_part1
      P2=$DATA/wb2_512x256_19f_part2
      if [[ ! -f "$P1/data.npy" ]]; then
        extract_and_flatten "$DATA/dataset_512x256_part1.tar.zst" "$P1"
      fi
      if [[ ! -f "$P2/data.npy" ]]; then
        extract_and_flatten "$DATA/dataset_512x256_part2.tar.zst" "$P2"
      fi
      echo "[3/5 $(date +%H:%M:%S)] concat time-axis -> $BASE_19F_DIR"
      python scripts/concat_time_chunks.py --parts "$P1" "$P2" --out "$BASE_19F_DIR"
      rm -f "$P1/data.npy" "$P2/data.npy"
      echo "[4/5 $(date +%H:%M:%S)] adding time features 19f -> 23f"
      python scripts/add_time_features.py --src "$BASE_19F_DIR" --dst "$OUT_DIR"
      rm -f "$BASE_19F_DIR/data.npy"
      ;;

    two_feat_chunks)
      echo "[2/5 $(date +%H:%M:%S)] scenario=two_feat_chunks"
      FT=$DATA/wb2_512x256_4f_time
      if [[ ! -f "$BASE_19F_DIR/data.npy" ]]; then
        extract_and_flatten "$DATA/dataset_512x256_19f.tar.zst" "$BASE_19F_DIR"
      fi
      if [[ ! -f "$FT/data.npy" ]]; then
        extract_and_flatten "$DATA/dataset_512x256_4f_time.tar.zst" "$FT"
      fi
      echo "[3/5 $(date +%H:%M:%S)] concat feat-axis -> $OUT_DIR"
      python scripts/concat_feat_chunks.py --base "$BASE_19F_DIR" --extra "$FT" --out "$OUT_DIR"
      echo "[4/5] cleanup"
      rm -f "$BASE_19F_DIR/data.npy" "$FT/data.npy"
      ;;

    *)
      echo "Unknown SCENARIO=$SCENARIO"; exit 1
      ;;
  esac
else
  echo "[2/5..4/5] $OUT_DIR/data.npy already exists, skipping prep"
fi

df -h /data | tail -1
ls -lh "$OUT_DIR"

# ===== 5. launch training =====
echo "[5/5 $(date +%H:%M:%S)] launching training"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd "$REPO"
nvidia-smi | head -20

python -m src.main experiments/wb2_512x256_34f_ar_v3

echo "[done $(date)]"
