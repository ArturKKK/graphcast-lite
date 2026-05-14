#!/usr/bin/env bash
# Russia multires 33f training pipeline. VM = graphcast_v4-hts83x.
#
# Шаги:
#   1. venv + git pull
#   2. убедиться что multires_russia_19f есть (если нет — собрать как в _mlc_run_russia.sh:
#      нужны base 19f tar.zst + Russia 0.25° regional tar; адаптировать команды или
#      скопировать датасет с v3 VM через mlc scp / external storage).
#   3. убедиться что global_512x256_extra_2010-2021_07deg есть (он был на v4 для v3 GLOBAL).
#   4. собрать multires_russia_33f через scripts/build_multires_russia_33f.py
#   5. запустить training с --pretrained от v3 GLOBAL best
#
# Запуск:
#   mlc job exec graphcast_v4-hts83x -- bash -lc '(setsid nohup bash scripts/_mlc_run_russia_33f_v4.sh </dev/null >/dev/null 2>&1 &); sleep 3; tail -20 /data/logs/russia_33f.log'

set -uo pipefail

LOG=/data/logs/russia_33f.log
mkdir -p /data/logs
exec >>"$LOG" 2>&1

trap 'echo "[ERR $(date +%H:%M:%S)] failure at line $LINENO ($BASH_COMMAND)"' ERR

echo "============================================================"
echo "[start $(date)] Russia multires 33f (v3 finetune) training"
echo "============================================================"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
MULTI_19F=$DATA/multires_russia_19f
MULTI_33F=$DATA/multires_russia_33f
EXTRA_DIR=$DATA/global_512x256_extra_2010-2021_07deg
EXP=experiments/multires_russia_33f_v3_noroi
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

cd "$REPO"

# --- venv + repo ---
if [[ ! -d "$REPO/.git" ]]; then
  echo "[1/5] cloning repo"
  mkdir -p /workdir
  git clone -b main-arthur https://github.com/ArturKKK/graphcast-lite.git "$REPO"
else
  echo "[1/5] pulling repo"
  (cd "$REPO" && git pull --rebase --autostash) || echo "  pull failed (continuing)"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
fi
source "$VENV/bin/activate"
echo "python = $(which python); torch = $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# --- sanity: multires_russia_19f ---
if [[ ! -f "$MULTI_19F/data.npy" ]]; then
  echo "[ERR] $MULTI_19F/data.npy отсутствует. Этот VM (v4) изначально не содержит Russia dataset."
  echo "      Варианты:"
  echo "        (а) скопировать с v3 VM:  mlc scp <v3>:$MULTI_19F $MULTI_19F"
  echo "        (б) собрать с нуля: понадобятся base 19f tar.zst и Russia regional tar."
  echo "      Сейчас прерываемся, нужны действия пользователя."
  exit 2
fi

# --- sanity: global extra (должен быть на v4 от v3 GLOBAL) ---
if [[ ! -f "$EXTRA_DIR/data_extra.npy" ]]; then
  echo "[ERR] $EXTRA_DIR/data_extra.npy отсутствует. Требуется тот же файл, что использовался для v3 GLOBAL."
  exit 3
fi

# --- sanity: v3 GLOBAL pretrained ---
if [[ ! -f "$PRETRAINED" ]]; then
  echo "[ERR] $PRETRAINED missing. На v4 он должен быть после v3 обучения."
  exit 4
fi

# --- step 2: build multires_russia_33f if not yet ---
if [[ ! -f "$MULTI_33F/data_extra.npy" ]]; then
  echo "[2/5 $(date +%H:%M:%S)] building multires_russia_33f"
  python scripts/build_multires_russia_33f.py \
    --multires-dir "$MULTI_19F" \
    --extra-dir    "$EXTRA_DIR" \
    --out-dir      "$MULTI_33F"
  echo "[2/5] build done"
  ls -lh "$MULTI_33F"
else
  echo "[2/5] multires_russia_33f already built"
fi

# --- step 3: launch training ---
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
