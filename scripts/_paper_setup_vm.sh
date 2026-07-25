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
# ЛИМИТ ДИСКА: платформа убивает job при ~240+ ГБ на локальном диске
# (reason=MLCoreBigNodeDiskUsage). Бюджет: глобальный 82 ГБ + merge 81 ГБ +
# Krsk 2.3 ГБ ≈ 165 ГБ. Всё лишнее в /data/datasets удаляем — восстановится из S3.
# Для 33f-линии дополнительно нужны global_extra (43 ГБ) и multires_krsk_33f (~60 ГБ):
# на такой VM ПОСЛЕ сборки merge удалить wb2_512x256_19f_ar (SLIM_AFTER_MERGE=1).
echo "[disk] /data занято: $(du -sh /data/datasets 2>/dev/null | cut -f1) (лимит платформы ~240 ГБ)"

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE_DIR=$DATA/wb2_512x256_19f_ar
KRSK_DIR=$DATA/region_krsk_61x41_19f_2010-2020_025deg
MERGE_DIR=$DATA/multires_krsk_19f_merge
cd "$REPO"

# ----- 1. venv -----
# Проверяем по pip (сломанный venv без ensurepip имеет python, но не pip)
if [[ ! -x "$VENV/bin/pip" ]]; then
  echo "[1/4 $(date +%H:%M:%S)] creating venv at $VENV"
  rm -rf "$VENV"
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    echo "[1/4] системный python3 без ensurepip; пробую conda-python"
    rm -rf "$VENV"
    CONDA_PY=$(ls /home/mlcore/conda/bin/python3* 2>/dev/null | head -1)
    if [[ -n "$CONDA_PY" ]] && "$CONDA_PY" -m venv "$VENV" 2>/dev/null; then
      echo "[1/4] venv создан conda-python ($CONDA_PY)"
    else
      echo "[1/4] пробую apt install python3.10-venv"
      rm -rf "$VENV"
      apt-get update -q >/dev/null 2>&1 || true
      apt-get install -y -q python3.10-venv >/dev/null 2>&1 \
        || sudo apt-get install -y -q python3.10-venv >/dev/null 2>&1 || true
      python3 -m venv "$VENV"
    fi
  fi
  echo "[1/4] venv created"
else
  echo "[1/4] venv already present"
fi
# Требования ставим/доставляем по факту: проверка именно import torch,
# а не наличие pip (venv мог остаться полуготовым после прерванной установки)
if ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "[1/4 $(date +%H:%M:%S)] installing requirements (torch отсутствует)"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r "$REPO/requirements.txt" \
    || "$VENV/bin/pip" install -q -r "$REPO/requirements.txt" \
         --extra-index-url https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple
  echo "[1/4] requirements installed"
fi
source "$VENV/bin/activate"
echo "python=$(which python); torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"

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
  # ВАЖНО: архив (~74 ГБ) удаляем сразу — платформа убивает job при >~240 ГБ
  # локального диска (reason=MLCoreBigNodeDiskUsage). Восстановится из S3 при перезапуске.
  rm -f "$ARC"
  echo "[2/4] global extracted; архив удалён ($(du -sh "$BASE_DIR" | cut -f1) в $BASE_DIR)"
else
  echo "[2/4] global already extracted"
fi

# ----- 3. extract Krsk regional (19f + extra) из paper-архива -----
# paper_krsk_datasets.tar.zst содержит обе папки:
#   region_krsk_61x41_19f_2010-2020_025deg/  (19 каналов, для merge)
#   region_krsk_61x41_extra_2010-2020_025deg/ (10 каналов @250/@1000, для 33f)
if [[ ! -f "$KRSK_DIR/data.npy" ]]; then
  echo "[3/4 $(date +%H:%M:%S)] extracting Krsk datasets from paper archive"
  PAPER_ARC=$(ls "$DATA"/paper_krsk_datasets.tar.zst 2>/dev/null | head -1)
  if [[ -n "$PAPER_ARC" ]]; then
    command -v zstd >/dev/null 2>&1 || { apt-get install -y -q zstd >/dev/null 2>&1 || true; }
    tar --use-compress-program=unzstd -xf "$PAPER_ARC" -C "$DATA"
    find "$DATA/region_krsk_61x41_19f_2010-2020_025deg" "$DATA/region_krsk_61x41_extra_2010-2020_025deg" \
         -name "._*" -delete 2>/dev/null || true
  else
    # fallback: отдельный tar/tar.gz регионального 19f
    mkdir -p "$KRSK_DIR"
    ARC=$(ls "$DATA"/region_krsk_61x41*.tar.gz "$DATA"/region_krsk_61x41*.tar 2>/dev/null | head -1)
    [[ -z "$ARC" ]] && { echo "[3/4] ERROR: ни paper_krsk_datasets.tar.zst, ни архив Krsk 61x41 не найдены"; ls -la "$DATA"; exit 3; }
    tar -xf "$ARC" -C "$KRSK_DIR" --strip-components=1 2>/dev/null || tar -xf "$ARC" -C "$KRSK_DIR"
    found=$(find "$KRSK_DIR" -maxdepth 4 -name data.npy -type f | head -1)
    if [[ -n "$found" && "$(dirname "$found")" != "$KRSK_DIR" ]]; then
      mv "$(dirname "$found")"/* "$KRSK_DIR"/ 2>/dev/null || true
    fi
  fi
  [[ -f "$KRSK_DIR/data.npy" ]] || { echo "[3/4] ERROR: $KRSK_DIR/data.npy не найден после распаковки"; ls -la "$DATA" | head -20; exit 3; }
  rm -f "$PAPER_ARC" 2>/dev/null || true
  echo "[3/4] Krsk datasets extracted: 19f=$(du -sh "$KRSK_DIR" | cut -f1), extra=$(du -sh "$DATA/region_krsk_61x41_extra_2010-2020_025deg" 2>/dev/null | cut -f1)"
else
  echo "[3/4] Krsk regional already extracted"
fi

# ----- 3b. распаковка чекпойнтов в репозиторий -----
if [[ ! -f "$REPO/experiments/multires_merge_freeze6_v2/best_model.pth" ]]; then
  CKPT_ARC=$(ls "$DATA"/paper_ckpts.tar.zst 2>/dev/null | head -1)
  if [[ -n "$CKPT_ARC" ]]; then
    echo "[3b $(date +%H:%M:%S)] extracting checkpoints -> $REPO"
    tar --use-compress-program=unzstd -xf "$CKPT_ARC" -C "$REPO"
    find "$REPO/experiments" -name "._*" -delete 2>/dev/null || true
    rm -f "$CKPT_ARC" 2>/dev/null || true
    echo "[3b] checkpoints:"
    find "$REPO/experiments" -name "*.pth" -newermt "-2 hours" -printf "     %s  %p\n" 2>/dev/null | head -10
  else
    echo "[3b] WARN: paper_ckpts.tar.zst не найден — инференс не запустится без чекпойнтов"
  fi
else
  echo "[3b] checkpoints already present"
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

# ----- 5. (опция) slim: освободить глобальный датасет после сборки merge -----
# Нужно на VM, где дополнительно собирается 33f (global_extra 43 ГБ + 33f 60 ГБ).
# Внимание: wb2_512x256_19f_ar требуется для эксперимента M4 (глобальный инференс v2),
# поэтому на VM линии статьи SLIM_AFTER_MERGE НЕ включать.
if [[ "${SLIM_AFTER_MERGE:-0}" == "1" && -f "$MERGE_DIR/data.npy" ]]; then
  # ВАЖНО: coords.npz глобальной сетки нужен сборщику 33f (у global_extra своих
  # координат нет) — сохраняем его в global_extra ДО удаления базового датасета.
  for _gx in "$DATA"/global_512x256_extra_*; do
    if [[ -d "$_gx" && ! -f "$_gx/coords.npz" && -f "$BASE_DIR/coords.npz" ]]; then
      cp -p "$BASE_DIR/coords.npz" "$_gx/coords.npz"
      echo "[5] coords.npz скопирован в $(basename "$_gx") (нужен для сборки 33f)"
    fi
  done
  echo "[5] SLIM: удаляю $BASE_DIR (глобальный 19f больше не нужен, merge собран)"
  rm -rf "$BASE_DIR"
  echo "[5] /data теперь: $(du -sh "$DATA" 2>/dev/null | cut -f1)"
fi
echo "[disk] итог: $(du -sh /data/datasets 2>/dev/null | cut -f1)"

echo "============================================================"
echo "[done $(date)] P0 complete. Чекпойнты моделей переносит пользователь:"
echo "  experiments/multires_merge_freeze6_v2/best_model.pth"
echo "  experiments/multires_nores_freeze6/{best_model.pth,checkpoint.pth}"
echo "  experiments/multires_nores_nofreeze/{best_model.pth,checkpoint.pth}"
echo "  experiments/region_krsk_cds_19f/'best_model (18).pth'  (для VM-B)"
echo "  (wb2_512x256_19f_ar_v2/best_model.pth — уже в git)"
echo "============================================================"
