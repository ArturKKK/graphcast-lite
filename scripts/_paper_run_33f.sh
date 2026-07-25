#!/usr/bin/env bash
# 33f-линия Красноярска: сборка датасета (19 базовых + 10 plev + 4 time) → обучение.
# Уровни @250/@1000 в ROI — РЕАЛЬНЫЕ 0.25° (region_krsk_61x41_extra), вне ROI — билинейные
# из глобального extra. Схема обучения совпадает с дипломной 19f (max_ar=4, 32 эпохи, freeze6),
# чтобы сравнение 19f vs 33f отличалось только набором каналов.
# Запуск: nohup setsid bash scripts/_paper_run_33f.sh </dev/null >/dev/null 2>&1 &
# Лог:    /data/paper_results/m33_master.log
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
OUT=/data/paper_results
MERGE=$DATA/multires_krsk_19f_merge
GEXTRA=$DATA/global_512x256_extra_2010-2021_07deg
REXTRA=$DATA/region_krsk_61x41_extra_2010-2020_025deg
OUT33=$DATA/multires_krsk_33f
EXP=experiments/multires_krsk_33f
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

mkdir -p "$OUT"
MASTER="$OUT/m33_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")

log "=== 33f LINE START (commit $GIT_COMMIT) ==="
log "disk: $(du -sh $DATA | cut -f1)"

# ---------- проверки входов ----------
for p in "$MERGE/data.npy" "$GEXTRA" "$REXTRA/data_extra.npy" "$PRETRAINED"; do
  [[ -e "$p" ]] || { log "FATAL: нет $p"; exit 2; }
done
log "inputs OK; pretrained md5=$(md5sum "$PRETRAINED" | cut -d' ' -f1)"

# ---------- 1. сборка 33f ----------
if [[ ! -f "$OUT33/data_extra.npy" ]]; then
  BL="$OUT/m33_build.log"
  CMD="python -u scripts/build_multires_russia_33f.py --multires-dir $MERGE --extra-dir $GEXTRA --region-extra-dir $REXTRA --out-dir $OUT33"
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag:        m33_build"
    echo "# started:    $(date -Iseconds)"
    echo "# host:       $(hostname)"
    echo "# git commit: $GIT_COMMIT"
    echo "# merge:      $(tr -d '\n ' < "$MERGE/dataset_info.json" | cut -c1-260)"
    echo "# global-extra: $(tr -d '\n ' < "$GEXTRA/dataset_info_extra.json" 2>/dev/null | cut -c1-260)"
    echo "# region-extra: $(tr -d '\n ' < "$REXTRA/dataset_info_extra.json" | cut -c1-260)"
    echo "# COMMAND:"; echo "#   $CMD"
    echo "##############################################################"; echo
  } > "$BL"
  log "START build 33f → $BL"
  eval "$CMD" >> "$BL" 2>&1
  rc=$?
  echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$BL"
  [[ -f "$OUT33/data_extra.npy" ]] || { log "FATAL: сборка 33f не удалась (rc=$rc), см. $BL"; exit 3; }
  log "DONE build 33f: $(du -sh "$OUT33" | cut -f1) | disk: $(du -sh $DATA | cut -f1)"
else
  log "33f dataset already present: $(du -sh "$OUT33" | cut -f1)"
fi

# ---------- 2. обучение ----------
TL="$OUT/m33_train.log"
CMD="python -u -m src.main $EXP --pretrained $PRETRAINED"
{
  echo "### PROVENANCE ###############################################"
  echo "# tag:        m33_train"
  echo "# started:    $(date -Iseconds)"
  echo "# host:       $(hostname)"
  echo "# git commit: $GIT_COMMIT"
  echo "# gpu:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "# dataset:    $OUT33"
  echo "#   info:     $(tr -d '\n ' < "$OUT33/dataset_info.json" 2>/dev/null | cut -c1-300)"
  echo "# config md5: $(md5sum $EXP/config.json | cut -d' ' -f1)"
  echo "# pretrained: $(md5sum "$PRETRAINED" | cut -d' ' -f1)  $PRETRAINED"
  echo "# COMMAND:"; echo "#   $CMD"
  echo "##############################################################"; echo
} > "$TL"
log "START train 33f → $TL (ожидание ~36-42 ч)"
eval "$CMD" >> "$TL" 2>&1
rc=$?
echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$TL"
log "DONE train 33f rc=$rc"
tail -3 "$EXP/training_log.txt" 2>/dev/null
log "=== 33f LINE DONE ==="
