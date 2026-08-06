#!/usr/bin/env bash
# S1 — диагностика стыка глобальной и региональной частей графа.
#
# Закрывает главную недоказанную заявку статьи. В тексте утверждается, что
# жёсткая склейка узлов не требует отдельной процедуры сшивания, а переходный
# слой модель формирует сама при обмене сообщениями. Никакой проверки этого
# утверждения до сих пор не было.
#
# Что делает: прогоняет флагманскую модель на merge-датасете с сохранением
# предсказаний, чтобы затем построить (отдельным скриптом, на CPU):
#   - карту приземной температуры на +24 ч через границу вставки;
#   - зависимость RMSE от расстояния до границы вставки.
# Если у стыка обнаружится всплеск ошибки — это тоже публикуемо, формулировку
# смягчаем до «контролируемый переходный слой».
#
# ⚠️ ДИСКИ: тяжёлый .pt пишем в /data (там место есть, но всё стирается при
# рестарте и действует лимит платформы ~240 ГБ). В /workdir нельзя — там квота
# около 8 ГБ, превышение гасит виртуалку.
#
# Размер: N сроков x 4 горизонта x 19 каналов x 133 279 узлов x 4 байта, и всё
# это в двух экземплярах (предсказание + истина). При N=100 выходит около 8 ГБ.
#
# Запуск: nohup setsid bash scripts/_paper_run_seam.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/seam_master.log
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
MERGE=/data/datasets/multires_krsk_19f_merge
ROI="50 60 83 98"
MAXN=100

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/seam_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== SEAM START (commit $GIT_COMMIT), MAXN=$MAXN ==="
[[ -d "$MERGE" ]] || { log "НЕТ ДАТАСЕТА $MERGE"; exit 1; }
log "свободно на /data: $(df -h /data | tail -1 | awk '{print $4}')"

TAG=seam_flagship_preds
LF="$OUT/$TAG.log"; PRED="$HEAVY/seam_flagship_preds.pt"
CMD="python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir $MERGE --split test_only --ar-steps 4 --max-samples $MAXN --per-channel --region $ROI --save $PRED"
{
  echo "### PROVENANCE ###############################################"
  echo "# tag: $TAG | started: $(date -Iseconds) | host: $(hostname)"
  echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# dataset: $MERGE"
  echo "# ckpt md5: $(md5sum experiments/multires_merge_freeze6_v2/best_model.pth | cut -d' ' -f1)"
  echo "# COMMAND:"; echo "#   $CMD"
  echo "##############################################################"; echo
} > "$LF"
log "START $TAG"
eval "$CMD" >> "$LF" 2>&1
rc=$?
log "DONE  $TAG rc=$rc | $(grep -oE 'skill=[-0-9.]+%' "$LF" | tail -1)"
log "размер предсказаний: $(du -h "$PRED" 2>/dev/null | cut -f1)"
log "=== SEAM DONE ==="
