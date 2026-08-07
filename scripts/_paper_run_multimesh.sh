#!/usr/bin/env bash
# v4: обучение multires_krsk_33f_multimesh — полный многоуровневый меш (0..6).
#
# Единственное отличие от основной модели статьи — набор уровней меша.
# Вершин столько же, добавляются рёбра нижних уровней: +25 % рёбер, ноль новых
# параметров. Уровень 0 даёт рёбра длиной около 7000 км, тогда как сейчас самое
# длинное ребро около 440 км, из-за чего дальние связи модель строить не может.
#
# Веса глобальной v3 переносятся: формы обучаемых параметров от набора рёбер не
# зависят. Геометрические буферы иного размера отбрасываются при загрузке — в
# логе будет строка «Отброшены записи с иным размером», это ожидаемо.
#
# Ждёт готовности датасета, поэтому можно запускать одновременно со скриптом
# заморозки: тот соберёт данные, этот дождётся.
#
# Запуск: nohup setsid bash scripts/_paper_run_multimesh.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/multimesh_master.log
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33=/data/datasets/multires_krsk_33f
EXP=multires_krsk_33f_multimesh
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

mkdir -p "$OUT"
MASTER="$OUT/multimesh_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== MULTIMESH (v4) START ==="

# Ждём датасет и окружение — их готовит скрипт заморозки, если запущен параллельно
for i in $(seq 1 240); do
  [[ -f "$D33/data.npy" && -x "$VENV/bin/python" ]] && break
  [[ $((i % 20)) -eq 0 ]] && log "жду датасет и окружение… ($((i/2)) мин)"
  sleep 30
done
[[ -f "$D33/data.npy" ]] || { log "датасета так и нет — стоп"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения — модель будет невоспроизводима"
log "GPU до старта: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

log "START обучения $EXP"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" >> "$OUT/multimesh_train.log" 2>&1
log "DONE обучение rc=$?"

log "START оценки на тестовом окне"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33" \
    --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
    --region 50 60 83 98 --save-sample-metrics "$OUT/multimesh_roi_samples.npz" \
    >> "$OUT/multimesh_roi.log" 2>&1
log "DONE оценка | $(grep -oE 'skill=[-0-9.]+%' "$OUT/multimesh_roi.log" | tail -1) | $(grep -E '^\s+t2m' "$OUT/multimesh_roi.log" | tail -1 | tr -s ' ' | cut -c1-60)"
log "сравнивать с multires_krsk_33f: t2m 1.32/1.53/1.59/1.66 °C, успешность 75.31 %"
log "=== ALL DONE ==="
