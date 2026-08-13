#!/usr/bin/env bash
# Многоуровневый меш с логарифмическим кодированием длины ребра.
#
# Что проверяем. 11.08 многоуровневый меш проиграл базовому 21 %, и мы нашли
# измеримую причину-кандидата: признаки рёбер нормируются на максимальную длину
# В ТЕКУЩЕМ графе. При уровнях [4, 6] максимум 419 км, и рёбра ложатся на 0,25 и
# 1,0. При уровнях [0..6] максимум 6699 км — и 98,4 % рёбер оказываются в полоске
# шириной 0,06 у нуля, а шкалу им задают тридцать рёбер нулевого уровня, то есть
# 0,018 % от общего числа. Кодировщику приходится различать содержательные
# масштабы (105, 209, 419 км) по почти одинаковому входу.
#
# Режим unit_log разводит это: три числа — единичный вектор направления,
# четвёртое — логарифм длины в [0, 1]. Каждый уровень дробления ровно вдвое
# короче предыдущего, поэтому в логарифме уровни встают равномерно (0, 1/6,
# 2/6 … 1) при любом наборе уровней. Размерность прежняя, конфиги не меняются.
#
# С ЧЕМ СРАВНИВАТЬ. Только с прежним многоуровневым прогоном
# (multires_krsk_33f_multimesh): оба дообучаются от глобальной v3, обученной в
# режиме legacy, то есть процессор в обоих случаях стартует с незнакомого
# распределения признаков и загрязнение одинаковое. С основной моделью статьи
# сравнивать НЕЛЬЗЯ.
#
# Опорные числа прежнего прогона (mean-агрегация, legacy-признаки):
#   до обучения 0.21405 | эпоха 1 0.03460 | эпоха 8 0.01929 | эпоха 15 0.01841
# Если logedge стартует заметно ниже 0.21405 и идёт быстрее — причина найдена.
#
# Запуск: bash scripts/_paper_run_logedge.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/logedge_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/logedge_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
EXP=multires_krsk_33f_multimesh_logedge
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

mkdir -p "$OUT"
MASTER="$OUT/logedge_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== MULTIMESH + LOGEDGE START ==="

if pgrep -f "src.main" >/dev/null; then
  log "на карте уже идёт обучение — стоп"
  exit 1
fi

# ── Окружение и датасет ───────────────────────────────────────────────
# /data стирается при выключении виртуалки, поэтому проверяем всегда.
if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "нет окружения или регионального датасета — подготовка (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/logedge_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "venv не поднялся — стоп"; exit 1; }
export PYTHONPATH="$REPO"

if [[ ! -f "$D33R/data.npy" ]]; then
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  if [[ ! -f "$GX/coords.npz" && -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]]; then
    cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
    log "coords.npz скопирован в global_extra"
  fi
  log "собираю региональный 33-канальный датасет (часы CPU)"
  python -u scripts/build_multires_russia_33f.py \
      --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir "$GX" \
      --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/logedge_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "датасета нет — стоп"; exit 1; }
[[ -f "$PRETRAINED" ]] || { log "нет весов глобальной v3 — стоп"; exit 1; }

# ── Обучение ──────────────────────────────────────────────────────────
RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "найден чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения"
log "START обучения $EXP $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/logedge_train.log" 2>&1
log "DONE обучение rc=$?"

# ── Оценка по области интереса ────────────────────────────────────────
log "START оценки на тестовом окне"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" \
    --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
    --region 50 60 83 98 --save-sample-metrics "$OUT/logedge_roi_samples.npz" \
    >> "$OUT/logedge_roi.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/logedge_roi.log" | tail -1)"
grep -E '^\s+(t2m|msl)\b' "$OUT/logedge_roi.log" | tail -2
log "сравнивать с прежним многоуровневым прогоном, НЕ с основной моделью"
log "=== ALL DONE ==="
