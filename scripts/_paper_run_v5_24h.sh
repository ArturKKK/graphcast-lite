#!/usr/bin/env bash
# v5: глобальная модель с СУТОЧНЫМ шагом (вход t-24ч и t, цель t+24ч).
#
# Зачем. Ошибка растёт не столько от дальности горизонта, сколько от того, что
# каждый шаг кормит модель её собственной ошибкой: у v3 приземная температура
# идёт 0.85 -> 1.04 -> 1.12 -> 1.24 °C за четыре шестичасовых шага. А central
# эксперимент статьи — усвоение на 14 суток, это 56 шагов подряд. С суточным
# шагом их четырнадцать.
#
# Проверяемый вопрос: прогноз на сутки ОДНИМ шагом против 1.24 °C, полученных
# четырьмя. Если заметно лучше — накопление ошибки доминирует, и идею стоит
# разворачивать на всю статью, включая раздел про усвоение.
#
# Отличие от v3 ровно одно — data.time_stride=4. Всё остальное совпадает.
#
# ⚠️ Только одно обучение на видеокарте.
#
# Запуск: bash scripts/_paper_run_v5_24h.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/v5_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/v5_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33G=/data/datasets/wb2_512x256_33f_v3
BASE=/data/datasets/wb2_512x256_19f_ar
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
EXP=wb2_512x256_33f_ar_v5_24h

mkdir -p "$OUT"
MASTER="$OUT/v5_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== V5 (суточный шаг) START ==="

if pgrep -f "src.main" >/dev/null; then
  log "на карте уже идёт обучение — стоп, разберись вручную"
  exit 1
fi

# ── 1. Окружение и базовые датасеты ───────────────────────────────────
if [[ ! -x "$VENV/bin/python" || ! -d "$BASE" ]]; then
  log "нет окружения или базового датасета — подготовка (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/v5_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "venv не поднялся — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ── 2. Глобальный 33-канальный датасет ────────────────────────────────
if [[ ! -f "$D33G/data.npy" ]]; then
  for d in "$BASE" "$GEXTRA"; do
    [[ -d "$d" ]] || { log "НЕТ ИСХОДНИКА $d — сборка невозможна"; exit 1; }
  done
  log "собираю глобальный 33-канальный датасет (часы CPU)"
  python -u scripts/build_v3_extra_with_time.py \
      --base-dir "$BASE" --extra-dir "$GEXTRA" --out-dir "$D33G" \
      >> "$OUT/v5_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33G" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33G/data.npy" ]] || { log "глобального датасета нет — стоп"; exit 1; }

# ── 3. Обучение ───────────────────────────────────────────────────────
log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения"
log "START обучения $EXP (шаг 24 ч, с нуля, без pretrained)"
python -u -m src.main "experiments/$EXP" >> "$OUT/v5_train.log" 2>&1
log "DONE обучение rc=$?"

# ── 4. Оценка: горизонты +24/+48/+72 ч ────────────────────────────────
# ВАЖНО: у этой модели шаг 24 ч, поэтому --ar-steps 3 даёт +24/+48/+72,
# а не +18. predict.py подписывает горизонты по data.time_stride.
log "START оценки (3 шага = +24/+48/+72 ч)"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33G" \
    --split test_only --ar-steps 3 --max-samples 200 --per-channel --no-save \
    --save-sample-metrics "$OUT/v5_global_samples.npz" \
    >> "$OUT/v5_global.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/v5_global.log" | tail -1)"
grep -E '^\s+(t2m|z500|msl)\b' "$OUT/v5_global.log" | tail -3

log "--- ГЛАВНОЕ СРАВНЕНИЕ ---"
log "v5, +24 ч ОДНИМ шагом — см. первую колонку таблицы по горизонтам выше"
log "v3, +24 ч ЧЕТЫРЬМЯ шагами: t2m 1.24 °C (200 сроков, 11.08.2026)"
log "если v5 заметно ниже 1.24 — накопление ошибки доминирует, идею разворачиваем"
log "=== ALL DONE ==="
