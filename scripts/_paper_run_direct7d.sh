#!/usr/bin/env bash
# ПРЯМОЙ прогноз на 7 суток одним применением модели.
#
# Крайняя точка тренда, который мы мерили всю неделю по приземной температуре:
#   56 шагов (v3)      -> 5.04 °C, ХУЖЕ инерции (4.96)
#   14 шагов (v5_ext)  -> 4.30 °C, на 13 % лучше инерции
#    7 шагов (v5_ext)  -> 3.00 °C на +168 ч, на 28 % лучше инерции
#    1 шаг             -> ?
#
# Чем меньше применений, тем лучше на дальних сроках. Один шаг мы ни разу не
# пробовали, а это ноль накопления ошибки по определению.
#
# Ключевое отличие от v5: шаги входа и цели РАЗВЯЗАНЫ. Вход остаётся
# шестичасовым (t−6ч и t), цель — t+168ч. У суточной модели вход шёл через
# сутки, и она теряла информацию о тенденции — отсюда её проигрыш на +24 ч
# (1.31 против 1.24 °C). Здесь итерировать не нужно, поэтому частый вход можно
# сохранить.
#
# Запуск: bash scripts/_paper_run_direct7d.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/direct7d_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33G=/data/datasets/wb2_512x256_33f_v3
BASE=/data/datasets/wb2_512x256_19f_ar
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
EXP=wb2_512x256_33f_direct_7d
mkdir -p "$OUT"; exec >>"$OUT/direct7d_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ПРЯМОЙ ПРОГНОЗ НА 7 СУТОК ==="
pgrep -f "src.main" >/dev/null && { log "карта занята — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -d "$BASE" ]]; then
  log "подготовка окружения и датасетов (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/direct7d_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33G/data.npy" ]]; then
  for d in "$BASE" "$GEXTRA"; do [[ -d "$d" ]] || { log "нет исходника $d"; exit 1; }; done
  log "собираю глобальный 33-канальный датасет"
  python -u scripts/build_v3_extra_with_time.py --base-dir "$BASE" --extra-dir "$GEXTRA" \
      --out-dir "$D33G" >> "$OUT/direct7d_build.log" 2>&1
  log "сборка rc=$?"
fi
[[ -f "$D33G/data.npy" ]] || { log "датасета нет — стоп"; exit 1; }

RESUME=""; [[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "найден чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD)"
log "START обучения $EXP (24 эпохи, вход 6 ч, цель +168 ч, косинус) $RESUME"
python -u -m src.main "experiments/$EXP" $RESUME >> "$OUT/direct7d_train.log" 2>&1
log "DONE обучение rc=$?"

# Оценка: ОДНО применение, авторегрессии нет
log "START оценки (1 шаг = +168 ч)"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33G" --split test_only \
    --ar-steps 1 --max-samples 200 --per-channel --no-save \
    --save-sample-metrics "$OUT/direct7d_samples.npz" >> "$OUT/direct7d.log" 2>&1
log "DONE оценка rc=$?"
grep -E '^\s+t2m' "$OUT/direct7d.log" | tail -2
log "--- С ЧЕМ СРАВНИВАТЬ на +168 ч (t2m) ---"
log "  инерция:            4.17 °C"
log "  v3, 28 шагов:       3.82 °C  (лучше инерции на 8 %)"
log "  v5_ext, 7 шагов:    3.00 °C  (на 28 %)"
log "  прямой, 1 шаг:      см. выше"
log "=== ALL DONE ==="
