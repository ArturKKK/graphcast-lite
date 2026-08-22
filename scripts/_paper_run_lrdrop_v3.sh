#!/usr/bin/env bash
# КОНТРОЛЬ к результату про суточный шаг: тот же сброс темпа на ШЕСТИЧАСОВОЙ
# глобальной модели.
#
# Зачем. Суточная модель получила 15 дополнительных эпох со сбросом темпа,
# шестичасовая — ничего. Значит сравнение v3 против v5_ext нечестное: часть
# выигрыша в 17 п.п. может быть не от шага, а просто от лишнего обучения.
# Без этого контроля вывод про суточный шаг публиковать нельзя.
#
# Полного чекпойнта v3 не сохранилось, только веса, поэтому не возобновление, а
# дообучение: старт от best_model.pth, сразу на четырёх шагах развёртки
# (initial_ar_steps=4), темп 7.5e-5 — четверть от прежних 3e-4 — с косинусным
# спадом до нуля за 8 эпох.
#
# Опорные числа v3 на пяти сутках: t2m 2.97 °C, успешность 26.0 % на +120 ч.
# Опорные числа v5_ext:            t2m 2.46 °C, успешность 43.3 %.
#
# Запуск: bash scripts/_paper_run_lrdrop_v3.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/lrdrop_v3_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33G=/data/datasets/wb2_512x256_33f_v3
BASE=/data/datasets/wb2_512x256_19f_ar
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
EXP=wb2_512x256_33f_ar_v3_lrdrop
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth
mkdir -p "$OUT"; exec >>"$OUT/lrdrop_v3_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== КОНТРОЛЬ: сброс темпа на шестичасовой глобальной ==="
# Сторож ловит и обучение, и инференс: 22.08.2026 чуть не запустили дожиг
# поверх пятисуточной развёртки на той же карте — прежний сторож знал
# только про src.main и predict.py бы не заметил.
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -d "$BASE" ]]; then
  log "подготовка окружения и датасетов (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/lrdrop_v3_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33G/data.npy" ]]; then
  for d in "$BASE" "$GEXTRA"; do [[ -d "$d" ]] || { log "нет исходника $d"; exit 1; }; done
  log "собираю глобальный 33-канальный датасет"
  python -u scripts/build_v3_extra_with_time.py --base-dir "$BASE" --extra-dir "$GEXTRA" \
      --out-dir "$D33G" >> "$OUT/lrdrop_v3_build.log" 2>&1
  log "сборка rc=$?"
fi
[[ -f "$D33G/data.npy" ]] || { log "датасета нет — стоп"; exit 1; }
[[ -f "$PRETRAINED" ]] || { log "нет весов v3 — стоп"; exit 1; }

RESUME=""; [[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "найден чекпойнт — продолжаю"; }
log "START обучения $EXP (8 эпох, AR=4, темп 7.5e-5 с косинусом) $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/lrdrop_v3_train.log" 2>&1
log "DONE обучение rc=$?"

log "START оценки на пяти сутках (20 шагов)"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33G" --split test_only \
    --ar-steps 20 --max-samples 200 --per-channel --no-save \
    --save-sample-metrics "$OUT/lrdrop_v3_d5_samples.npz" >> "$OUT/lrdrop_v3_d5.log" 2>&1
log "DONE оценка rc=$?"
grep -E '^\s+\+(24|72|120)h:' "$OUT/lrdrop_v3_d5.log" | tail -3
log "--- ГЛАВНОЕ СРАВНЕНИЕ на +120 ч ---"
log "  v3 без сброса темпа: t2m 2.97 °C, успешность 26.0 %"
log "  v5_ext, суточный шаг: t2m 2.46 °C, успешность 43.3 %"
log "если v3 со сбросом заметно подтянулась — часть выигрыша была не от шага"
log "=== ALL DONE ==="
