#!/usr/bin/env bash
# Сброс темпа на ОСНОВНОЙ МОДЕЛИ СТАТЬИ. Продолжение с 28-й эпохи ещё на 12.
#
# Приём, давший суточной глобальной модели 4.8 п.п. на прогнозе: у сошедшейся
# модели резко снизить темп на стадии длинной развёртки. Основная модель стоит
# на плато с 15-й эпохи (0.01796) и к 29-й ухудшилась до 0.01834 — картина ровно
# та же, что была у v5 перед сбросом.
#
# Если сработает, улучшатся ПУБЛИКУЕМЫЕ числа статьи без переобучения с нуля.
# Опорные значения: t2m 1.32/1.53/1.59/1.66 °C, успешность по области 73.41 %
# (последняя эпоха, лог m33_last_roi).
#
# Запуск: bash scripts/_paper_run_lrdrop_krsk.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/lrdrop_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f; EXP=multires_krsk_33f_lrdrop
mkdir -p "$OUT"; exec >>"$OUT/lrdrop_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== СБРОС ТЕМПА: основная модель статьи ==="
pgrep -f "src.main" >/dev/null && { log "карта занята — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/lrdrop_krsk_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33R/data.npy" ]]; then
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  [[ ! -f "$GX/coords.npz" && -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]] && \
    cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
  log "собираю региональный датасет"
  python -u scripts/build_multires_russia_33f.py --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir "$GX" --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/lrdrop_krsk_build.log" 2>&1
  log "сборка rc=$?"
fi
[[ -f "$D33R/data.npy" ]] || { log "датасета нет — стоп"; exit 1; }

# Копируем состояние, чтобы не затереть чекпойнт, на котором стоят числа статьи
if [[ ! -f "experiments/$EXP/checkpoint.pth" ]]; then
  cp -p "experiments/$SRC/checkpoint.pth" "experiments/$EXP/checkpoint.pth" || { log "нет чекпойнта $SRC"; exit 1; }
  cp -p "experiments/$SRC/best_model.pth" "experiments/$EXP/best_model.pth" 2>/dev/null
  log "состояние перенесено из $SRC"
fi
log "START обучения $EXP (40 эпох, косинус, возобновление)"
python -u -m src.main "experiments/$EXP" --resume >> "$OUT/lrdrop_krsk_train.log" 2>&1
log "DONE обучение rc=$?"

log "START оценки по области интереса"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 \
    --save-sample-metrics "$OUT/lrdrop_krsk_roi_samples.npz" >> "$OUT/lrdrop_krsk_roi.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/lrdrop_krsk_roi.log" | tail -1)"
grep -E '^\s+(t2m|msl)\b' "$OUT/lrdrop_krsk_roi.log" | tail -2
log "сравнивать: основная модель t2m 1.32/1.53/1.59/1.66 °C, успешность 73.41 %"
log "=== ALL DONE ==="
