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
# --pretrained обязателен, хотя веса тут же перезапишутся чекпойнтом.
# Основная модель обучалась от глобальной v3 с заморозкой процессора, поэтому у
# её оптимизатора ДВЕ группы параметров (интерфейсы и процессор с пониженным
# темпом), и обе лежат в чекпойнте. Без --pretrained main.py создаёт оптимизатор
# с одной группой, и возобновление падает с
# "loaded state dict has a different number of parameter groups"
# (наступили 15.08.2026, потеряли полтора часа).
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth
[[ -f "$PRETRAINED" ]] || { log "нет $PRETRAINED — нужен для структуры оптимизатора"; exit 1; }
log "START обучения $EXP (40 эпох, косинус, возобновление)"
# --reset-patience обязателен: терпение приходит из чекпойнта уже
# исчерпанным (13 при пороге 12), и без обнуления ранняя остановка
# срабатывает после первой же эпохи — 16.08.2026 приём так и не проверили.
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" --resume --reset-patience \
    >> "$OUT/lrdrop_krsk_train.log" 2>&1
log "DONE обучение rc=$?"

# ВАЖНО: оценивать надо ПОСЛЕДНЮЮ эпоху, а не best_model.pth.
# Лучшее значение (0.01796) досталось в наследство от исходного прогона вместе с
# чекпойнтом, и дообучение его не побило. Значит best_model.pth так и остался
# весами 15-й эпохи ИСХОДНОЙ модели — оценка мерила бы её, а не дообученную.
mkdir -p /data/paper_heavy
python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_lrdrop_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch")
    print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
else:
    print("[prep] чекпойнта нет")
PYEOF
CK=""
[[ -f /data/paper_heavy/krsk_lrdrop_last.pth ]] && CK="--ckpt /data/paper_heavy/krsk_lrdrop_last.pth"
log "START оценки по области интереса $CK"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 $CK \
    --save-sample-metrics "$OUT/lrdrop_krsk_roi_samples.npz" >> "$OUT/lrdrop_krsk_roi.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/lrdrop_krsk_roi.log" | tail -1)"
grep -E '^\s+(t2m|msl)\b' "$OUT/lrdrop_krsk_roi.log" | tail -2
log "сравнивать: основная модель t2m 1.32/1.53/1.59/1.66 °C, успешность 73.41 %"
log "=== ALL DONE ==="
