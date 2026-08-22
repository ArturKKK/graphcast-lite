#!/usr/bin/env bash
# Основная модель статьи на ДРУГОМ СИДЕ — оценка разброса.
#
# Зачем. Все выводы наших абляций получены на одном сиде: заморозка процессора
# хуже на 7,5 %, косинус даёт 1,6 %, многоуровневый меш нейтрален. Насколько из
# этого шум — неизвестно, разброс мы не мерили ни разу. Этот прогон отличается
# от основной модели ровно одним: random_seed 43 вместо 42.
#
# Если разность между сидами окажется сопоставима с эффектами абляций, часть
# наших выводов придётся переформулировать как «в пределах разброса».
#
# ВАЖНО: до 16.08.2026 сид был зашит в src/main.py и конфиг игнорировался,
# поэтому такой прогон был невозможен в принципе. Исправлено тем же числом.
#
# Запуск: bash scripts/_paper_run_krsk_from_v3lr.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/krsk_seed43_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
EXP=multires_krsk_33f_seed43
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth
mkdir -p "$OUT"; exec >>"$OUT/krsk_seed43_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ОСНОВНАЯ МОДЕЛЬ НА СИДЕ 43 ==="
# Сторож ловит и обучение, и инференс: 22.08.2026 чуть не запустили дожиг
# поверх пятисуточной развёртки на той же карте — прежний сторож знал
# только про src.main и predict.py бы не заметил.
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/krsk_seed43_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33R/data.npy" ]]; then
  # Сборка регионального 33-канального набора. На свежей виртуалке /data пусто,
  # и без этого шага раннер просто останавливался (17.08.2026 потеряли прогон).
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  if [[ ! -f "$GX/coords.npz" && -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]]; then
    cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
    log "coords.npz скопирован в global_extra"
  fi
  log "собираю региональный датасет (часы CPU)"
  python -u scripts/build_multires_russia_33f.py \
      --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir "$GX" \
      --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/krsk_seed43_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }
[[ -f "$PRETRAINED" ]] || { log "НЕТ ВЕСОВ $PRETRAINED — их надо запушить с машины, где считался контроль"; exit 1; }

RESUME=""; [[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "найден чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD)"
log "START обучения $EXP от $PRETRAINED $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/krsk_seed43_train.log" 2>&1
log "DONE обучение rc=$?"

# Оценивать последнюю эпоху: отбор по одношаговой ошибке систематически
# промахивается мимо дальних горизонтов (п. 4.4 статьи).
mkdir -p /data/paper_heavy
python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_seed43_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch"); print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF
CK=""; [[ -f /data/paper_heavy/krsk_seed43_last.pth ]] && CK="--ckpt /data/paper_heavy/krsk_seed43_last.pth"
log "START оценки по области $CK"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 $CK \
    --save-sample-metrics "$OUT/krsk_seed43_roi_samples.npz" >> "$OUT/krsk_seed43_roi.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/krsk_seed43_roi.log" | tail -1)"
grep -E '^\s+(t2m|msl)\b' "$OUT/krsk_seed43_roi.log" | tail -2
log "сравнивать с основной моделью: t2m 1.32/1.53/1.59/1.66 °C, агрегат 73.41 %"
log "=== ALL DONE ==="
