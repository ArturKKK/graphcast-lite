#!/usr/bin/env bash
# Сброс темпа на региональной модели — тем же рецептом, что сработал на глобальной.
#
# Почему заново. Прогон 16.08 (multires_krsk_33f_lrdrop) шёл через --resume и
# унаследовал из чекпойнта счётчик терпения 13 при пороге 12: ранняя остановка
# сработала после первой же эпохи, и приём остался непроверенным. Здесь берём
# рецепт глобального прогона буквально: короткий дожиг ВОСЬМИ эпох на полной
# развёртке, процессор разморожен, темп 7.5e-5 с косинусом до нуля, оптимизатор
# с нуля, никакого возобновления.
#
# Опорные числа основной модели по области: t2m 1.32/1.53/1.59/1.66 °C,
# агрегат 73.41 % (лог m33_last_roi). Разброс по сиду ~0.3 %, так что осмысленным
# считаем сдвиг агрегата примерно от 1 п.п.
#
# Запуск: bash scripts/_paper_run_drop8_krsk.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/drop8_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f; EXP=multires_krsk_33f_drop8
mkdir -p "$OUT"; exec >>"$OUT/drop8_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== СБРОС ТЕМПА НА РЕГИОНАЛЬНОЙ, РЕЦЕПТ ГЛОБАЛЬНОЙ ==="
pgrep -f "src.main" >/dev/null && { log "карта занята — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/drop8_krsk_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33R/data.npy" ]]; then
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  [[ ! -f "$GX/coords.npz" && -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]] && \
    cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
  log "собираю региональный датасет (часы CPU)"
  python -u scripts/build_multires_russia_33f.py --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir "$GX" --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/drop8_krsk_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

# Стартовые веса — ПОСЛЕДНЯЯ эпоха основной модели: именно на ней стоят числа
# статьи. best_model.pth основной модели остался на 15-й эпохе и берёт не то.
mkdir -p /data/paper_heavy
PRETRAINED=/data/paper_heavy/krsk33f_last_epoch.pth
python -u - "experiments/$SRC/checkpoint.pth" "$PRETRAINED" <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print("[prep] нет чекпойнта основной модели"); sys.exit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
e = ck.get("epoch")
print("[prep] стартуем с эпохи", (e + 1) if isinstance(e, int) else e)
PYEOF
[[ -f "$PRETRAINED" ]] || { log "нет стартовых весов — стоп"; exit 1; }

# Возобновление только как починка после падения — и тогда терпение обнуляем,
# иначе прогон умрёт на первой же эпохе (наступили 16.08.2026).
RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume --reset-patience"; log "нашёлся чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD 2>/dev/null)"
log "START обучения $EXP (8 эпох, AR=4, темп 7.5e-5 с косинусом до нуля) $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/drop8_krsk_train.log" 2>&1
log "DONE обучение rc=$?"

# Оцениваем и последнюю эпоху, и лучшую по валидации на четырёх горизонтах:
# у этого прогона best_model.pth пишется им самим, так что выбор осмысленный.
python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_drop8_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch")
    print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF

for TAG in last best; do
  CK=""
  [[ "$TAG" == last && -f /data/paper_heavy/krsk_drop8_last.pth ]] && CK="--ckpt /data/paper_heavy/krsk_drop8_last.pth"
  [[ "$TAG" == best && -f "experiments/$EXP/best_model.pth" ]] && CK="--ckpt experiments/$EXP/best_model.pth"
  log "START оценки по области ($TAG) $CK"
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
      --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 $CK \
      --save-sample-metrics "$OUT/drop8_krsk_roi_${TAG}_samples.npz" >> "$OUT/drop8_krsk_roi_$TAG.log" 2>&1
  RC=$?
  AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/drop8_krsk_roi_$TAG.log" | grep -oE 'skill=[-0-9.]+%')
  log "DONE оценка $TAG rc=$RC | агрегат по области $AGG"
  grep -E '^\s+(t2m|msl)\b' "$OUT/drop8_krsk_roi_$TAG.log" | tail -2
done
log "сравнивать с основной моделью: t2m 1.32/1.53/1.59/1.66 °C, агрегат 73.41 %"
log "=== ALL DONE ==="
