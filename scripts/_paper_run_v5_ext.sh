#!/usr/bin/env bash
# Продолжение v5: ещё 15 эпох на трёх шагах развёртки, с косинусным спадом темпа.
#
# Почему это, а не что-то другое. На пяти сутках суточный шаг обходит
# шестичасовой на 12,5 п.п. по успешности, и разрыв растёт с горизонтом. Это
# самый крупный результат, что у нас есть, и его надо доводить.
#
# Два изменения, у каждого своя улика:
#   1) ещё 15 эпох — последняя эпоха v5 обошла «лучшую» на дальних сроках
#      (38,5 против 37,3 % на +120 ч), значит поздние эпохи с длинной развёрткой
#      реально работают;
#   2) косинусный спад — у v5 одношаговая ошибка деградировала с 0.08065 до
#      0.08450 за последние одиннадцать эпох, а косинус эту деградацию снимает
#      (региональная абляция: 1,6 % по последней эпохе).
#
# Это НЕ абляция: два изменения сразу, вклады не разделить. Цель — модель, а не
# измерение. Разделять будем, если понадобится для статьи.
#
# Веса берутся из wb2_512x256_33f_ar_v5_24h, а обучение идёт в отдельном
# каталоге — чтобы уже посчитанные числа v5 остались воспроизводимыми.
#
# Запуск: bash scripts/_paper_run_v5_ext.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/v5ext_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/v5ext_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33G=/data/datasets/wb2_512x256_33f_v3
SRC=wb2_512x256_33f_ar_v5_24h
EXP=wb2_512x256_33f_ar_v5_24h_ext

mkdir -p "$OUT"
MASTER="$OUT/v5ext_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== V5 EXT: +15 эпох с косинусным спадом ==="

if pgrep -f "src.main" >/dev/null; then
  log "на карте уже идёт обучение — стоп"
  exit 1
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
[[ -f "$D33G/data.npy" ]] || { log "нет $D33G — стоп"; exit 1; }

# ── Перенос состояния из исходного каталога ───────────────────────────
# Копируем, а не продолжаем на месте: иначе чекпойнт 30-й эпохи, на котором
# посчитаны опубликованные числа пяти суток, будет затёрт.
if [[ ! -f "experiments/$EXP/checkpoint.pth" ]]; then
  if [[ ! -f "experiments/$SRC/checkpoint.pth" ]]; then
    log "нет исходного чекпойнта experiments/$SRC/checkpoint.pth — стоп"
    exit 1
  fi
  cp -p "experiments/$SRC/checkpoint.pth" "experiments/$EXP/checkpoint.pth"
  [[ -f "experiments/$SRC/best_model.pth" ]] && \
    cp -p "experiments/$SRC/best_model.pth" "experiments/$EXP/best_model.pth"
  log "состояние перенесено из $SRC"
fi

python - <<PY
import torch, pathlib
ck = torch.load(pathlib.Path("experiments/$EXP/checkpoint.pth"), map_location="cpu")
e = ck.get("epoch")
print("[prep] продолжаем с эпохи", (e + 1) if isinstance(e, int) else e,
      "| AR =", ck.get("ar_steps"), "| лучшее", round(ck.get("best_val_loss", 0), 5))
PY

log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения"
log "START обучения $EXP (45 эпох, косинус, возобновление)"
python -u -m src.main "experiments/$EXP" --resume >> "$OUT/v5ext_train.log" 2>&1
log "DONE обучение rc=$?"

# ── Оценка на пяти сутках, тем же окном ───────────────────────────────
log "START оценки (5 шагов = +24…+120 ч)"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33G" \
    --split test_only --ar-steps 5 --max-samples 200 --per-channel --no-save \
    --save-sample-metrics "$OUT/v5ext_samples.npz" \
    >> "$OUT/v5ext.log" 2>&1
log "DONE оценка rc=$?"
grep -E '^\s+\+(24|48|72|96|120)h:' "$OUT/v5ext.log" | tail -5

log "--- С ЧЕМ СРАВНИВАТЬ (успешность, 200 сроков) ---"
log "  +72 ч:  v3 46.4 %   v5 51.8 %"
log "  +96 ч:  v3 35.9 %   v5 45.2 %"
log "  +120 ч: v3 26.0 %   v5 38.5 %"
log "t2m на +120 ч: v3 2.97 °C, v5 2.69 °C"
log "=== ALL DONE ==="
