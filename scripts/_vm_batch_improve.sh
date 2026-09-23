#!/usr/bin/env bash
# Улучшение основной модели: два эксперимента на двух картах (23.09.2026).
#
#   long — тот же дожиг, что у основной модели chw, но 16 эпох вместо 8.
#          По логу chw ошибка на валидации падала каждую эпоху и встала лишь
#          потому, что косинус довёл темп до нуля: модель недоучена.
#          ~44 ч обучения + ~1,5 ч оценки.
#
#   ema  — сначала диагностика основной модели (~40 мин): поля ошибки по
#          области на сроках 00/12 UTC, чтобы разложить отставание от GraphCast
#          на крупный и мелкий масштаб. Затем дожиг 8 эпох как у chw, но с
#          усреднением весов (EMA 0,999) и обрезкой градиента (1,0). Число
#          шагов то же, что у chw, — сравнение при равном бюджете.
#          ~40 мин + ~22 ч + ~1,5 ч.
#
# Оба стартуют с того же состояния, что и chw, и оцениваются так же, как
# основная модель в статье: широтные веса, ACC против климатологии WB2.
#
# Запуск:  bash scripts/_vm_batch_improve.sh long     (на одной машине)
#          bash scripts/_vm_batch_improve.sh ema      (на другой)
# Лог:     /workdir/paper_results/improve_<вариант>_master.log
set -uo pipefail
V=${1:-}
[[ "$V" == "long" || "$V" == "ema" ]] || { echo "вариант: long или ema"; exit 1; }

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/improve_${V}_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
ROI="50 60 83 98"
CLIM=$REPO/docs/paper/runs/clim_wb2_nodes.npz
SRC=multires_krsk_33f_chw
EXP=multires_krsk_33f_chw_${V}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/improve_${V}_master.log"
exec 9>"$OUT/.improve_${V}.lock"
flock -n 9 || { echo "[$(date '+%d.%m %H:%M:%S')] уже идёт" >> "$MASTER"; exit 0; }
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== УЛУЧШЕНИЕ: $V ($(git rev-parse --short HEAD)) ==="

BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# ---------- окружение и датасет ----------
if [[ ! -x "$VENV/bin/python" || ! -f "$D33/data_extra.npy" ]]; then
  log "нет окружения или датасета — восстанавливаю (~40 мин), лог /workdir/paper_logs/restore33f.log"
  DAEMONIZED=1 FREE=1 bash scripts/_vm_restore33f.sh
  log "восстановление rc=$?"
fi
[[ -f "$D33/data_extra.npy" ]] || { log "датасет так и не собрался — стоп"; exit 1; }
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ---------- состояния моделей ----------
extract() {   # checkpoint.pth -> голый state_dict
  python - "$1" "$2" <<'PY'
import sys, pathlib, torch
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print(f"[prep] нет {src}"); raise SystemExit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
print(f"[prep] {src.parent.name}: эпоха {ck.get('epoch','?')} -> {dst}")
PY
}
START=$HEAVY/krsk33f_last_epoch.pth
[[ -f "$START" ]] || cp -p "$OUT/krsk33f_last_epoch.pth" "$START" 2>/dev/null \
  || extract "experiments/multires_krsk_33f/checkpoint.pth" "$START" \
  || { log "нет стартового состояния multires_krsk_33f — стоп"; exit 1; }
CHW=$HEAVY/krsk33f_chw_last.pth
[[ -f "$CHW" ]] || extract "experiments/$SRC/checkpoint.pth" "$CHW" || true

# ---------- оценка ----------
run() {   # run <тег> <опыт> <чекпойнт> [доп. ключи]
  local tag="$1" exp="$2" ck="$3"; shift 3
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag (уже посчитан)"; return 0; }
  log "START $tag"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D33" \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $ROI --lat-weight --ckpt "$ck" \
      --save-sample-metrics "$npz" --save-region-errors "$OUT/${tag}_errors.npz" "$@" \
      > "$lf" 2>&1
  local rc=$? t2
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $t2"
}

# Диагностика основной модели: без климатологии, так вдвое быстрее — нужны
# только поля ошибки.
if [[ "$V" == "ema" ]]; then
  [[ -f "$CHW" ]] && run d_chw_diag "$SRC" "$CHW" \
    || log "нет состояния chw — диагностику пропускаю"
fi

# ---------- конфиг ----------
mkdir -p "experiments/$EXP"
python - "experiments/$SRC/config.json" "experiments/$EXP/config.json" "$V" <<'PY'
import json, sys
src, dst, v = sys.argv[1:4]
c = json.load(open(src))
if v == "long":
    c["num_epochs"] = 16
else:
    c["ema_decay"] = 0.999
    c["grad_clip_norm"] = 1.0
c["early_stopping_patience"] = 100      # косинус до нуля: останавливаться рано незачем
c["_comment"] = f"chw_{v}: см. scripts/_vm_batch_improve.sh"
json.dump(c, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"[prep] {dst}: эпох {c['num_epochs']}, EMA {c.get('ema_decay', 0)}, "
      f"обрезка {c.get('grad_clip_norm', 0)}, темп {c['learning_rate']} {c.get('lr_schedule')}")
PY
python - "experiments/$EXP/config.json" <<'PY' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
ExperimentConfig(**json.load(open(sys.argv[1])))
PY

# ---------- обучение ----------
RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "нашёлся чекпойнт — продолжаю"; }
log "START обучение $EXP $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$START" $RESUME \
    >> "$OUT/improve_${V}_train.log" 2>&1
log "DONE  обучение rc=$?"
tail -12 "experiments/$EXP/training_log.txt" 2>/dev/null

# ---------- оценка результата ----------
FIN=$HEAVY/${EXP}_last.pth
if [[ "$V" == "ema" && -f "experiments/$EXP/ema_last.pth" ]]; then
  cp -p "experiments/$EXP/ema_last.pth" "$FIN"
else
  extract "experiments/$EXP/checkpoint.pth" "$FIN" || { log "нет чекпойнта — оценки не будет"; exit 1; }
fi
run t_${V}_roi "$EXP" "$FIN" --climatology "$CLIM"

log "=== ВСЁ ==="
grep -E "DONE  " "$MASTER" | tail -4
