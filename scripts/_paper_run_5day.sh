#!/usr/bin/env bash
# Сравнение шага 6 ч и 24 ч на горизонте пяти суток (+120 ч).
#
# Зачем. На +72 ч точка перелома уже пройдена: суточный шаг выигрывает 5,5 п.п.
# по успешности, и разрыв растёт (−2,6 → +1,8 → +5,5 на +24/+48/+72 ч).
# Разделы 5.7–5.8 статьи живут на 7 и 14 сутках, где у шестичасовой модели
# 28 и 56 шагов. Пять суток — промежуточная точка, которая покажет, сохраняется
# ли тренд далеко за пределами обучающей развёртки.
#
# 120 ч = 20 шагов у v3 и 5 шагов у v5.
#
# Третий прогон — v5 на ПОСЛЕДНЕЙ эпохе вместо «лучшей». Лучшая у неё 19-я, ещё
# на стадии двух шагов развёртки; последняя видела три и может оказаться лучше
# именно на дальних сроках. У основной модели статьи так и вышло — на этом
# построен п. 4.4 о ненадёжности отбора по одношаговой ошибке.
#
# Обучать ничего не надо, это только инференс: около часа на все три.
#
# Запуск: bash scripts/_paper_run_5day.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/d5_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/d5_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33G=/data/datasets/wb2_512x256_33f_v3
EXP3=wb2_512x256_33f_ar_v3
EXP5=wb2_512x256_33f_ar_v5_24h
MAXN=${MAXN:-200}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/d5_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== 5 СУТОК: шаг 6 ч против 24 ч ==="

if pgrep -f "src.main" >/dev/null; then
  log "на карте идёт обучение — стоп, инференс подождёт"
  exit 1
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
[[ -f "$D33G/data.npy" ]] || { log "нет $D33G — стоп"; exit 1; }

# Последняя эпоха v5 отдельным файлом
python - <<PY
import torch, pathlib
s = pathlib.Path("experiments/$EXP5/checkpoint.pth")
if s.exists():
    ck = torch.load(s, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), "$HEAVY/v5_last.pth")
    e = ck.get("epoch")
    print("[prep] v5 последняя эпоха", (e + 1) if isinstance(e, int) else e)
PY

run() {   # run <тег> <эксперимент> <шагов> [доп. аргументы]
  local tag=$1 exp=$2 steps=$3; shift 3
  log "START $tag ($steps шагов)"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D33G" \
      --split test_only --ar-steps "$steps" --max-samples "$MAXN" \
      --per-channel --no-save --save-sample-metrics "$OUT/${tag}_samples.npz" \
      "$@" >> "$OUT/${tag}.log" 2>&1
  log "DONE $tag rc=$?"
  grep -E '^\s+\+(24|48|72|96|120)h:' "$OUT/${tag}.log" | tail -5
}

run d5_v3      "$EXP3" 20
run d5_v5_best "$EXP5" 5
[[ -f "$HEAVY/v5_last.pth" ]] && run d5_v5_last "$EXP5" 5 --ckpt "$HEAVY/v5_last.pth"

log "--- ИТОГ: приземная температура по горизонтам ---"
for t in d5_v3 d5_v5_best d5_v5_last; do
  [[ -f "$OUT/$t.log" ]] || continue
  log "$t: $(grep -E '^\s+t2m' "$OUT/$t.log" | tail -1 | tr -s ' ')"
done
log "у v3 шаг 6 ч (20 шагов до +120 ч), у v5 шаг 24 ч (5 шагов)"
log "опорные числа на +72 ч: v3 2.11 °C, v5 2.06 °C"
log "=== ALL DONE ==="
