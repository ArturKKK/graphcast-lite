#!/usr/bin/env bash
# Сравнение глобальных v3 и v4 в физических единицах на одном тестовом окне.
#
# Зачем. По значению функционала v4 (многоуровневый меш) отстаёт от v3 на 21 %,
# и разрыв не сокращается за две стадии развёртки. Но нормированный MSE — плохой
# судья: 20 % по нему могут оказаться и заметной разницей, и шумом в третьем
# знаке t2m. Решаем по градусам и гектопаскалям.
#
# Что делает: дожидается конца стадии AR=3 (24-я эпоха), останавливает обучение,
# достаёт последний чекпойнт v4 и прогоняет обе модели с одинаковыми настройками.
#
# ⚠️ Оговорка о честности. У v4 берётся эпоха 24, у v3 — её опубликованные веса
# (лучшая эпоха 30). Бюджет неравный, и он в пользу v3. Но v3 вышла на плато уже
# к 18-й эпохе и за следующие двенадцать улучшилась с 0.01868 до 0.01865, то есть
# ни на сколько, — так что перекос невелик. Это консервативная проверка для v4:
# если она выигрывает даже так, вывод надёжен.
#
# Запуск: bash scripts/_paper_run_v3v4_compare.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/v3v4_master.log
set -uo pipefail

# Уходим в фон сами: длинную строку с nohup/setsid терминал рвёт при вставке.
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/v3v4_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33G=/data/datasets/wb2_512x256_33f_v3
EXP3=wb2_512x256_33f_ar_v3
EXP4=wb2_512x256_33f_ar_v4_multimesh
STOP_EPOCH=${STOP_EPOCH:-24}
MAXN=${MAXN:-200}          # сроков; 200 — как в майском сравнении v2 vs v3
ARSTEPS=${ARSTEPS:-4}      # +6…+24 ч

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/v3v4_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

LOG4="experiments/$EXP4/training_log.txt"
last_epoch() { grep -E '^\s+[0-9]+\s+[0-9]+\s' "$LOG4" 2>/dev/null | tail -1 | awk '{print $1}'; }

log "=== V3 vs V4 COMPARE START (ждём эпоху $STOP_EPOCH) ==="

# ── 1. Ждём конца стадии AR=3 ─────────────────────────────────────────
for i in $(seq 1 300); do   # до 25 часов
  e=$(last_epoch)
  [[ -n "$e" && "$e" -ge "$STOP_EPOCH" ]] && { log "достигнута эпоха $e"; break; }
  [[ $((i % 12)) -eq 0 ]] && log "эпоха $e, ждём…"
  pgrep -f "src.main" >/dev/null || { log "обучение уже не идёт (эпоха $e) — продолжаю"; break; }
  sleep 300
done

# ── 2. Освободить карту ───────────────────────────────────────────────
if pgrep -f "src.main" >/dev/null; then
  log "останавливаю обучение v4 на эпохе $(last_epoch)"
  pkill -f "src.main"; sleep 30
fi
pgrep -f "src.main" >/dev/null && { log "процесс не умер — стоп"; exit 1; }
log "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1)"

source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
[[ -f "$D33G/data.npy" ]] || { log "нет глобального датасета $D33G — стоп"; exit 1; }

# ── 3. Последний чекпойнт v4 отдельным файлом ─────────────────────────
python - <<PY
import torch, pathlib
s = pathlib.Path("experiments/$EXP4/checkpoint.pth")
if s.exists():
    ck = torch.load(s, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), "$HEAVY/v4_last.pth")
    # В train.py в чекпойнт кладётся счётчик цикла (с нуля), а в лог пишется
    # epoch+1. Печатаем в той же нумерации, что и лог, иначе расходится.
    e = ck.get("epoch")
    print("[prep] v4 чекпойнт эпохи", (e + 1) if isinstance(e, int) else e,
          "(в нумерации training_log.txt)")
else:
    print("[prep] чекпойнта v4 нет!")
PY
CKPT_EPOCH=$(python - <<PY
import torch, pathlib
s = pathlib.Path("experiments/$EXP4/checkpoint.pth")
e = torch.load(s, map_location="cpu").get("epoch") if s.exists() else None
print((e + 1) if isinstance(e, int) else "?")
PY
)

# ── 4. Прогон обеих моделей одинаковыми настройками ───────────────────
run() {                     # run <тег> <каталог эксперимента> [доп. аргументы]
  local tag=$1 exp=$2; shift 2
  log "START $tag"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D33G" \
      --split test_only --ar-steps "$ARSTEPS" --max-samples "$MAXN" \
      --per-channel --no-save --save-sample-metrics "$OUT/${tag}_samples.npz" \
      "$@" >> "$OUT/${tag}.log" 2>&1
  local rc=$?
  log "DONE  $tag rc=$rc | $(grep -oE 'skill=[-0-9.]+%' "$OUT/${tag}.log" | tail -1)"
  grep -E '^\s+(t2m|z500|msl|u10|v10)\b' "$OUT/${tag}.log" | tail -5
}

if [[ -f "$HEAVY/v4_last.pth" ]]; then
  run v4_global "$EXP4" --ckpt "$HEAVY/v4_last.pth"
else
  log "пропускаю v4: нет весов"
fi

if [[ -f "experiments/$EXP3/best_model.pth" ]]; then
  run v3_global "$EXP3"
else
  log "ПРОПУСКАЮ v3: нет experiments/$EXP3/best_model.pth"
  log "  веса не в репозитории (в .gitignore) — восстановить с другой машины или из S3"
fi

# ── 5. Сводка ─────────────────────────────────────────────────────────
log "--- ИТОГ (одно окно, $MAXN сроков, +6…+24 ч, глобальная сетка) ---"
for t in v3_global v4_global; do
  [[ -f "$OUT/$t.log" ]] || continue
  log "$t: $(grep -oE 'skill=[-0-9.]+%' "$OUT/$t.log" | tail -1)  $(grep -E '^\s+t2m' "$OUT/$t.log" | tail -1 | tr -s ' ')"
done
log "напоминание: у v4 эпоха ${CKPT_EPOCH:-?} (реальная, из чекпойнта), у v3 — лучшая эпоха 30, бюджет в пользу v3"
log "=== ALL DONE ==="
