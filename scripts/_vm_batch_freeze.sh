#!/usr/bin/env bash
# Доверительные интервалы для табл. 4: заморозка процессора против её отсутствия.
#
# Единственное место статьи, где разность заявлена без интервала: «по агрегату
# различие составляет 1,3 п.п. (60,8 против 59,5 %) в пользу заморозки». Старые
# прогоны шли без --save-sample-metrics, поэтому из сохранённого это не достать.
#
# Три состояния, ровно как строки табл. 4:
#   с заморозкой,  лучшая по val (эпоха 16)  — best_model.pth
#   без заморозки, лучшая по val (эпоха 7)   — best_model.pth
#   без заморозки, равный бюджет (эпоха 16)  — последняя из checkpoint.pth
#
# 19-канальная линия: прямое предсказание поля, отсюда --no-residual.
#
# Запуск:  bash scripts/_vm_batch_freeze.sh      (уходит в фон)
# Лог:     /workdir/paper_results/freeze_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/freeze_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D19=/data/datasets/multires_krsk_19f_merge
ROI="50 60 83 98"

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/freeze_master.log"
LOCK=$OUT/.freeze_batch.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%d.%m %H:%M:%S')] батч уже идёт — выхожу" >> "$MASTER"; exit 0
fi
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== ТАБЛ. 4: ЗАМОРОЗКА, ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ($(git rev-parse --short HEAD)) ==="
[[ -f "$D19/data.npy" ]] || { log "FATAL: нет $D19 — он собирается scripts/_vm_restore33f.sh"; exit 1; }

# Состояние последней эпохи для конфигурации без заморозки: равный бюджет.
CK16=$HEAVY/nofreeze_last.pth
if [[ ! -f "$CK16" ]]; then
  python - "$CK16" <<'PY'
import sys, pathlib, torch
src = pathlib.Path("/workdir/graphcast-lite/experiments/multires_nores_nofreeze/checkpoint.pth")
if not src.exists():
    print(f"[prep] нет {src}"); raise SystemExit(0)
ck = torch.load(src, map_location="cpu")
sd = ck.get("model_state_dict", ck)
torch.save(sd, sys.argv[1])
print(f"[prep] nofreeze: epoch={ck.get('epoch','?')} ar={ck.get('ar_steps','?')} -> {sys.argv[1]}")
PY
fi

run() {
  local tag="$1" exp="$2"; shift 2
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag"; return 0; }
  log "START $tag"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D19" \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $ROI --no-residual --save-sample-metrics "$npz" "$@" > "$lf" 2>&1
  local rc=$? sk t2
  sk=$(grep -oE 'skill=[0-9.]+%' "$lf" | tail -1)
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $sk | $t2"
}

run f_freeze_best   multires_nores_freeze6
run f_nofreeze_best multires_nores_nofreeze
[[ -f "$CK16" ]] && run f_nofreeze_ep16 multires_nores_nofreeze --ckpt "$CK16" \
  || log "состояния последней эпохи нет — строки равного бюджета не будет"

log "=== ВСЁ ==="
grep -E "DONE  f_" "$MASTER" | tail -4
