#!/usr/bin/env bash
# ACC против климатологии WeatherBench 2 — чтобы наши числа и числа GraphCast
# стали сопоставимы.
#
# До сих пор наш ACC считался против собственной климатологии по девяти годам,
# а у GraphCast — против климатологии WB2 за 1990–2019. Ставить 0,975 и 0,988
# рядом было нельзя. Здесь основная модель пересчитывается против той же
# таблицы, против которой меряют GraphCast и HRES.
#
# Таблица лежит в репозитории (docs/paper/runs/clim_wb2_nodes.npz, 63 МБ):
# шесть каналов, которые статья отчитывает поканально, на 2501 узле области.
# Нормировку она получает от scalers.npz самого набора.
#
# Запуск:  bash scripts/_vm_batch_accwb2.sh     (уходит в фон)
# Лог:     /workdir/paper_results/accwb2_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/accwb2_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
ROI="50 60 83 98"
INNER="55.5 56.5 92 94"
CLIM=$REPO/docs/paper/runs/clim_wb2_nodes.npz

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/accwb2_master.log"
LOCK=$OUT/.accwb2.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%d.%m %H:%M:%S')] батч уже идёт" >> "$MASTER"; exit 0; }
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== ACC против климатологии WB2 ($(git rev-parse --short HEAD)) ==="
[[ -f "$CLIM" ]] || { log "FATAL: нет $CLIM — сделайте git pull"; exit 1; }
[[ -f "$D33/data.npy" ]] || { log "FATAL: нет $D33"; exit 1; }
CK=$HEAVY/krsk33f_chw_last.pth
[[ -f "$CK" ]] || { log "FATAL: нет состояния $CK"; exit 1; }

run() {
  local tag="$1" reg="$2"
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag"; return 0; }
  log "START $tag"
  python -u scripts/predict.py experiments/multires_krsk_33f_chw --data-dir "$D33" \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $reg --lat-weight --climatology "$CLIM" --ckpt "$CK" \
      --save-sample-metrics "$npz" > "$lf" 2>&1
  local rc=$? acc t2
  acc=$(grep -oE 'ACC=[0-9.]+' "$lf" | tail -1)
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $acc | $t2"
}

run g_chw_roi   "$ROI"
run g_chw_inner "$INNER"

log "=== ВСЁ ==="
grep -E "DONE  g_" "$MASTER" | tail -3
