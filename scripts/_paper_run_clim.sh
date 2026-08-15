#!/usr/bin/env bash
# Климатологический эталон для основной модели статьи.
#
# ТОЛЬКО ПРОЦЕССОР — видеокарту не трогает, можно запускать параллельно с
# обучением. Читает датасет, поэтому идёт с пониженным приоритетом, чтобы не
# отбирать диск у даталоадера.
#
# Считает климатологию гармониками (3 годовые + суточный ход) по ОБУЧАЮЩЕЙ части
# выборки и сравнивает с ней модель на тех же сроках, что в
# m33_last_roi_samples.npz — то есть построчно сопоставимо с таблицами статьи.
#
# Запуск: bash scripts/_paper_run_clim.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/clim_master.log
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне (только процессор). лог: /workdir/paper_results/clim_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
mkdir -p "$OUT"; exec >>"$OUT/clim_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== КЛИМАТОЛОГИЧЕСКИЙ ЭТАЛОН ==="

source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

# Сроки берём те же, на которых посчитаны публикуемые числа: файл лежит в
# репозитории, на самой машине его может не быть после перезапуска.
S=$(ls -1 "$OUT"/m33_last_roi_samples.npz docs/paper/runs/*/m33_last_roi_samples.npz 2>/dev/null | head -1)
[[ -n "$S" ]] || { log "не нашёл m33_last_roi_samples.npz — стоп"; exit 1; }
log "сроки из: $S"

nice -n 19 python -u scripts/paper_climatology.py \
    --data-dir "$D33R" --samples "$S" --region 50 60 83 98 \
    --out "$OUT/clim_krsk.npz" >> "$OUT/clim.log" 2>&1
log "DONE rc=$?"
tail -20 "$OUT/clim.log"
log "=== ALL DONE ==="
