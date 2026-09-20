#!/usr/bin/env bash
# Восстановление датасета multires_krsk_33f после стирания /data. БЕЗ обучения:
# только распаковка, сборка merge и сборка 33f — всё, что нужно для инференса.
#
# Запуск:  bash scripts/_vm_restore33f.sh          (уходит в фон сам)
# Лог:     /workdir/paper_logs/restore33f.log
#
# Место. Платформа убивает job примерно на 240 ГБ в /data, а цепочка требует
# пика около 215 ГБ поверх уже занятого. Поэтому скрипт СНАЧАЛА считает баланс
# и, если не сходится, останавливается и печатает, что предлагает освободить.
# Удалять сам он ничего не станет: для этого надо повторить запуск с FREE=1.
#
# Освобождаются всероссийские наборы (region_russia_*, около 100 ГБ): для
# красноярской линии они не нужны, а корпуса постобработки из них уже собраны
# в parquet. По замыслу платформы они возвращаются из S3.
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_logs
  DAEMONIZED=1 FREE="${FREE:-0}" setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_logs/restore33f.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
LOG=/workdir/paper_logs/restore33f.log
BASE=$DATA/wb2_512x256_19f_ar
MERGE=$DATA/multires_krsk_19f_merge
GEXTRA=$DATA/global_512x256_extra_2010-2021_07deg
REXTRA=$DATA/region_krsk_61x41_extra_2010-2020_025deg
OUT33=$DATA/multires_krsk_33f
LIMIT_GB=240
PEAK_GB=215   # пик цепочки: global 82 + merge 81 + krsk 2 + запас

mkdir -p /workdir/paper_logs
exec >>"$LOG" 2>&1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
cd "$REPO" || exit 1

log "=== ВОССТАНОВЛЕНИЕ 33f ==="
[[ -f "$OUT33/data_extra.npy" ]] && { log "33f уже собран: $(du -sh "$OUT33" | cut -f1) — делать нечего"; exit 0; }

# ---------- баланс места ----------
used=$(du -sm "$DATA" 2>/dev/null | cut -f1); used=$((used / 1024))
log "в $DATA занято ${used} ГБ, лимит платформы ~${LIMIT_GB} ГБ, пик цепочки ~${PEAK_GB} ГБ"
russia=$(du -sm "$DATA"/region_russia_* 2>/dev/null | awk '{s+=$1} END{print int(s/1024)}')
if (( used + PEAK_GB > LIMIT_GB )); then
  log "не помещается: нужно освободить не менее $((used + PEAK_GB - LIMIT_GB)) ГБ"
  log "кандидаты — всероссийские наборы (${russia:-0} ГБ):"
  du -sh "$DATA"/region_russia_* 2>/dev/null | sed 's/^/    /'
  if [[ "${FREE:-0}" != "1" ]]; then
    log "САМ НЕ УДАЛЯЮ. если согласен — повтори запуск так:"
    log "    FREE=1 bash scripts/_vm_restore33f.sh"
    exit 1
  fi
  log "FREE=1 — удаляю всероссийские наборы"
  rm -rf "$DATA"/region_russia_*
  used=$(du -sm "$DATA" 2>/dev/null | cut -f1); used=$((used / 1024))
  log "стало ${used} ГБ"
fi

# ---------- venv ----------
if [[ ! -x "$VENV/bin/python" ]]; then
  log "нет venv — поднимаю (VENV_ONLY)"
  VENV_ONLY=1 bash scripts/_paper_setup_vm.sh >> "$LOG" 2>&1 \
    || { log "FATAL: venv не поднялся"; exit 2; }
fi
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
log "python: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | head -1)"

# ---------- распаковка и merge ----------
# SLIM_AFTER_MERGE=1: после сборки merge глобальный 19f (82 ГБ) удаляется —
# для 33f он больше не нужен, а без этого не хватит места на сам 33f.
if [[ ! -f "$MERGE/data.npy" ]]; then
  log "START распаковка + сборка merge (это самая долгая часть)"
  SLIM_AFTER_MERGE=1 bash scripts/_paper_setup_vm.sh >> "$LOG" 2>&1
  rc=$?
  [[ -f "$MERGE/data.npy" ]] || { log "FATAL: merge не собрался (rc=$rc), см. /workdir/paper_logs/paper_setup.log"; exit 3; }
  log "DONE merge: $(du -sh "$MERGE" | cut -f1)"
else
  log "merge уже на месте: $(du -sh "$MERGE" | cut -f1)"
fi

# ---------- сборка 33f ----------
for p in "$GEXTRA" "$REXTRA/data_extra.npy"; do
  [[ -e "$p" ]] || { log "FATAL: нет $p — без него 33f не собрать"; exit 4; }
done
log "START сборка 33f → $OUT33 (выделяется около 60 ГБ)"
python -u scripts/build_multires_russia_33f.py \
    --multires-dir "$MERGE" --extra-dir "$GEXTRA" \
    --region-extra-dir "$REXTRA" --out-dir "$OUT33" >> "$LOG" 2>&1
rc=$?
[[ -f "$OUT33/data_extra.npy" ]] || { log "FATAL: сборка 33f не удалась (rc=$rc)"; exit 5; }

log "DONE 33f: $(du -sh "$OUT33" | cut -f1) | в $DATA занято $(du -sh "$DATA" | cut -f1)"
log "=== ГОТОВО, можно запускать инференс ==="
