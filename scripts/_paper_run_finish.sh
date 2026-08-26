#!/usr/bin/env bash
# Всё, что осталось досчитать под статью. Один запуск, оба пункта подряд.
#
# 1. Длинные развёртки серии усвоения на chw — три прогона, которые не успели
#    26.08: long_oi10, long_oi10_first4, long_oi1. Именно из first4 берётся
#    кривая затухания: на сколько усвоение улучшает прогноз через 6, 12, 18 и
#    24 часа после последнего усвоенного наблюдения. Уже посчитанные пункты
#    серии пропускаются, пересчитывать девять часов не придётся.
#
# 2. Пятисуточная развёртка для chw и chwb — основной модели статьи на длинных
#    сроках мы не мерили вовсе, а сравнивать ar12 было не с чем.
#
# Оба нужны именно здесь: веса chw и chwb лежат на этой машине.
#
# Примерно 4 часа на первое и 2,2 на второе, итого около 6,5 ч.
#
# Запуск: bash scripts/_paper_run_finish.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/finish_master.log"; exit 0
fi
OUT=/workdir/paper_results; mkdir -p "$OUT"
exec >>"$OUT/finish_master.log" 2>&1
cd "$(dirname "$0")/.." || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== ДОСЧЁТ ПОД СТАТЬЮ ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

log "--- 1/2: длинные развёртки серии усвоения на chw ---"
DAEMONIZED=1 bash scripts/_paper_run_roiw30_assim.sh multires_krsk_33f_chw bg
log "серия усвоения rc=$?"

log "--- 2/2: пятисуточная развёртка для chw и chwb ---"
DAEMONIZED=1 bash scripts/_paper_run_krsk_5day.sh \
    "multires_krsk_33f_chw:chw multires_krsk_33f_chwb:chwb"
log "пятисуточная rc=$?"

log "=== ВСЁ ДОСЧИТАНО ==="
