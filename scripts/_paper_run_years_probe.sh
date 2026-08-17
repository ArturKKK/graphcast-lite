#!/usr/bin/env bash
# Замер: с какой скоростью виртуалка тянет ERA5 из облака.
#
# Зачем. Расширение выборки — единственный крупный рычаг, к которому мы не
# притрагивались: сейчас 12 лет (2010–2021), а исходник покрывает 1959–2022.
# Внешний ориентир — MR-GNF: 1,6 млн параметров против наших 5,9 и не хуже
# тяжёлых систем, но обучение на сорока годах.
#
# Прежде чем тратить дни, качаем ДВА года и засекаем время. По ним считаем,
# сколько уйдёт на десять или двадцать, и решаем, браться ли вообще.
#
# Объём по нашим же данным: 6,8 ГБ на год для 19 каналов плюс 3,6 ГБ на
# дополнительные уровни — итого около 10,4 ГБ на год.
#
# Запуск: bash scripts/_paper_run_years_probe.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/years_probe.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
DST=/data/datasets/probe_19f_2008_2009
mkdir -p "$OUT"; exec >>"$OUT/years_probe.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== ЗАМЕР СКОРОСТИ ЗАГРУЗКИ ERA5 ==="
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
df -h /data | tail -1 | awk '{print "  свободно на /data: "$4}'

START=$(date +%s)
python -u scripts/build_dataset_512x256.py --out-dir "$DST" \
    --start-year 2008 --end-year 2009 >> "$OUT/years_probe_build.log" 2>&1
RC=$?
END=$(date +%s)
SEC=$((END - START))
SIZE=$(du -sb "$DST" 2>/dev/null | cut -f1)
log "готово rc=$RC за $((SEC / 60)) мин, объём $((SIZE / 1024 / 1024 / 1024)) ГБ"
python3 - "$SEC" "$SIZE" <<'PY'
import sys
sec, size = int(sys.argv[1]), int(sys.argv[2])
if sec > 0 and size > 0:
    gb_h = size / 1024**3 / (sec / 3600)
    print(f"  скорость: {gb_h:.1f} ГБ/час")
    for years in (10, 20, 30):
        gb = years * 10.4          # 6,8 базовых + 3,6 дополнительных на год
        print(f"  {years} лет ≈ {gb:.0f} ГБ → {gb / gb_h:.1f} ч загрузки")
PY
log "каталог замера можно удалить: rm -rf $DST"
log "=== ALL DONE ==="
