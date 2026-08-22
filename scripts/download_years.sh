#!/usr/bin/env bash
# Скачивание дополнительных лет ERA5 порциями — для запуска НА НОУТБУКЕ.
#
# Зачем так. Виртуалка берёт данные из S3, а положить туда можно только с
# корпоративного ноутбука. Сырой набор за 8 лет весит около 84 ГБ, и одновременно
# держать его и архивы места не хватит. Поэтому качаем по два года: собрал —
# сжал — удалил сырое. Пик занятости около 36 ГБ вместо 84.
#
# Объём по нашим же данным: 6,8 ГБ на год для 19 базовых каналов плюс 3,6 ГБ на
# дополнительные уровни (250 и 1000 гПа) — итого 10,4 ГБ на год.
#
# Что нужно на ноутбуке:
#   pip install numpy xarray zarr gcsfs dask
#   zstd (если нет — скрипт возьмёт gzip, архивы будут больше)
#
# Запуск:
#   bash scripts/download_years.sh 2002 2009            # в ./era5_out
#   bash scripts/download_years.sh 2002 2009 /path/out  # свой каталог
#
# После каждой порции в каталоге появляются два .tar.* — их и заливай в S3,
# потом можно удалять.
set -uo pipefail

FROM=${1:-2002}
TO=${2:-2009}
OUT=${3:-./era5_out}
STEP=2

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
LOG="$OUT/download.log"
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*" | tee -a "$LOG"; }

if command -v zstd >/dev/null; then Z="zstd"; EXT="tar.zst"; else Z="gzip"; EXT="tar.gz"; fi
# Фигурные скобки обязательны: дальше идёт длинное тире, и в локалях, где
# старшие байты считаются буквами (у Артура на ноутбуке именно так), bash
# втягивает его в имя переменной и падает на "FROM–: unbound variable".
log "=== ЗАГРУЗКА ERA5 ${FROM}–${TO}, порциями по ${STEP} года, сжатие ${Z} ==="
log "ожидаемый объём: примерно $(( (TO - FROM + 1) * 10 )) ГБ до сжатия"

pack() {  # pack <каталог> <имя архива>
  local dir=$1 name=$2
  if [[ "$Z" == "zstd" ]]; then
    tar -I 'zstd -3 -T0' -cf "$OUT/$name.tar.zst" -C "$(dirname "$dir")" "$(basename "$dir")"
  else
    tar -czf "$OUT/$name.tar.gz" -C "$(dirname "$dir")" "$(basename "$dir")"
  fi
}

for (( y=FROM; y<=TO; y+=STEP )); do
  y2=$(( y + STEP - 1 )); (( y2 > TO )) && y2=$TO
  tag="${y}_${y2}"
  base="$OUT/raw/wb2_512x256_19f_$tag"
  extra="$OUT/raw/global_512x256_extra_$tag"

  if [[ -f "$OUT/wb2_512x256_19f_$tag.$EXT" && -f "$OUT/global_512x256_extra_$tag.$EXT" ]]; then
    log "порция $tag уже упакована, пропускаю"
    continue
  fi

  log "--- порция $tag: базовые 19 каналов ---"
  mkdir -p "$OUT/raw"
  python3 scripts/build_dataset_512x256.py --out-dir "$base" \
      --start-year "$y" --end-year "$y2" --resume 2>&1 | tail -3 | tee -a "$LOG"
  [[ -f "$base/data.npy" ]] || { log "СБОЙ на базовых каналах, порция $tag"; exit 1; }

  log "--- порция $tag: дополнительные уровни ---"
  python3 scripts/build_dataset_512x256_30f.py --out-dir "$extra" --base-dir "$base" \
      --start-year "$y" --end-year "$y2" --resume 2>&1 | tail -3 | tee -a "$LOG"

  log "--- порция $tag: упаковка ---"
  pack "$base" "wb2_512x256_19f_$tag"
  [[ -d "$extra" ]] && pack "$extra" "global_512x256_extra_$tag"
  rm -rf "$base" "$extra"

  log "порция $tag готова: $(du -sh "$OUT"/*_"$tag".$EXT 2>/dev/null | awk '{printf "%s ", $1}')"
  log "свободно на диске: $(df -h "$OUT" | tail -1 | awk '{print $4}')"
done

rmdir "$OUT/raw" 2>/dev/null
log "=== ГОТОВО ==="
log "архивы в $OUT — залей их в S3, дальше их подхватит виртуалка"
ls -la "$OUT"/*.$EXT 2>/dev/null | awk '{print "  "$9" — "int($5/1024/1024/1024)" ГБ"}'
