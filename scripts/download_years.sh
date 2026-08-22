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

# Упаковка через конвейер, а не через tar -I.
#
# 22.08.2026: на macOS tar — это bsdtar, у которого -I значит совсем другое
# (список файлов), и он попытался открыть файл с именем "zstd -3 -T0". tar
# отработал с ошибкой, архив вышел нулевым, а скрипт следом удалил сырые
# данные — четыре часа загрузки в никуда. Конвейер работает и с GNU tar, и с
# bsdtar, а функция возвращает ненулевой код при любом сбое.
pack() {  # pack <каталог> <имя архива> → 0 только если архив реально собран
  local dir=$1 name=$2 out rc
  if [[ "$Z" == "zstd" ]]; then
    out="$OUT/$name.tar.zst"
    tar -cf - -C "$(dirname "$dir")" "$(basename "$dir")" | zstd -3 -T0 -q -f -o "$out"
    rc=("${PIPESTATUS[@]}")
    [[ "${rc[0]}" == 0 && "${rc[1]}" == 0 ]] || { echo "  tar rc=${rc[0]} zstd rc=${rc[1]}"; rm -f "$out"; return 1; }
    zstd -t "$out" 2>/dev/null || { echo "  архив не проходит проверку целостности"; rm -f "$out"; return 1; }
  else
    out="$OUT/$name.tar.gz"
    tar -czf "$out" -C "$(dirname "$dir")" "$(basename "$dir")" || { rm -f "$out"; return 1; }
    gzip -t "$out" 2>/dev/null || { echo "  архив не проходит проверку целостности"; rm -f "$out"; return 1; }
  fi
  # Порция весит гигабайты; всё, что меньше сотни мегабайт, — признак обрыва.
  local sz; sz=$(wc -c < "$out" | tr -d ' ')
  if (( sz < 100000000 )); then
    echo "  архив подозрительно мал: $sz байт"; rm -f "$out"; return 1
  fi
  return 0
}

# Проверяем упаковку на игрушечном каталоге ДО первой загрузки: сбой должен
# стоить две секунды, а не четыре часа.
smoke_pack() {
  local d="$OUT/.packtest" f
  rm -rf "$d"; mkdir -p "$d"
  # 120 МБ нулей, чтобы пройти и порог размера, и сжатие
  dd if=/dev/zero of="$d/probe.bin" bs=1048576 count=120 2>/dev/null
  if pack "$d" ".packtest_probe"; then
    rm -rf "$d" "$OUT/.packtest_probe.$EXT"
    log "проверка упаковки пройдена ($Z)"
    return 0
  fi
  rm -rf "$d" "$OUT/.packtest_probe.$EXT"
  log "УПАКОВКА НЕ РАБОТАЕТ ($Z) — загрузку не начинаю, иначе данные будет некуда девать"
  return 1
}
smoke_pack || exit 1

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
  ok=1
  pack "$base" "wb2_512x256_19f_$tag" || ok=0
  if [[ $ok == 1 && -d "$extra" ]]; then
    pack "$extra" "global_512x256_extra_$tag" || ok=0
  fi
  # Сырое удаляем ТОЛЬКО после успешной упаковки и проверки целостности.
  if [[ $ok != 1 ]]; then
    log "УПАКОВКА ПОРЦИИ $tag НЕ УДАЛАСЬ — сырые данные оставляю в $OUT/raw, останавливаюсь"
    log "починить упаковку и запустить скрипт заново: уже скачанное подхватится по --resume"
    exit 1
  fi
  rm -rf "$base" "$extra"

  log "порция $tag готова: $(du -sh "$OUT"/*_"$tag".$EXT 2>/dev/null | awk '{printf "%s ", $1}')"
  log "свободно на диске: $(df -h "$OUT" | tail -1 | awk '{print $4}')"
done

rmdir "$OUT/raw" 2>/dev/null
log "=== ГОТОВО ==="
log "архивы в $OUT — залей их в S3, дальше их подхватит виртуалка"
ls -la "$OUT"/*.$EXT 2>/dev/null | awk '{print "  "$9" — "int($5/1024/1024/1024)" ГБ"}'
