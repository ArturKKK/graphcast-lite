#!/usr/bin/env bash
# Сборка корпуса постобработки на красноярской модели.
#
# Что делает: разворачивает модель авторегрессионно от начальных полей ERA5,
# выбирает значения в точках 71 станции области, сопоставляет с наблюдениями
# ISD-Lite и складывает в parquet. Сроки прогноза 6-120 ч, инициализации в
# 00 и 12 UTC.
#
# Собирается на красноярской модели chw — основной в статье. Кадры читаются
# прямо из глобальной сетки и региональной вставки; слитый датасет на 81 ГБ,
# на сборке которого цепочка падала, не нужен.
#
# Про порядок узлов, раз он тут восстанавливается построением, а не читается из
# слитого набора. Совпало число узлов — 133 279, из них 2501 региональный,
# ровно как в наборе, на котором обучалась модель. Но само по себе это ещё не
# совпадение порядка, поэтому две подстраховки. Первая: сети всё равно, в каком
# порядке пронумерованы узлы — обучаемых весов, привязанных к номеру узла, в
# ней нет вовсе, а рёбра строятся по координатам, так что перенумерация даёт
# тот же граф и тот же прогноз, лишь переставленный так же. Значения в точках
# станций всё равно берутся по координатам. Существенно только то, чтобы
# координаты и данные были согласованы между собой. Вторая: это и проверяется
# на первом же сроке счёта — по гладкости поля, отдельно по всей сетке и
# отдельно по вставке (см. check_node_order в build_corpus.py).
#
# Годы задаются аргументами. На одной машине весь диапазон сразу:
#   bash scripts/postproc/_run_corpus_krsk.sh 2016 2020
# либо, если хочется поделить между двумя картами, по половине и потом слить.
#
# Сам уходит в фон. Лог: /workdir/paper_results/corpus_<годы>_master.log
set -uo pipefail
Y0=${1:-2016}; Y1=${2:-2020}; EXP=${3:-}
TAG="${Y0}_${Y1}"
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне, годы $Y0-$Y1. лог: /workdir/paper_results/corpus_${TAG}_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
DATA=/data/datasets
mkdir -p "$OUT"; exec >>"$OUT/corpus_${TAG}_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
# Замок на весь раннер целиком. Сторож ниже смотрит на процесс счёта, но
# подготовка идёт минут пятнадцать, и в это окно 28.08.2026 проскочил второй
# запуск — две установки затоптали друг друга, слияние вернуло код 4.
exec 9>"$OUT/.corpus_${TAG}.lock"
flock -n 9 || { log "уже работает другой раннер этих лет — стоп"; exit 1; }
log "=== КОРПУС ПОСТОБРАБОТКИ, годы $Y0-$Y1 ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py|build_corpus\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

GBASE="$DATA/wb2_512x256_19f_ar"
RBASE="$DATA/region_krsk_61x41_19f_2010-2020_025deg"
REXTRA="$DATA/region_krsk_61x41_extra_2010-2020_025deg"
GEXTRA="$DATA/global_512x256_extra_2010-2021_07deg"
META="$DATA/multires_krsk_33f_meta"

# Слитый датасет на 81 ГБ нам не нужен: сборщик читает кадры прямо из
# глобальной и региональной сеток, а порядок узлов восстанавливается тем же
# построением, что и при слиянии. Раньше цепочка падала именно на его сборке.
if [[ ! -x "$VENV/bin/python" || ! -f "$GBASE/data.npy" || ! -f "$RBASE/data.npy" ]]; then
  log "подготовка окружения и датасетов"
  bash scripts/_paper_setup_vm.sh >> "$OUT/corpus_${TAG}_setup.log" 2>&1
  log "подготовка rc=$? (сбой на сборке слитого датасета не помеха — он не нужен)"
fi
# У базовых сеток массив зовётся data.npy, у каталогов *_extra_* —
# data_extra.npy (его и открывает build_corpus.py). Проверка, требовавшая
# data.npy от всех четырёх, останавливала запуск на файле, которого там
# отродясь не было: 28.08.2026 на этом потеряно полдня.
for d in "$GBASE" "$RBASE"; do
  [[ -f "$d/data.npy" ]] || { log "нет $d/data.npy — стоп"; exit 1; }
done
for d in "$GEXTRA" "$REXTRA"; do
  [[ -f "$d/data_extra.npy" ]] || {
    log "нет $d/data_extra.npy — стоп"
    log "  в каталоге сейчас: $(ls "$d" 2>/dev/null | tr '\n' ' ' || echo 'каталога нет')"
    exit 1; }
done
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# Сборщику из multires-каталога нужны только coords.npz, scalers.npz и
# variables.json — сам массив он не читает. Полный 33-канальный датасет весит
# 56 ГБ и вместе со слитым источником на 81 ГБ упирается в лимит диска
# платформы, поэтому собираем одни метаданные.
# Слитый датасет не обязателен, но если он на диске — берём его: именно на
# этом пути сборка досчитала до конца 27.08.2026, тогда как обход собран
# позже и целиком ни разу не прогонялся. Координаты при этом берутся ровно те,
# с которыми обучалась модель, а не восстановленные построением.
MERGE="$DATA/multires_krsk_19f_merge"
# Набор аргументов повторяет ровно тот, на котором сборка досчитала до конца:
# при слитом источнике отдельные сетки не передаются вовсе.
META_MERGE=()
if [[ -f "$MERGE/data.npy" && -f "$MERGE/coords.npz" ]]; then
  SRC_ARGS=(--merged-base "$MERGE"); META_MERGE=(--merged "$MERGE")
  log "источник кадров: слитый $MERGE"
else
  SRC_ARGS=(--global-base "$GBASE" --regional-base "$RBASE")
  log "слитого источника нет — читаю кадры из глобальной сетки и вставки"
fi

if [[ ! -f "$META/scalers.npz" ]]; then
  log "собираю метаданные 33-канального датасета"
  python -u scripts/postproc/make_multires33f_meta.py "${META_MERGE[@]}" \
      --global-base "$GBASE" --region-base "$RBASE" --extra "$GEXTRA" \
      --roi 50 60 83 98 --expect-nodes 133279 --out "$META" 2>&1 | tail -16
  [[ -f "$META/scalers.npz" ]] || { log "метаданные не собрались — стоп"; exit 1; }
fi

# Модель. Предпочитаем chw (основная в статье); если её веса не пережили
# пересоздание виртуалки — падаем на базовую 33-канальную, она лежит в git.
if [[ -z "$EXP" ]]; then
  if [[ -f experiments/multires_krsk_33f_chw/checkpoint.pth ]]; then
    EXP=multires_krsk_33f_chw
  else
    EXP=multires_krsk_33f
    log "ВНИМАНИЕ: весов chw нет, беру базовую модель $EXP — корпус будет"
    log "  собран на менее точном прогнозе, чем в статье"
  fi
fi
[[ -f "experiments/$EXP/checkpoint.pth" || -f "experiments/$EXP/best_model.pth" ]] \
  || { log "нет весов у $EXP — стоп"; exit 1; }
log "модель: $EXP"

# Наблюдения. На виртуалке интернета нет, но общероссийский набор ISD-Lite уже
# лежит в /data/datasets — наши станции внутри него, отдельная выборка не
# нужна: сборщик ищет файлы по номеру станции и году.
ISD=""
for cand in "$DATA/isd_lite_russia" data/isd_lite_krsk data/isd_lite_russia; do
  if [[ -d "$cand" ]] && [[ $(ls "$cand" 2>/dev/null | wc -l) -gt 100 ]]; then ISD="$cand"; break; fi
done
if [[ -z "$ISD" ]]; then
  log "наблюдений нет ни в $DATA/isd_lite_russia, ни в data/isd_lite_*"
  log "  без интернета их надо закинуть на машину — 12 МБ по 71 станции за 2016-2020"
  exit 1
fi
log "наблюдения: $ISD ($(ls "$ISD" | wc -l) файлов)"

# Корпус кладём на /data, а не в /workdir: там квота 8 ГБ, а сам корпус и его
# черновик тянут около гигабайта каждый. /data рестарт не переживает, поэтому
# готовый файл в конце копируем в репозиторий, если он туда влезает.
CORPUS_DIR=/data/postproc
mkdir -p "$CORPUS_DIR"
# Если pyarrow есть в окружении — пишем parquet, иначе сборщик сам откатится на
# pickle. Попытка доустановить дешёвая и на машине без сети просто не сработает.
# Попытка доустановить pyarrow: строго с таймаутом и без повторов. Без сети
# pip уходит в долгие ретраи с нарастающей паузой, и раннер выглядит зависшим.
if ! python -c "import pyarrow" 2>/dev/null; then
  log "pyarrow нет — одна попытка поставить, 60 секунд"
  timeout 60 pip install -q --retries 0 --timeout 10 pyarrow \
      >> "$OUT/corpus_${TAG}_pip.log" 2>&1
  if python -c "import pyarrow" 2>/dev/null; then
    log "pyarrow поставлен"
  else
    log "pyarrow недоступен — корпус будет записан в pickle, это штатно"
  fi
fi
log "START сборки (развёртка до 120 ч, инициализации 00 и 12 UTC)"
ARGS=(scripts/postproc/build_corpus.py
    --experiment-dir "experiments/$EXP"
    --multires-dir   "$META"
    "${SRC_ARGS[@]}"
    --global-extra   "$GEXTRA"
    --regional-extra "$REXTRA"
    --stations-json  data/krsk_postproc_stations.json
    --isd-dir        "$ISD"
    --top-stations   71
    --years "$Y0" "$Y1"
    --out-parquet    "$CORPUS_DIR/corpus_krsk_${TAG}.parquet")
# Подробности уходят в build-лог, а вехи — сюда, в master. Раньше проверка
# порядка узлов печаталась только в build-лог, и в master её искали впустую.
# Статус берём у python, а не у конвейера: grep вернёт 1, если вех не было.
VEHI="^(\[(порядок узлов|part|cfg|model|stations|inits|inference|join|done|save)\]|  \[[0-9])"
python -u "${ARGS[@]}" 2>&1 | tee -a "$OUT/corpus_${TAG}_build.log" \
    | grep --line-buffered -E "$VEHI"
RC=${PIPESTATUS[0]}
log "DONE сборка rc=$RC"

# Развёртка идёт больше двух часов и по ходу сбрасывает посчитанное в черновик.
# Если упало уже после неё — на сшивке или записи — пересчитывать нечего:
# досбираем из черновика.
PART="$CORPUS_DIR/corpus_krsk_${TAG}.partial.pkl"
if [[ $RC -ne 0 && -s "$PART" ]]; then
  log "упало (rc=$RC), но черновик на месте ($(du -h "$PART" | cut -f1)) — досбор без пересчёта"
  python -u "${ARGS[@]}" --from-partial 2>&1 | tee -a "$OUT/corpus_${TAG}_build.log" \
      | grep --line-buffered -E "$VEHI"
  RC=${PIPESTATUS[0]}
  log "DONE досбор rc=$RC"
fi
CORPUS=$(ls -1 "$CORPUS_DIR"/corpus_krsk_${TAG}.parquet "$CORPUS_DIR"/corpus_krsk_${TAG}.pkl.gz 2>/dev/null | head -1)
if [[ -n "$CORPUS" ]]; then
  log "корпус: $CORPUS ($(du -h "$CORPUS" | cut -f1))"
  # Дубль в /workdir, чтобы корпус пережил пересоздание виртуалки, — но только
  # если после копии останется хотя бы гигабайт свободного места.
  SZ=$(du -m "$CORPUS" | cut -f1)
  FREE=$(df -Pm /workdir | awk 'NR==2{print $4}')
  if (( FREE - SZ > 1024 )); then
    mkdir -p data/postproc && cp -p "$CORPUS" data/postproc/ \
      && log "копия в data/postproc/$(basename "$CORPUS") (в /workdir было $FREE МБ)"
  else
    log "в /workdir всего $FREE МБ, копию не делаю — корпус только на /data,"
    log "  а он не переживёт пересоздание виртуалки"
  fi
  python -u - "$CORPUS" <<'PYEOF'
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_pickle(p, compression="gzip") if p.endswith(".pkl.gz") else pd.read_parquet(p)
print(f"[итог] строк {len(df):,}, столбцов {len(df.columns)}")
print(f"[итог] станций {df['station_usaf'].nunique()}, "
      f"сроков прогноза {sorted(df['lead_h'].unique())}")
print(f"[итог] период {df['valid_time_utc'].min()} — {df['valid_time_utc'].max()}")
PYEOF
else
  log "корпус не собрался — см. corpus_${TAG}_build.log"
fi
log "=== ALL DONE ==="
