#!/usr/bin/env bash
# Ночной прогон: корпус с признаками окрестности станции — и всё, что дальше.
#
# Зачем. Семь настроек постпроцессора легли в 2,289-2,331 °C: ни данные, ни
# ёмкость, ни регуляризация, ни наблюдения станции его больше не двигают.
# Потолок семейства (обучение и проверка на одном 2020 году) — 2,080, то есть
# из выучиваемой структуры взято уже около трёх четвертей. Значит нужны не
# лучшие настройки, а новые СВЕДЕНИЯ.
#
# Разбор по станциям говорит, каких именно: выигрыш связан с сырой ошибкой на
# 0,91, с модулем смещения на 0,85, с высотой на 0,72, у 26 станций из 71 его
# нет вовсе, а наибольший — у станций на 420-1850 м с холодным смещением до
# -4,7 °C. Правится несоответствие площадки ячейке сетки. При этом в корпусе до
# сих пор нет ничего про то, КАК изрезана местность вокруг станции: берётся один
# ближайший узел. Теперь берём шесть и считаем по ним разброс поля и рельефа.
#
# Число соседей задаётся аргументом: 6 узлов дали 0,71 % по температуре и
# 0,43 % по ветру при разбросе от жребия 0,17 % — то есть выигрыш настоящий, и
# логично спросить, растёт ли он с числом соседей. Двенадцать узлов охватывают
# круг вдвое шире: примерно 40 км вместо 29.
#
# Занимает около трёх с половиной часов на настройку. Всё считается само.
# Запуск: bash scripts/postproc/_run_night_krsk.sh [число соседей, по умолчанию 6]
# Лог:    /workdir/paper_results/night_krsk_nb<N>_master.log
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/night_krsk_nb${1:-6}_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
NB_ARG=${1:-6}
mkdir -p "$OUT"; exec >>"$OUT/night_krsk_nb${NB_ARG}_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.night_krsk_nb${NB_ARG}.lock"
flock -n 9 || { log "уже работает — стоп"; exit 1; }
NB=${1:-6}
log "=== НОЧНОЙ ПРОГОН: окрестность станции, $NB узлов ==="

# 1. Корпус с шестью соседними узлами. Раннер корпуса сам уходит в фон, поэтому
# зовём его уже «раздемонизированным», чтобы дождаться конца.
TAG=2016_2020_nb${NB}
CORPUS=""
# С break: без него цикл брал ПОСЛЕДНЕЕ совпадение, то есть копию в /workdir
# вместо оригинала на /data, и все производные наборы уходили под квоту 8 ГБ.
for c in /data/postproc/corpus_krsk_${TAG}.parquet data/postproc/corpus_krsk_${TAG}.parquet; do
  [[ -f "$c" ]] && { CORPUS="$c"; break; }
done
if [[ -z "$CORPUS" ]]; then
  log "шаг 1: корпус с окрестностью (около 2,5 часов)"
  DAEMONIZED=1 NB=$NB bash scripts/postproc/_run_corpus_krsk.sh 2016 2020
  for c in /data/postproc/corpus_krsk_${TAG}.parquet data/postproc/corpus_krsk_${TAG}.parquet \
           /data/postproc/corpus_krsk_${TAG}.pkl.gz data/postproc/corpus_krsk_${TAG}.pkl.gz; do
    [[ -f "$c" ]] && { CORPUS="$c"; break; }
  done
  [[ -z "$CORPUS" ]] && { log "корпус не собрался — см. corpus_${TAG}_master.log"; exit 1; }
fi
log "шаг 1 готов: $CORPUS ($(du -h "$CORPUS" | cut -f1))"

source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
DIR=/data/postproc; mkdir -p "$DIR" 2>/dev/null || DIR=$(dirname "$CORPUS")

# 2. Признаки-наблюдения. Климатическая норма — только по годам обучения.
LAGS="$DIR/corpus_krsk_lags2_nb${NB}.parquet"
if [[ ! -f "$LAGS" ]]; then
  log "шаг 2: признаки-наблюдения"
  python -u scripts/postproc/add_obs_lags.py --in "$CORPUS" --out "$LAGS" \
      --clim-years 2016 2017 2018 2>&1 | tail -4
  [[ -f "$LAGS" ]] || { log "признаки не собрались — стоп"; exit 1; }
fi

# 3. Базовые линии на той же выборке, что у сети.
log "шаг 3: базовые линии"
python -u scripts/postproc/baselines.py --corpus "$LAGS" \
    --train-years 2016 2017 2018 --test-years 2020 --complete-obs 2>&1 \
    | grep --line-buffered -E "только полные|^=== |сырой прогноз|станция×месяц×час|таблица \+"

# 4. Деление по годам и обучение.
if [[ ! -f "$DIR/nb${NB}_test.parquet" ]]; then
  log "шаг 4: деление по годам"
  python -u scripts/postproc/split_corpus.py --in "$LAGS" --out-dir "$DIR" \
      --prefix nb${NB} train=2016,2017,2018 val=2019 test=2020
  [[ -f "$DIR/nb${NB}_test.parquet" ]] || { log "деление не удалось — стоп"; exit 1; }
fi

EXP=experiments/neural_postproc_krsk_nb${NB}
if [[ ! -f "$EXP/best_model.pth" ]]; then
  log "шаг 5: обучение, 20 эпох"
  python -u scripts/postproc/train_neural_postproc_v3.py \
      --train-parquet "$DIR/nb${NB}_train.parquet" --val-parquet "$DIR/nb${NB}_val.parquet" \
      --out-dir "$EXP" --epochs 20 --batch-size 4096 --station-emb-dim 32 \
      --hidden 192,192,128 2>&1 \
      | grep --line-buffered -E "^\[(cfg|model|dataset|ep )|^Done:"
  # --line-buffered обязателен: без него grep копит строки в буфере и при
  # записи в файл эпохи не видны до самого конца обучения. 28.08.2026 из-за
  # этого прогон выглядел зависшим, хотя шёл нормально.
fi
[[ -f "$EXP/best_model.pth" ]] || { log "весов нет — стоп"; exit 1; }

log "шаг 6: проверка на 2020"
python -u scripts/postproc/eval_per_lead_v2.py \
    --val-parquet "$DIR/nb${NB}_test.parquet" --ckpt "$EXP/best_model.pth" \
    --out-dir "$EXP/eval_test2020" 2>&1 | grep --line-buffered -E "^\[eval\]|^Overall"
python -u scripts/postproc/record_run.py \
    --eval-json "$EXP/eval_test2020/eval_per_lead_v2.json" \
    --name "$(basename "$EXP")" --note "окрестность $NB узлов" 2>&1 | tail -2
log "шаг 7: разбор по станциям"
python -u scripts/postproc/eval_per_station_v2.py \
    --val-parquet "$DIR/nb${NB}_test.parquet" --ckpt "$EXP/best_model.pth" \
    --out-dir "$EXP/eval_stations" --bbox 50 60 83 98 --label krsk 2>&1 | tail -6
log "=== ALL DONE ==="
log "сравнивать с: без окрестности t2m 2,297 / ветер 1,727; лучшая таблица 2,498 / 1,916"
