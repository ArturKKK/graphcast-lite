#!/usr/bin/env bash
# Постобработка красноярского корпуса: признаки-наблюдения и базовые линии.
#
# Считает две настройки таблицы поправок и сравнивает их:
#   A — обучение на 2016-2018. Эти годы сеть видела при обучении (разбивка
#       хронологическая, последние 20% отданы под контроль, см.
#       src/data/dataloader_chunked.py:219 — обучение кончается около
#       октября 2018). Прогноз там оптимистично хорош.
#   B — обучение на 2019. Этот год сеть при обучении не видела.
# Проверка в обоих случаях на 2020, который сеть не видела тоже. Если A заметно
# слабее B на проверке, значит поправка, настроенная на выученных годах, на
# рабочих данных не работает — а это ровно то, как постобработку и применяют.
#
# Климатология станций считается только по годам обучения своей настройки:
# иначе в признак obs_t2m_anom протекает проверочный год.
#
# Запуск: bash scripts/postproc/_run_postproc_krsk.sh
# Лог:    /workdir/paper_results/postproc_krsk_master.log
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/postproc_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
mkdir -p "$OUT"; exec >>"$OUT/postproc_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.postproc_krsk.lock"
flock -n 9 || { log "уже работает другой раннер — стоп"; exit 1; }
log "=== ПОСТОБРАБОТКА, красноярский корпус ==="

# Корпус мог остаться от старого запуска в /workdir или лежать на /data.
CORPUS=""
for c in /data/postproc/corpus_krsk_2016_2020.parquet \
         data/postproc/corpus_krsk_2016_2020.parquet \
         /data/postproc/corpus_krsk_2016_2020.pkl.gz \
         data/postproc/corpus_krsk_2016_2020.pkl.gz; do
  [[ -f "$c" ]] && { CORPUS="$c"; break; }
done
[[ -z "$CORPUS" ]] && { log "корпус не найден — сначала _run_corpus_krsk.sh"; exit 1; }
log "корпус: $CORPUS ($(du -h "$CORPUS" | cut -f1))"
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

DIR=$(dirname "$CORPUS")
run_variant() {   # имя, годы обучения (через запятую для имени файла), годы обучения списком
  local tag=$1; shift
  local -a years=("$@")
  # lags2: в прежних файлах признаки ошибки звались err_lag* — без имени
  # переменной. Теперь их три набора, по одному на переменную, поэтому имя файла
  # тоже другое: иначе раннер молча взял бы старый и посчитал бы вчерашнее.
  local lags="$DIR/corpus_krsk_lags2_${tag}.parquet"
  rm -f "$DIR/corpus_krsk_lags_${tag}.parquet"
  log "--- настройка $tag: обучение ${years[*]}, проверка 2020"
  if [[ ! -f "$lags" ]]; then
    log "    признаки-наблюдения (климатология только по ${years[*]})"
    python -u scripts/postproc/add_obs_lags.py --in "$CORPUS" --out "$lags" \
        --clim-years "${years[@]}" 2>&1 | tail -12
    [[ -f "$lags" ]] || { log "    не собрались признаки — пропускаю настройку"; return 1; }
  else
    log "    признаки уже посчитаны"
  fi
  log "    базовые линии"
  python -u scripts/postproc/baselines.py --corpus "$lags" \
      --train-years "${years[@]}" --test-years 2020 --per-lead 2>&1
}

run_variant seen 2016 2017 2018
run_variant unseen 2019
log "=== ALL DONE ==="
