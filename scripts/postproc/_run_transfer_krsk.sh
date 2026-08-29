#!/usr/bin/env bash
# Перенос поправки на станции, которых модель не видела.
#
# ЗАЧЕМ. Это главный незакрытый вопрос для оперативного применения. Если
# постпроцессор работает только там, где у него годы наблюдений, поставить его
# на новую площадку нельзя — а таких площадок большинство. Поставить вопрос до
# сих пор было невозможно: у модели вложение станции, и для новой строки просто
# нет, никакое приближение её не заменит.
#
# Придерживаем 14 станций из 71 (по жребию, зерно 42) и обучаем на остальных 57
# ДВЕ модели: обычную, с вложением станции, и без привязки — там площадку
# описывают только признаки: широта, долгота, высота, разность высот с рельефом
# модели.
#
# Что получим:
#   с вложением, знакомые станции   — сколько теряем, отказавшись от вложения
#   без привязки,  знакомые станции — цена отказа
#   без привязки,  НЕЗНАКОМЫЕ       — ответ на сам вопрос
# Сырой прогноз печатается рядом в каждом случае, так что выигрыш считается на
# своей выборке и подмены не происходит.
#
# Серия задаётся аргументами вида «придержать:жребий». По умолчанию — то, что
# нужно для статьи: три жребия при 14 придержанных станциях и три при 28.
#
# Три жребия обязательны. При одном делении нельзя отличить потерю ОТ ПЕРЕНОСА
# от того, что жребий выдал станции труднее среднего: у придержанных 29.08.2026
# сырая ошибка была 3,231 против 2,917 у остальных, и часть падения выигрыша
# объясняется этим, а не переносом.
#
# Две величины придержанного отвечают на второй вопрос: упирается перенос в
# ЧИСЛО обучающих станций или в их РАЗНООБРАЗИЕ. При 28 придержанных обучающая
# выборка вдвое меньше; если выигрыш на незнакомых не изменится — дело не в
# объёме.
#
#   bash scripts/postproc/_run_transfer_krsk.sh                 # вся серия, ~1,5 ч
#   bash scripts/postproc/_run_transfer_krsk.sh 14:42           # одна настройка
#
# Лог: /workdir/paper_results/transfer_krsk_master.log
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/transfer_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
mkdir -p "$OUT"; exec >>"$OUT/transfer_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.transfer_krsk.lock"
flock -n 9 || { log "уже работает — стоп"; exit 1; }
SERIES=("$@")
[[ ${#SERIES[@]} -eq 0 ]] && SERIES=(14:42 14:43 14:44 28:42 28:43 28:44)
log "=== ПЕРЕНОС НА НЕЗНАКОМЫЕ СТАНЦИИ: ${SERIES[*]} ==="
BUSY=$(pgrep -af "^python.*(src\.main|build_corpus\.py|train_neural)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# Берём набор с окрестностью из 12 узлов: на нём выигрыш вышел на полку
# (2,271 против 2,277 у шести и 2,269 у двадцати четырёх).
# /data стирается при каждом перезапуске виртуалки, а он случается и от часа
# простоя — вместе с /data исчезает venv. Восстанавливаем его сами: данные у нас
# свои, в parquet, поэтому датасеты распаковывать не нужно (VENV_ONLY=1).
if [[ ! -x "$VENV/bin/python" ]]; then
  log "окружения нет (стёрлось с /data) — восстанавливаю, это несколько минут"
  VENV_ONLY=1 bash scripts/_paper_setup_vm.sh >>"$OUT/venv_restore.log" 2>&1
  log "восстановление rc=$? (подробности в venv_restore.log)"
fi
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }

# Проверяем не наличие venv, а то, что он РАБОТАЕТ для нашей задачи: весь
# корпус лежит в parquet, и без движка чтения всё падает на первом же файле.
# Проверять по факту, а не по списку зависимостей: окружение могло остаться от
# прежней версии requirements.
if ! python -c "import pyarrow" 2>/dev/null; then
  log "нет pyarrow (корпус в parquet без него не прочитать) — ставлю"
  pip install -q pyarrow \
      --extra-index-url https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple \
      >>"$OUT/venv_restore.log" 2>&1
  python -c "import pyarrow" 2>/dev/null \
    || { log "pyarrow не поставился — см. venv_restore.log"; exit 1; }
  log "pyarrow готов"
fi
export PYTHONPATH="$REPO"
DIR=/data/postproc; mkdir -p "$DIR" 2>/dev/null || DIR=data/postproc

# Ищем набор с признаками-наблюдениями. Если его нет — ищем сам корпус и
# достраиваем признаки: это полминуты против четырёх часов развёртки.
#
# Нужно потому, что /data стирается при каждом перезапуске виртуалки, а он
# случается и от часа простоя. Корпус при этом копируется в /workdir и
# переживает перезапуск, а производные наборы — нет. Раннер, который в такой
# ситуации просто отказывается, отправляет считать заново то, что уже посчитано.
# Порядок предпочтения — по качеству корпуса, а не по тому, что подвернулось:
# nb12 (окрестность вышла на полку), затем nb6, затем без окрестности. Внутри
# каждого сначала ищем готовые признаки, потом сам корпус — достроить признаки
# стоит полминуты, а развёртка корпуса четыре часа, так что выбрать готовый
# худший корпус вместо сырого лучшего было бы неверной экономией.
find_first() { for c in "$@"; do [[ -f "$c" ]] && { echo "$c"; return; }; done; }

LAGS=""; RAW=""
for suf in _nb12 _nb6 ""; do
  LAGS=$(find_first "$DIR/corpus_krsk_lags2${suf:-_plain}.parquet" \
                    "data/postproc/corpus_krsk_lags2${suf:-_plain}.parquet")
  [[ -n "$LAGS" ]] && break
  RAW=$(find_first "$DIR/corpus_krsk_2016_2020${suf}.parquet" \
                   "data/postproc/corpus_krsk_2016_2020${suf}.parquet")
  if [[ -n "$RAW" ]]; then
    LAGS="$DIR/corpus_krsk_lags2${suf:-_plain}.parquet"
    log "признаков для${suf:- базового} набора нет, но корпус на месте: $RAW"
    log "достраиваю признаки-наблюдения (полминуты) -> $LAGS"
    python -u scripts/postproc/add_obs_lags.py --in "$RAW" --out "$LAGS" \
        --clim-years 2016 2017 2018 2>&1 | tail -3
    [[ -f "$LAGS" ]] && break
    LAGS=""
  fi
done

if [[ -z "$LAGS" ]]; then
  log "нет ни набора с признаками, ни самого корпуса."
  log "  Смотрел в $DIR и data/postproc. Корпус стирается вместе с /data при"
  log "  перезапуске виртуалки; если копии в /workdir тоже нет — придётся"
  log "  пересобрать: bash scripts/postproc/_run_night_krsk.sh 12"
  exit 1
fi
log "корпус: $LAGS ($(du -h "$LAGS" | cut -f1))"

run_one() {   # придержать, жребий
  local hold=$1 seed=$2 tag="h${1}s${2}"
  log ""
  log "########## придержано $hold станций, жребий $seed ##########"

  if [[ ! -f "$DIR/${tag}_seen_test.parquet" ]]; then
    log "делю по станциям и годам"
    python -u scripts/postproc/split_stations.py --in "$LAGS" --out-dir "$DIR" \
        --prefix "$tag" --holdout "$hold" --seed "$seed" || return 1
    for part in seen unseen; do
      python -u scripts/postproc/split_corpus.py --in "$DIR/${tag}_${part}.parquet" \
          --out-dir "$DIR" --prefix "${tag}_${part}" \
          train=2016,2017,2018 val=2019 test=2020 || return 1
    done
  fi

  local exp="experiments/neural_postproc_transfer_${tag}"
  if [[ -f "$exp/best_model.pth" ]]; then
    log "веса уже есть"
  else
    log "обучение без привязки к станции, 20 эпох"
    python -u scripts/postproc/train_neural_postproc_v3.py \
        --train-parquet "$DIR/${tag}_seen_train.parquet" \
        --val-parquet   "$DIR/${tag}_seen_val.parquet" \
        --out-dir "$exp" --epochs 20 --batch-size 4096 \
        --hidden 192,192,128 --no-station-emb 2>&1 \
        | grep --line-buffered -E "^\[(cfg|model|dataset|ep )|^Done:"
    [[ -f "$exp/best_model.pth" ]] || { log "весов нет — пропускаю"; return 1; }
  fi

  for part in seen unseen; do
    python -u scripts/postproc/eval_per_lead_v2.py \
        --val-parquet "$DIR/${tag}_${part}_test.parquet" --ckpt "$exp/best_model.pth" \
        --out-dir "$exp/eval_${part}" 2>&1 \
        | grep --line-buffered -E "^\[eval\]|^Overall|незнакомых"
    [[ -f "$exp/eval_${part}/eval_per_lead_v2.json" ]] && \
      python -u scripts/postproc/record_run.py \
          --eval-json "$exp/eval_${part}/eval_per_lead_v2.json" \
          --name "transfer_${tag}_${part}" \
          --note "перенос: придержано $hold, жребий $seed, станции $part" 2>&1 | tail -1
  done
}

for item in "${SERIES[@]}"; do
  run_one "${item%%:*}" "${item##*:}"
done

log "=== ALL DONE ==="
log "сравнивать: разброс между жребиями при одном и том же числе придержанных"
log "  показывает, сколько в 15,1 % от трудности выборки, а не от переноса;"
log "  разница между 14 и 28 придержанными — упирается ли перенос в число станций"
