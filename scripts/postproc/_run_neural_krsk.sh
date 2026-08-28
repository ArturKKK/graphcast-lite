#!/usr/bin/env bash
# Нейронный постпроцессор на красноярском корпусе.
#
# Сравнение честное: обучение на 2016-2018, проверка на 2020 — ровно те же годы,
# на которых считались базовые линии. 2019 отдан под отбор эпохи, чтобы
# проверочный год не участвовал в обучении вообще никак.
#
# С чем сравниваем (лучший способ базовых линий на 2020):
#   t2m 2.636 (15.0%),  10u 1.444 (17.0%),  10v 1.259 (20.6%)
# Если сеть их не бьёт — значит признаков ей хватает, а нелинейности не нужны,
# и это тоже результат, который стоит написать.
#
# Запуск: bash scripts/postproc/_run_neural_krsk.sh [эпох]
# Лог:    /workdir/paper_results/neural_krsk_master.log
set -uo pipefail
EPOCHS=${1:-20}
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/neural_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
mkdir -p "$OUT"; exec >>"$OUT/neural_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.neural_krsk.lock"
flock -n 9 || { log "уже работает другой раннер — стоп"; exit 1; }
log "=== НЕЙРОННЫЙ ПОСТПРОЦЕССОР ==="
BUSY=$(pgrep -af "^python.*(src\.main|build_corpus\.py|train_neural)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# Берём набор с климатологией по 2016-2018: норма обязана считаться только по
# годам обучения, иначе признак отклонения тянет в себя проверочный год.
LAGS=""
for c in /data/postproc/corpus_krsk_lags2_seen.parquet \
         data/postproc/corpus_krsk_lags2_seen.parquet; do
  [[ -f "$c" ]] && { LAGS="$c"; break; }
done
[[ -z "$LAGS" ]] && { log "нет корпуса с признаками — сначала _run_postproc_krsk.sh"; exit 1; }
log "корпус: $LAGS ($(du -h "$LAGS" | cut -f1))"
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

DIR=$(dirname "$LAGS"); EXP=experiments/neural_postproc_krsk
if [[ ! -f "$DIR/krsk_test.parquet" ]]; then
  log "делю корпус по годам: обучение 2016-2018, отбор 2019, проверка 2020"
  python -u - "$LAGS" "$DIR" <<'PYEOF'
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
y = pd.to_datetime(df["valid_time_utc"]).dt.year
for name, years in (("train", [2016, 2017, 2018]), ("val", [2019]), ("test", [2020])):
    part = df[y.isin(years)]
    p = f"{out}/krsk_{name}.parquet"
    part.to_parquet(p, index=False)
    print(f"  {name}: {len(part):,} строк -> {p}", flush=True)
PYEOF
  [[ -f "$DIR/krsk_test.parquet" ]] || { log "деление не удалось — стоп"; exit 1; }
fi

# Базовые линии на ТОЙ ЖЕ выборке. Датасет сети выбрасывает строку, если
# пропущена любая из трёх целей, — а ветра нет у каждой пятой. Прежние базовые
# линии считались по температуре на всех 618 тысячах строк, и сравнивать с ними
# было бы подменой выборки. Считаем заново на выборке сети.
log "базовые линии на выборке сети (только полные наблюдения)"
python -u scripts/postproc/baselines.py --corpus "$LAGS" \
    --train-years 2016 2017 2018 --test-years 2020 --complete-obs 2>&1 \
    | grep -E "только полные|обучение |сырой прогноз|станция×месяц×час|регрессия|таблица \+"

# Две настройки. Основная видит признаки-наблюдения станции, абляция — нет.
# Без абляции неясно, за счёт чего выигрыш: линейная регрессия на тех же
# признаках даёт 12,1%, а без них 0,5%, так что вклад самих наблюдений огромен,
# и надо отделить его от вклада нелинейности и вложения станции.
# Обучающий и отборочный наборы можно подменить: так проверяется настройка на
# одном годе, которого сеть-прогнозист не видела.
TR="$DIR/krsk_train.parquet"; VA="$DIR/krsk_val.parquet"
train_and_eval() {
  local tag=$1; shift
  local exp="experiments/neural_postproc_krsk${tag}"
  if [[ -f "$exp/best_model.pth" ]]; then
    log "--- $exp: веса уже есть, обучение пропускаю"
  else
    log "--- $exp: обучение, эпох $EPOCHS, обучающий $(basename "$TR")"
    python -u scripts/postproc/train_neural_postproc_v3.py \
        --train-parquet "$TR" \
        --val-parquet   "$VA" \
        --out-dir       "$exp" \
        --epochs "$EPOCHS" --batch-size 4096 --station-emb-dim 32 \
        --hidden 192,192,128 "$@" 2>&1 \
        | grep -E "^\[(cfg|model|dataset|ep )|^Done:"
    [[ -f "$exp/best_model.pth" ]] || { log "    весов нет — пропускаю оценку"; return 1; }
  fi
  log "    проверка на 2020"
  python -u scripts/postproc/eval_per_lead_v2.py \
      --val-parquet "$DIR/krsk_test.parquet" --ckpt "$exp/best_model.pth" \
      --out-dir "$exp/eval_test2020" 2>&1 | grep -E "^\[eval\]|^Overall"
}

train_and_eval ""
train_and_eval "_noobs" --no-obs-features

# Вероятностная настройка: сеть выдаёт не только поправку, но и разброс. Для
# статьи это отдельная величина — прогноз с оценкой собственной надёжности.
train_and_eval "_prob" --probabilistic

# Настройка на одном 2019-м. У табличных поправок выбор лет не менял ничего,
# но у сети втрое меньше данных, и это уже не очевидно. Отбор эпохи по 2018-му:
# он тоже не проверочный год.
if [[ ! -f "$DIR/krsk_train1y.parquet" ]]; then
  log "готовлю набор на одном годе: обучение 2019, отбор 2018"
  python -u - "$DIR" <<'SPLIT1Y'
import sys, pandas as pd
d = sys.argv[1]
df = pd.concat([pd.read_parquet(f"{d}/krsk_train.parquet"),
                pd.read_parquet(f"{d}/krsk_val.parquet")], ignore_index=True)
y = pd.to_datetime(df["valid_time_utc"]).dt.year
for name, years in (("train1y", [2019]), ("val1y", [2018])):
    part = df[y.isin(years)]
    part.to_parquet(f"{d}/krsk_{name}.parquet", index=False)
    print(f"  {name}: {len(part):,} строк", flush=True)
SPLIT1Y
fi
if [[ -f "$DIR/krsk_train1y.parquet" ]]; then
  TR="$DIR/krsk_train1y.parquet"; VA="$DIR/krsk_val1y.parquet"
  train_and_eval "_1y"
  TR="$DIR/krsk_train.parquet"; VA="$DIR/krsk_val.parquet"
fi

# Больше данных. Один год дал 2,356 против 2,297 на трёх — значит данные ещё
# не насытились. Берём четыре: 2016 — октябрь 2019 на обучение, ноябрь-декабрь
# 2019 на отбор эпохи. Отбор отделён по времени, а не случайной выборкой:
# соседние по времени строки почти дубликаты, и случайный отбор их бы перемешал.
if [[ ! -f "$DIR/krsk_train4y.parquet" ]]; then
  log "готовлю набор на четырёх годах"
  python -u - "$DIR" <<'SPLIT4Y'
import sys, pandas as pd
d = sys.argv[1]
df = pd.concat([pd.read_parquet(f"{d}/krsk_train.parquet"),
                pd.read_parquet(f"{d}/krsk_val.parquet")], ignore_index=True)
t = pd.to_datetime(df["valid_time_utc"])
cut = pd.Timestamp("2019-11-01")
for name, sel in (("train4y", t < cut), ("val4y", t >= cut)):
    part = df[sel]
    part.to_parquet(f"{d}/krsk_{name}.parquet", index=False)
    print(f"  {name}: {len(part):,} строк", flush=True)
SPLIT4Y
fi
if [[ -f "$DIR/krsk_train4y.parquet" ]]; then
  TR="$DIR/krsk_train4y.parquet"; VA="$DIR/krsk_val4y.parquet"
  train_and_eval "_4y"
  TR="$DIR/krsk_train.parquet"; VA="$DIR/krsk_val.parquet"
fi

# Сильнее прижать переобучение. Проверочная ошибка встаёт на шестой эпохе, а
# обучающая падает до сороковой — значит модель запоминает, а не обобщает.
# Ёмкость наращивать бессмысленно, нужна регуляризация; берём две ступени,
# чтобы увидеть направление, а не гадать.
train_and_eval "_reg"  --dropout 0.25 --weight-decay 1e-3
train_and_eval "_reg2" --dropout 0.40 --weight-decay 3e-3

# Явная разность высот станции и рельефа модели (см. dataset.py). Разбор по
# станциям показал, где живёт поправка: связь выигрыша с сырой ошибкой 0,91, с
# модулем смещения 0,85, с высотой 0,72, а у 26 станций из 71 выигрыша нет
# вовсе. Значит правится несоответствие площадки ячейке сетки, и величину,
# которая его задаёт, стоит дать модели прямо, а не двумя слагаемыми по разным
# осям. Признаки добавляются сами, поэтому настройка та же, что у основной.
train_and_eval "_dz"

# Потолок этого семейства моделей. Обучаем и проверяем на одном 2020 году: это
# НЕ результат, а верхняя оценка того, сколько в остатке вообще есть выучиваемой
# структуры. Если она близка к тому, что уже достигнуто, дальше вкладываться в
# признаки и ёмкость незачем — остаток это шум площадки, и следующий шаг должен
# быть другим по существу.
TR="$DIR/krsk_test.parquet"; VA="$DIR/krsk_val.parquet"
train_and_eval "_ceil"
TR="$DIR/krsk_train.parquet"; VA="$DIR/krsk_val.parquet"

# Разбор по станциям у основной модели: где поправка работает, а где нет.
log "--- разбор по станциям (основная модель)"
python -u scripts/postproc/eval_per_station_v2.py \
    --val-parquet "$DIR/krsk_test.parquet" \
    --ckpt experiments/neural_postproc_krsk/best_model.pth \
    --out-dir experiments/neural_postproc_krsk/eval_stations \
    --bbox 50 60 83 98 --label krsk 2>&1 | tail -25

log "=== ALL DONE ==="
