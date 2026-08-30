#!/usr/bin/env bash
# Признаки рельефа: считаем описатели вокруг станций и проверяем, помогают ли.
#
# Проверка устроена так, чтобы её нельзя было принять за выигрыш по случайности.
# Разброс по начальному состоянию у нас 0,004-0,005 °C, а признаки окрестности
# дали 0,020 — вчетверо выше шума. Поэтому рельеф гоняется ТРЕМЯ жребиями, и
# сравнивается с тремя же жребиями без рельефа на том же корпусе. Один прогон
# ничего не доказал бы: разница в 0,003 °C неотличима от перетасовки весов.
#
# Признаки рельефа СТАТИЧНЫ, корпус пересобирать не нужно — таблица на 71 строку
# приклеивается к готовому корпусу по номеру станции за минуту.
#
# Запуск: bash scripts/postproc/_run_terrain_krsk.sh [узлов окрестности] [эпох]
# Лог:    /workdir/paper_results/terrain_krsk_master.log
set -uo pipefail
NB=${1:-12}
EPOCHS=${2:-20}
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/terrain_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
mkdir -p "$OUT"; exec >>"$OUT/terrain_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.terrain_krsk.lock"
flock -n 9 || { log "уже работает другой раннер — стоп"; exit 1; }
log "=== ПРИЗНАКИ РЕЛЬЕФА (окрестность $NB узлов) ==="
BUSY=$(pgrep -af "^python.*(src\.main|build_corpus\.py|train_neural)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# Окружение живёт на /data и стирается вместе с ним при перезапуске виртуалки.
if [[ ! -x "$VENV/bin/python" ]]; then
  log "окружения нет (стёрлось с /data) — восстанавливаю, это несколько минут"
  VENV_ONLY=1 bash scripts/_paper_setup_vm.sh >>"$OUT/venv_restore.log" 2>&1
  log "восстановление rc=$? (подробности в venv_restore.log)"
fi
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
if ! python -c "import pyarrow" 2>/dev/null; then
  log "нет pyarrow (корпус в parquet без него не прочитать) — ставлю"
  pip install -q pyarrow \
      --extra-index-url https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple \
      >>"$OUT/venv_restore.log" 2>&1
  python -c "import pyarrow" 2>/dev/null \
    || { log "pyarrow не поставился — см. venv_restore.log"; exit 1; }
fi
export PYTHONPATH="$REPO"
DIR=/data/postproc

# 1. Листы матрицы высот. Без них считать нечего, и лучше сказать это сразу.
#    Каталог ищем в нескольких местах: наборы данных на этой виртуалке лежат
#    в /data/datasets, а не в /data, и на этом уже споткнулись 30.08.2026.
#    Можно задать явно: DEM_DIR=/куда/угодно bash scripts/postproc/_run_terrain_krsk.sh
DEM=""
CANDIDATES=("${DEM_DIR:-}" /data/datasets/dem /data/dem "$REPO/data/dem"
            /workdir/dem /workdir/graphcast-lite/data/dem)
for d in "${CANDIDATES[@]}"; do
  [[ -z "$d" ]] && continue
  if [[ -d "$d" ]] && compgen -G "$d/*.hgt*" >/dev/null 2>&1; then DEM="$d"; break; fi
done
if [[ -z "$DEM" ]]; then
  log "нет листов высот. Искал здесь:"
  for d in "${CANDIDATES[@]}"; do
    [[ -z "$d" ]] && continue
    if [[ -d "$d" ]]; then log "    $d — каталог есть, но файлов *.hgt* в нём нет"
    else log "    $d — нет такого каталога"; fi
  done
  log "  положи листы в любой из них или задай DEM_DIR=/путь (см. list_dem_tiles.py)"
  exit 1
fi
NTILES=$(compgen -G "$DEM/*.hgt*" | wc -l)
log "листы высот: $DEM ($NTILES шт., $(du -sh "$DEM" | cut -f1))"
if [[ "$NTILES" -lt 100 ]]; then
  log "  ВНИМАНИЕ: листов меньше сотни, а для 71 станции нужно 106."
  log "  Станции без листов останутся без рельефа — смотри строку [рельеф] ниже."
fi

# 2. Исходный корпус с признаками окрестности.
#    /data стирается при перезапуске виртуалки, а корпус жил именно там —
#    30.08.2026 он так и пропал. Поэтому ищем в нескольких местах, а не в одном,
#    и при неудаче печатаем весь список, чтобы было видно, чего не хватает.
#    Можно задать явно: CORPUS=/путь/к.parquet bash scripts/postproc/_run_terrain_krsk.sh
LAGS=""
CORPUS_CANDIDATES=("${CORPUS:-}"
  "$DIR/corpus_krsk_lags2_nb${NB}.parquet"
  "$REPO/data/postproc/corpus_krsk_lags2_nb${NB}.parquet"
  "/data/datasets/postproc/corpus_krsk_lags2_nb${NB}.parquet")
for c in "${CORPUS_CANDIDATES[@]}"; do
  [[ -n "$c" && -f "$c" ]] && { LAGS="$c"; break; }
done

# Готового набора нет, но может уцелеть СЫРОЙ корпус. Признаки-наблюдения
# считаются из него за минуты — это пересчёт по таблице, без развёртки модели.
# Восстанавливать самим стоит: 30.08.2026 пропажа lags2 выглядела как «всё
# заново», хотя пересобрать надо было только надстройку.
if [[ -z "$LAGS" ]]; then
  RAW=""
  for r in "$DIR/corpus_krsk_2016_2020_nb${NB}.parquet" \
           "$REPO/data/postproc/corpus_krsk_2016_2020_nb${NB}.parquet" \
           "/data/datasets/postproc/corpus_krsk_2016_2020_nb${NB}.parquet"; do
    [[ -f "$r" ]] && { RAW="$r"; break; }
  done
  if [[ -n "$RAW" ]]; then
    LAGS="$(dirname "$RAW")/corpus_krsk_lags2_nb${NB}.parquet"
    log "готового набора нет, но есть сырой корпус: $RAW"
    log "  считаю признаки-наблюдения (минуты, развёртки модели не будет)"
    python -u scripts/postproc/add_obs_lags.py --in "$RAW" --out "$LAGS" \
        --clim-years 2016 2017 2018 2>&1 | tail -4
    [[ -f "$LAGS" ]] || { log "признаки не собрались — стоп"; exit 1; }
  fi
fi
if [[ -z "$LAGS" ]]; then
  log "нет корпуса. Искал здесь:"
  for c in "${CORPUS_CANDIDATES[@]}"; do [[ -n "$c" ]] && log "    $c"; done
  log "  ...а также сырой corpus_krsk_2016_2020_nb${NB}.parquet в тех же местах"
  log "  (из сырого признаки-наблюдения досчитываются за минуты)."
  log "  Корпус лежал в /data/postproc, а /data стирается при перезапуске"
  log "  виртуалки. Найти уцелевшую копию:"
  log "    find /data /workdir /root -maxdepth 6 -name 'corpus_krsk*' 2>/dev/null"
  log "  Если копии нет — пересобрать: bash scripts/postproc/_run_night_krsk.sh $NB"
  log "  (это несколько часов: развёртка модели по всем срокам заново)"
  exit 1
fi
log "корпус: $LAGS ($(du -h "$LAGS" | cut -f1))"

# Корпус пережил перезапуск только если лежит вне /data. Копию класть некуда:
# в /workdir квота 8 ГБ. Но сказать об этом стоит — чтобы пропажа не удивляла.
[[ "$LAGS" == /data/postproc/* ]] && \
  log "  ВНИМАНИЕ: корпус в /data/postproc — он не переживёт перезапуск виртуалки"

# 3. Приклейка описателей рельефа.
TERR="$DIR/corpus_krsk_lags2_nb${NB}_terr.parquet"
if [[ ! -f "$TERR" ]]; then
  log "шаг 1: считаю описатели рельефа и клею к корпусу"
  python -u scripts/postproc/add_terrain.py --corpus "$LAGS" --out "$TERR" \
      --stations data/krsk_postproc_stations.json --dem-dir "$DEM" \
      --terrain-json "$DIR/terrain_krsk.json" 2>&1 | tail -20
  [[ -f "$TERR" ]] || { log "приклейка не удалась — стоп"; exit 1; }
fi
log "корпус с рельефом: $(du -h "$TERR" | cut -f1)"

# 4. Деление по годам. Годы те же, что у всех прежних опытов, иначе сравнение
#    перестанет быть сравнением.
if [[ ! -f "$DIR/terr${NB}_test.parquet" ]]; then
  log "шаг 2: деление по годам (обучение 2016-2018, отбор 2019, проверка 2020)"
  python -u scripts/postproc/split_corpus.py --in "$TERR" --out-dir "$DIR" \
      --prefix terr${NB} train=2016,2017,2018 val=2019 test=2020
  [[ -f "$DIR/terr${NB}_test.parquet" ]] || { log "деление не удалось — стоп"; exit 1; }
fi

# 5. Три жребия. Меньше трёх не даст судить о значимости.
run_seed() {
  local seed=$1
  local exp="experiments/neural_postproc_krsk_terr${NB}_s${seed}"
  if [[ ! -f "$exp/best_model.pth" ]]; then
    log "--- жребий $seed: обучение, эпох $EPOCHS"
    python -u scripts/postproc/train_neural_postproc_v3.py \
        --train-parquet "$DIR/terr${NB}_train.parquet" \
        --val-parquet   "$DIR/terr${NB}_val.parquet" \
        --out-dir "$exp" --epochs "$EPOCHS" --batch-size 4096 \
        --station-emb-dim 32 --hidden 192,192,128 --seed "$seed" 2>&1 \
        | grep --line-buffered -E "^\[(cfg|model|dataset|ep )|^Done:"
  else
    log "--- жребий $seed: веса уже есть, обучение пропускаю"
  fi
  [[ -f "$exp/best_model.pth" ]] || { log "    весов нет — пропускаю"; return 1; }
  log "    проверка на 2020"
  python -u scripts/postproc/eval_per_lead_v2.py \
      --val-parquet "$DIR/terr${NB}_test.parquet" --ckpt "$exp/best_model.pth" \
      --out-dir "$exp/eval_test2020" 2>&1 | grep --line-buffered -E "^\[eval\]|^Overall"
  python -u scripts/postproc/record_run.py \
      --eval-json "$exp/eval_test2020/eval_per_lead_v2.json" \
      --name "$(basename "$exp")" --note "рельеф, окрестность $NB, жребий $seed" 2>&1 | tail -2
}
for s in 42 43 44; do run_seed "$s"; done

# 6. Итог одной строкой: сравнивать надо с теми же тремя жребиями без рельефа.
log "шаг 3: сводка по трём жребиям"
NB=$NB python -u - <<'PY' 2>&1 | tail -20
import json
import os
from pathlib import Path

nb = os.environ["NB"]


def rmse(name):
    f = Path("experiments") / name / "eval_test2020" / "eval_per_lead_v2.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["overall"]["pp_rmse_t2m"]


with_t, without = [], []
for s in (42, 43, 44):
    a = rmse(f"neural_postproc_krsk_terr{nb}_s{s}")
    # Прогон без рельефа для жребия 42 ночной раннер сохранял без суффикса.
    b = rmse(f"neural_postproc_krsk_nb{nb}_s{s}")
    if b is None and s == 42:
        b = rmse(f"neural_postproc_krsk_nb{nb}")
    if a is not None:
        with_t.append(a)
    if b is not None:
        without.append(b)
print(f"с рельефом:  {['%.4f' % x for x in with_t]}")
print(f"без рельефа: {['%.4f' % x for x in without]}")
if with_t and without:
    m1 = sum(with_t) / len(with_t); m0 = sum(without) / len(without)
    print(f"среднее: {m0:.4f} -> {m1:.4f}, выигрыш {m0 - m1:+.4f} °C")
    # Порог значимости: разброс по жребию 0,004-0,005 °C. Всё, что меньше
    # 0,010, считать шумом и в статью не писать.
    print("ВЫВОД:", "есть" if (m0 - m1) > 0.010 else
          "НЕТ — в пределах шума по жребию (0,004-0,005 °C)")
else:
    print("не с чем сравнивать: нет прогонов без рельефа на том же корпусе")
PY
log "шаг 4: разбор по станциям для лучшего жребия"
BEST=experiments/neural_postproc_krsk_terr${NB}_s42
[[ -f "$BEST/best_model.pth" ]] && python -u scripts/postproc/eval_per_station_v2.py \
    --val-parquet "$DIR/terr${NB}_test.parquet" --ckpt "$BEST/best_model.pth" \
    --out-dir "$BEST/eval_stations" --bbox 50 60 83 98 --label krsk 2>&1 | tail -6
log "=== ALL DONE ==="
log "сравнивать с: сеть без рельефа 2,269 (23,8 %); лучшая таблица 2,498 (16,1 %)"
