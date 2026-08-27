#!/usr/bin/env bash
# Сборка корпуса постобработки на красноярской модели.
#
# Что делает: разворачивает модель авторегрессионно от начальных полей ERA5,
# выбирает значения в точках 71 станции области, сопоставляет с наблюдениями
# ISD-Lite и складывает в parquet. Сроки прогноза 6-120 ч, инициализации в
# 00 и 12 UTC.
#
# Прежний корпус собирался на общероссийской модели, веса которой потеряны;
# красноярская сохранилась и вдобавок точнее после августовской работы.
# 19-канальная часть у неё уже слита в один плоский массив, поэтому сборщику
# передаётся --merged-base вместо пары глобальная/региональная сетка.
#
# Годы делим между двумя виртуалками, потом сливаем:
#   виртуалка 1:  bash scripts/postproc/_run_corpus_krsk.sh 2016 2018
#   виртуалка 2:  bash scripts/postproc/_run_corpus_krsk.sh 2019 2020
#
# Сам уходит в фон. Лог: /workdir/paper_results/corpus_<годы>_master.log
set -uo pipefail
Y0=${1:-2016}; Y1=${2:-2018}; EXP=${3:-}
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
log "=== КОРПУС ПОСТОБРАБОТКИ, годы $Y0-$Y1 ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py|build_corpus\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$DATA/multires_krsk_33f/data.npy" ]]; then
  log "подготовка окружения и датасетов"
  bash scripts/_paper_setup_vm.sh >> "$OUT/corpus_${TAG}_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

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

# Наблюдения: 12 МБ, качаются за двадцать секунд, в репозитории их нет.
if [[ ! -d data/isd_lite_krsk ]] || [[ $(ls data/isd_lite_krsk 2>/dev/null | wc -l) -lt 300 ]]; then
  log "качаю наблюдения ISD-Lite"
  python -u scripts/download_russia_isd_lite.py --stations data/krsk_postproc_stations.json \
      --years 2016-2020 --out-dir data/isd_lite_krsk --workers 12 >> "$OUT/corpus_${TAG}_isd.log" 2>&1
  log "наблюдений: $(ls data/isd_lite_krsk | wc -l) файлов"
fi

mkdir -p data/postproc
log "START сборки (развёртка до 120 ч, инициализации 00 и 12 UTC)"
python -u scripts/postproc/build_corpus.py \
    --experiment-dir "experiments/$EXP" \
    --multires-dir   "$DATA/multires_krsk_33f" \
    --merged-base    "$DATA/multires_krsk_19f_merge" \
    --global-extra   "$DATA/global_512x256_extra_2010-2021_07deg" \
    --regional-extra "$DATA/region_krsk_61x41_extra_2010-2020_025deg" \
    --stations-json  data/krsk_postproc_stations.json \
    --isd-dir        data/isd_lite_krsk \
    --top-stations   71 \
    --years "$Y0" "$Y1" \
    --out-parquet    "data/postproc/corpus_krsk_${TAG}.parquet" \
    >> "$OUT/corpus_${TAG}_build.log" 2>&1
RC=$?
log "DONE сборка rc=$RC"
if [[ -f "data/postproc/corpus_krsk_${TAG}.parquet" ]]; then
  log "корпус: $(du -h "data/postproc/corpus_krsk_${TAG}.parquet" | cut -f1)"
  python -u - "data/postproc/corpus_krsk_${TAG}.parquet" <<'PYEOF'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
print(f"[итог] строк {len(df):,}, столбцов {len(df.columns)}")
print(f"[итог] станций {df['station_usaf'].nunique()}, "
      f"сроков прогноза {sorted(df['lead_h'].unique())}")
print(f"[итог] период {df['valid_time_utc'].min()} — {df['valid_time_utc'].max()}")
PYEOF
else
  log "корпус не собрался — см. corpus_${TAG}_build.log"
fi
log "=== ALL DONE ==="
