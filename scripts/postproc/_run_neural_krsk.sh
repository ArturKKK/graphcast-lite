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
EPOCHS=${1:-40}
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

log "обучение, эпох $EPOCHS"
python -u scripts/postproc/train_neural_postproc_v3.py \
    --train-parquet "$DIR/krsk_train.parquet" \
    --val-parquet   "$DIR/krsk_val.parquet" \
    --out-dir       "$EXP" \
    --epochs "$EPOCHS" --batch-size 4096 --station-emb-dim 32 \
    --hidden 192,192,128 2>&1 | tail -60
RC=${PIPESTATUS[0]}
log "обучение rc=$RC"
CKPT=$(ls -1 "$EXP"/best_model.pth "$EXP"/checkpoint.pth 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { log "весов нет — см. лог выше"; exit 1; }

# Проверка на 2020: тот же год, на котором мерились базовые линии.
log "проверка на 2020"
python -u scripts/postproc/eval_per_lead_v2.py \
    --val-parquet "$DIR/krsk_test.parquet" --ckpt "$CKPT" \
    --out-dir "$EXP/eval_test2020" 2>&1 | tail -40
log "=== ALL DONE ==="
