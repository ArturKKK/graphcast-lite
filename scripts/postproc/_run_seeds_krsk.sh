#!/usr/bin/env bash
# Разброс постпроцессора по случайному начальному состоянию.
#
# Зачем. Признаки окрестности станции дали 2,277 против 2,297 °C — 0,9 %. Восемь
# настроек до того легли в полосу 2,289-2,331, то есть 1,8 %, и часть этого
# разброса может быть просто разными начальными весами. Пока не измерено, сколько
# даёт сам жребий, выигрыш в 0,9 % утверждать нельзя.
#
# Считаем по два дополнительных начальных состояния для обеих настроек — с
# окрестностью и без. Если разброс внутри настройки заметно меньше разницы между
# настройками, выигрыш настоящий; если сравним — 0,9 % надо списать на шум.
#
# Запуск: bash scripts/postproc/_run_seeds_krsk.sh
# Лог:    /workdir/paper_results/seeds_krsk_master.log
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/seeds_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
mkdir -p "$OUT"; exec >>"$OUT/seeds_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
exec 9>"$OUT/.seeds_krsk.lock"
flock -n 9 || { log "уже работает — стоп"; exit 1; }
log "=== РАЗБРОС ПО НАЧАЛЬНОМУ СОСТОЯНИЮ ==="
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

find_split() {   # ищет нарезку по префиксу в обоих местах
  local pref=$1
  for d in /data/postproc data/postproc; do
    [[ -f "$d/${pref}_train.parquet" && -f "$d/${pref}_test.parquet" ]] && { echo "$d"; return; }
  done
}

run_seed() {     # префикс нарезки, имя настройки, жребий
  local pref=$1 name=$2 seed=$3
  local d; d=$(find_split "$pref")
  [[ -z "$d" ]] && { log "нет нарезки ${pref}_* — пропускаю"; return 1; }
  local exp="experiments/neural_postproc_${name}_s${seed}"
  if [[ ! -f "$exp/best_model.pth" ]]; then
    log "--- $name, жребий $seed"
    python -u scripts/postproc/train_neural_postproc_v3.py \
        --train-parquet "$d/${pref}_train.parquet" \
        --val-parquet   "$d/${pref}_val.parquet" \
        --out-dir "$exp" --epochs 20 --batch-size 4096 --station-emb-dim 32 \
        --hidden 192,192,128 --seed "$seed" 2>&1 \
        | grep --line-buffered -E "^\[(cfg|model|dataset)|^Done:"
    [[ -f "$exp/best_model.pth" ]] || { log "    весов нет — пропускаю"; return 1; }
  else
    log "--- $name, жребий $seed: веса уже есть"
  fi
  python -u scripts/postproc/eval_per_lead_v2.py \
      --val-parquet "$d/${pref}_test.parquet" --ckpt "$exp/best_model.pth" \
      --out-dir "$exp/eval_test2020" 2>&1 | grep --line-buffered -E "^Overall"
  python -u scripts/postproc/record_run.py \
      --eval-json "$exp/eval_test2020/eval_per_lead_v2.json" \
      --name "$(basename "$exp")" --note "жребий $seed" 2>&1 | tail -2
}

for seed in 43 44; do
  run_seed krsk "krsk"     "$seed"
  run_seed nb6  "krsk_nb6" "$seed"
done
log "=== ALL DONE ==="
log "жребий 42 уже посчитан: без окрестности 2,297 / 1,727; с окрестностью 2,277 / 1,714"
