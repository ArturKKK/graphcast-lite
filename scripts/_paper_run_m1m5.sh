#!/usr/bin/env bash
# M1: усвоение на длинных горизонтах (AR-28 = +168 ч) — закрывает разрыв
#     «мотивируем коллапсом на 4-е сутки, а показываем только +24 ч».
# M5: честный протокол ОИ — подбор (L, sigma_o) на VAL, отчёт на TEST, 3 сида наблюдений,
#     плюс нуджинг для сравнения. Везде --save-sample-metrics для бутстреп-ДИ.
# Запуск: nohup setsid bash scripts/_paper_run_m1m5.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/m1m5_master.log
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets/multires_krsk_19f_merge
OUT=/workdir/paper_results
ROI="50 60 83 98"
FLAG=multires_merge_freeze6_v2      # флагман (residual → без --no-residual)
DIPL=multires_nores_freeze6         # дипломная (no-residual)

mkdir -p "$OUT"
MASTER="$OUT/m1m5_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# run <tag> <exp> <split> <ar> <nmax> <extra...>
run() {
  local tag="$1" exp="$2" split="$3" ar="$4" nmax="$5"; shift 5
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  local cmd="python -u scripts/predict.py experiments/$exp --data-dir $DATA --split $split --ar-steps $ar --max-samples $nmax --per-channel --no-save --region $ROI --save-sample-metrics $npz $*"
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag: $tag | started: $(date -Iseconds) | host: $(hostname)"
    echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
    echo "# dataset: $DATA"
    echo "#   info: $(tr -d '\n ' < "$DATA/dataset_info.json" | cut -c1-260)"
    echo "# experiment: experiments/$exp (config md5 $(md5sum "experiments/$exp/config.json"|cut -d' ' -f1))"
    for _ck in "experiments/$exp"/best_model.pth; do
      [[ -f "$_ck" ]] && echo "#   ckpt: $(md5sum "$_ck"|cut -d' ' -f1)  $_ck"
    done
    echo "# COMMAND:"; echo "#   $cmd"
    echo "##############################################################"; echo
  } > "$lf"
  log "START $tag"
  eval "$cmd" >> "$lf" 2>&1
  local rc=$?
  echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$lf"
  local sk=$(grep -oE "skill=[-0-9.]+%" "$lf" | tail -1)
  local t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-70)
  log "DONE  $tag rc=$rc | $sk | $t2"
}

log "=== M1+M5 START (commit $GIT_COMMIT) ==="

# ───────────────────────── M1: длинные горизонты (AR-28) ─────────────────────────
# 200 сэмплов: AR-28 требует 28 шагов ground truth, прогоны тяжёлые
run m1_noda_ar28        "$FLAG" test_only 28 200
run m1_oi10_ar28        "$FLAG" test_only 28 200 --assim-method oi --obs-sparsity 0.1  --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5
run m1_oi10_first4_ar28 "$FLAG" test_only 28 200 --assim-method oi --obs-sparsity 0.1  --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5 --oi-first-k 4
run m1_oi1_ar28         "$FLAG" test_only 28 200 --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 --oi-corr-len 200000 --oi-sigma-o 0.5
# преемственность с дипломной моделью
run m1_dipl_noda_ar28   "$DIPL" test_only 28 200 --no-residual
run m1_dipl_oi10_ar28   "$DIPL" test_only 28 200 --no-residual --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5

# ───────────────── M5: подбор гиперпараметров ОИ на VAL (не на тесте!) ─────────────────
for L in 50000 100000 150000 200000; do
  for S in 0.3 0.5 1.0; do
    run "m5_val_oi_L${L}_s${S}" "$FLAG" val 4 200 \
        --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 \
        --oi-corr-len "$L" --oi-sigma-o "$S"
  done
done
log "=== M5 VAL SWEEP DONE — лучшие (L,sigma) выбираются из логов выше ==="

# ───────── M5: TEST — baseline, нуджинг, ОИ 1%; лучшая ОИ 10% с 3 сидами ─────────
run m5_test_baseline "$FLAG" test_only 4 2000
for A in 0.5 0.7 0.9; do
  run "m5_test_nudge_a${A}" "$FLAG" test_only 4 2000 \
      --assim-method nudging --nudging-mode sequential --nudging-alpha "$A" \
      --obs-sparsity 0.1 --obs-roi-only --obs-seed 42
done
run m5_test_oi1 "$FLAG" test_only 4 2000 \
    --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 \
    --oi-corr-len 200000 --oi-sigma-o 0.5
# ОИ 10% при (L=100 км, sigma_o=0.5) — конфигурация диплома; 3 сида сети наблюдений
for SEED in 42 43 44; do
  run "m5_test_oi10_seed${SEED}" "$FLAG" test_only 4 2000 \
      --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed "$SEED" \
      --oi-corr-len 100000 --oi-sigma-o 0.5
done

log "=== ALL DONE ==="
grep -E "DONE " "$MASTER" | tail -30
