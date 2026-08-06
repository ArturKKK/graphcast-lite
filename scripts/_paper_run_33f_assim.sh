#!/usr/bin/env bash
# Усвоение наблюдений на 33-канальной красноярской модели.
#
# Повторяет всю серию, ранее посчитанную на 19-канальной модели (M5 + M1), но
# на основной модели статьи, чтобы раздел об усвоении и раздел о качестве
# прогноза опирались на ОДНУ конфигурацию. Дополнительно развёртка продлена с
# 7 до 14 суток.
#
# Веса: эпоха 29 (последняя, стадия AR=4), а не «лучшая по val» (эпоха 15).
# Основание — наш же результат: на целевых горизонтах последняя точнее
# (1.66 против 1.71 °C на +24 ч), и отбирать состояние следует по метрике на
# целевом горизонте, а не по одношаговой ошибке.
#
# Модель остаточная (use_residual=true) — флаг --no-residual НЕ ставить.
#
# Порядок: сборка датасета (если его нет) → smoke → свип на val → тест → длинные.
# Запуск: nohup setsid bash scripts/_paper_run_33f_assim.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/a33_master.log
#
# Оценка времени: сборка датасета 2-4 ч CPU; свип на val ~1 ч; тест ~5 ч;
# длинные горизонты ~4 ч. Итого около 10 ч GPU после сборки.
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
MERGE=/data/datasets/multires_krsk_19f_merge
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
REXTRA=/data/datasets/region_krsk_61x41_extra_2010-2020_025deg
EXP=multires_krsk_33f
ROI="50 60 83 98"
INNER="55.5 56.5 92 94"

AR_LONG=56          # 14 суток
N_LONG=200          # сроков для длинной развёртки
N_TEST=2000         # полный тест (фактически 1607)
N_VAL=200           # для свипа гиперпараметров

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/a33_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== 33f ASSIM START (commit $GIT_COMMIT) ==="
log "диск /data: $(df -h /data | tail -1 | awk '{print $4}') свободно"

# ── 0. Датасет: собрать, если стёрт ───────────────────────────────────
if [[ ! -f "$D33/data.npy" ]]; then
  log "датасета $D33 нет — собираю (часы CPU, ~60 ГБ)"
  for d in "$MERGE" "$GEXTRA" "$REXTRA"; do
    [[ -d "$d" ]] || { log "НЕТ ИСХОДНИКА: $d — сборка невозможна"; exit 1; }
  done
  # Оси глобальной сетки сборщик берёт отсюда; их кладёт setup_vm, копируя из
  # основного глобального датасета. Если тот удалён ради места — восстанавливаем
  # оси из merge-сетки (её глобальные узлы образуют ту же решётку 512x256).
  if [[ ! -f "$GEXTRA/coords.npz" ]]; then
    log "в global_extra нет coords.npz — восстанавливаю из merge"
    python -u scripts/fix_global_extra_coords.py --merge-dir "$MERGE" --extra-dir "$GEXTRA" >> "$OUT/a33_build.log" 2>&1 \
      || { log "восстановить оси не удалось — см. a33_build.log"; exit 1; }
  fi
  python -u scripts/build_multires_russia_33f.py \
      --multires-dir "$MERGE" --extra-dir "$GEXTRA" \
      --region-extra-dir "$REXTRA" --out-dir "$D33" \
      >> "$OUT/a33_build.log" 2>&1
  rc=$?
  log "сборка датасета rc=$rc, размер: $(du -sh "$D33" 2>/dev/null | cut -f1)"
  [[ $rc -eq 0 ]] || exit 1
else
  log "датасет на месте: $(du -sh "$D33" | cut -f1)"
fi

# ── 1. Веса последней эпохи ───────────────────────────────────────────
CKPT="$HEAVY/krsk33f_last_epoch.pth"
python - <<PY >> "$MASTER" 2>&1
import torch, pathlib
src = pathlib.Path("$REPO/experiments/$EXP/checkpoint.pth")
dst = pathlib.Path("$CKPT")
if dst.exists():
    print("[prep] веса последней эпохи уже распакованы")
elif src.exists():
    ck = torch.load(src, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    torch.save(sd, dst)
    print(f"[prep] epoch={ck.get('epoch','?')} ar={ck.get('ar_steps','?')} -> {dst}")
else:
    print("[prep] НЕТ checkpoint.pth — будет использован best_model.pth (эпоха 15)")
PY
CK_ARG=""
[[ -f "$CKPT" ]] && CK_ARG="--ckpt $CKPT"

# run <tag> <extra-args...>
run() {
  local tag="$1"; shift
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  local cmd="python -u scripts/predict.py experiments/$EXP --data-dir $D33 $CK_ARG --per-channel --no-save --save-sample-metrics $npz $*"
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag: $tag | started: $(date -Iseconds) | host: $(hostname)"
    echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "# dataset: $D33"
    [[ -n "$CK_ARG" ]] && echo "# ckpt md5: $(md5sum "$CKPT" | cut -d' ' -f1) (эпоха 29)"
    echo "# COMMAND:"; echo "#   $cmd"
    echo "##############################################################"; echo
  } > "$lf"
  log "START $tag"
  eval "$cmd" >> "$lf" 2>&1
  local rc=$?
  local sk=$(grep -oE "skill=[-0-9.]+%" "$lf" | tail -1)
  local t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-70)
  log "DONE  $tag rc=$rc | $sk | $t2"
}

# ── 2. SMOKE ──────────────────────────────────────────────────────────
log "SMOKE: 5 сроков без усвоения (ожидаем t2m 1-2 °C)"
python -u scripts/predict.py experiments/$EXP --data-dir "$D33" $CK_ARG \
  --ar-steps 4 --max-samples 5 --per-channel --no-save --region $ROI 2>&1 \
  | grep -E "^\s+t2m|skill=" | tail -3

# ── 3. Подбор гиперпараметров ОИ на проверочной выборке ───────────────
for L in 50000 100000 150000 200000; do
  for S in 0.3 0.5 1.0; do
    run "a33_val_oi_L${L}_s${S}" --split val --ar-steps 4 --max-samples $N_VAL --region $ROI \
        --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 \
        --oi-corr-len $L --oi-sigma-o $S
  done
done
log "=== свип на val закончен, лучшую пару (L, sigma) выбрать по логам выше ==="

# ── 4. Тестовая выборка, полное окно ──────────────────────────────────
run a33_test_baseline --split test_only --ar-steps 4 --max-samples $N_TEST --region $ROI
run a33_test_baseline_inner --split test_only --ar-steps 4 --max-samples $N_TEST --region $INNER

for A in 0.5 0.7 0.9; do
  run "a33_test_nudge_a${A}" --split test_only --ar-steps 4 --max-samples $N_TEST --region $ROI \
      --assim-method nudging --nudging-mode sequential --nudging-alpha $A \
      --obs-sparsity 0.1 --obs-roi-only --obs-seed 42
done

for SEED in 42 43 44; do
  run "a33_test_oi10_seed${SEED}" --split test_only --ar-steps 4 --max-samples $N_TEST --region $ROI \
      --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed $SEED \
      --oi-corr-len 100000 --oi-sigma-o 0.5
done

run a33_test_oi1 --split test_only --ar-steps 4 --max-samples $N_TEST --region $ROI \
    --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 \
    --oi-corr-len 200000 --oi-sigma-o 0.5

# ── 5. Длинная развёртка: 14 суток ────────────────────────────────────
run a33_long_noda --split test_only --ar-steps $AR_LONG --max-samples $N_LONG --region $ROI

run a33_long_oi10 --split test_only --ar-steps $AR_LONG --max-samples $N_LONG --region $ROI \
    --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 \
    --oi-corr-len 100000 --oi-sigma-o 0.5

run a33_long_oi10_first4 --split test_only --ar-steps $AR_LONG --max-samples $N_LONG --region $ROI \
    --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 \
    --oi-corr-len 100000 --oi-sigma-o 0.5 --oi-first-k 4

run a33_long_oi1 --split test_only --ar-steps $AR_LONG --max-samples $N_LONG --region $ROI \
    --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 \
    --oi-corr-len 200000 --oi-sigma-o 0.5

log "=== ALL DONE ==="
grep -E "DONE " "$MASTER" | tail -25
