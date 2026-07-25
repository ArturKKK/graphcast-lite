#!/usr/bin/env bash
# M2 (+M3): канонические прогоны ядра для статьи — полный test_only, merge-датасет.
# Последовательно (одна GPU): флагман → freeze6 → nofreeze, ROI + внутренняя зона,
# затем M3 (контроль равных эпох ep16 из checkpoint.pth).
# Запуск: nohup setsid bash scripts/_paper_run_m2.sh </dev/null >/dev/null 2>&1 &
# Лог:    /data/paper_results/m2_master.log
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets/multires_krsk_19f_merge
OUT=/data/paper_results
ROI="50 60 83 98"          # ROI Красноярска (как в дипломе)
INNER="55.5 56.5 92 94"    # внутренняя зона (город)
NMAX=2000                  # > размера test_only → берётся весь сплит

mkdir -p "$OUT"
MASTER="$OUT/m2_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "=== M2 START (полный test_only, N<=$NMAX) ==="
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | head -5)

run() {  # run <tag> <exp> <extra-flags...>
  local tag="$1" exp="$2"; shift 2
  local lf="$OUT/${tag}.log"
  local cmd="python -u scripts/predict.py experiments/$exp --data-dir $DATA --split test_only --ar-steps 4 --max-samples $NMAX --per-channel --no-save $*"
  # --- ПРОВЕНАНС: всё нужное, чтобы прогон был воспроизводим и проверяем ---
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag:        $tag"
    echo "# started:    $(date -Iseconds)"
    echo "# host:       $(hostname)"
    echo "# git commit: $GIT_COMMIT"
    [[ -n "$GIT_DIRTY" ]] && echo "# git dirty:  $(echo "$GIT_DIRTY" | tr '\n' ';')"
    echo "# gpu:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "# dataset:    $DATA"
    echo "#   info:     $(tr -d '\n ' < "$DATA/dataset_info.json" 2>/dev/null | cut -c1-300)"
    echo "# experiment: experiments/$exp"
    echo "#   config:   $(md5sum "experiments/$exp/config.json" 2>/dev/null | cut -d' ' -f1)  (md5 config.json)"
    for _ck in "experiments/$exp"/best_model.pth "experiments/$exp"/*.pth; do
      [[ -f "$_ck" ]] && echo "#   ckpt:     $(md5sum "$_ck" | cut -d' ' -f1)  $(du -h "$_ck" | cut -f1)  $_ck"
    done
    # если чекпойнт передан явно через --ckpt
    for _a in "$@"; do
      [[ -f "$_a" && "$_a" == *.pth ]] && echo "#   ckpt(--ckpt): $(md5sum "$_a" | cut -d' ' -f1)  $_a"
    done
    echo "# COMMAND:"
    echo "#   $cmd"
    echo "##############################################################"
    echo
  } > "$lf"
  log "START $tag ($exp) → $lf"
  python -u scripts/predict.py "experiments/$exp" \
      --data-dir "$DATA" --split test_only --ar-steps 4 \
      --max-samples "$NMAX" --per-channel --no-save "$@" >> "$lf" 2>&1
  local rc=$?
  echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$lf"
  local skill=$(grep -oE "skill=[0-9.]+%" "$lf" | tail -1)
  local t2m=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ')
  log "DONE  $tag rc=$rc | $skill | $t2m"
}

# ---------- M2: ROI ----------
run m2_flagship_roi   multires_merge_freeze6_v2                --region $ROI
run m2_freeze6_roi    multires_nores_freeze6    --no-residual  --region $ROI
run m2_nofreeze_roi   multires_nores_nofreeze   --no-residual  --region $ROI

# ---------- M2: внутренняя зона ----------
run m2_flagship_inner multires_merge_freeze6_v2                --region $INNER
run m2_freeze6_inner  multires_nores_freeze6    --no-residual  --region $INNER
run m2_nofreeze_inner multires_nores_nofreeze   --no-residual  --region $INNER

# ---------- M3: контроль равного бюджета (эпоха 16 у обеих моделей) ----------
log "=== M3: распаковка model_state_dict из checkpoint.pth ==="
python - <<'PY'
import torch, pathlib
out = pathlib.Path("/data/paper_results")
for exp in ["multires_nores_freeze6", "multires_nores_nofreeze"]:
    p = pathlib.Path(f"/workdir/graphcast-lite/experiments/{exp}/checkpoint.pth")
    if not p.exists():
        print(f"[M3] SKIP {exp}: нет checkpoint.pth"); continue
    ck = torch.load(p, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    dst = out / f"{exp}_ep16.pth"
    torch.save(sd, dst)
    print(f"[M3] {exp}: epoch={ck.get('epoch','?')} ar={ck.get('ar_steps','?')} keys={len(sd)} → {dst}")
PY

for exp in multires_nores_freeze6 multires_nores_nofreeze; do
  ck="$OUT/${exp}_ep16.pth"
  [[ -f "$ck" ]] || { log "M3 SKIP $exp (нет $ck)"; continue; }
  run "m3_${exp}_ep16" "$exp" --no-residual --region $ROI --ckpt "$ck"
done

log "=== ALL DONE ==="
grep -E "^\[.*\] DONE" "$MASTER" | tail -20
