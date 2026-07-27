#!/usr/bin/env bash
# Сравнение 19 против 33 каналов на ОДНОМ тестовом окне и одной маске узлов.
# Схема обучения 33f специально выровнена с дипломной 19f-линией, поэтому
# различие конфигураций сводится к составу каналов.
#
# Дополнительно проверяется вывод из контрольного эксперимента M3: критерий
# минимума ошибки на проверочной выборке ненадёжен при curriculum по длине
# развёртки, так как val_loss несопоставим между стадиями AR. Поэтому 33f
# оценивается в ДВУХ состояниях: «лучшем по val» (эпоха 15, стадия AR=2)
# и последнем (эпоха 29, стадия AR=4, то есть обученном на целевой длине).
#
# Запуск: nohup setsid bash scripts/_paper_run_33f_eval.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/m33eval_master.log
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
D33=/data/datasets/multires_krsk_33f
D19=/data/datasets/multires_krsk_19f_merge
OUT=/workdir/paper_results
ROI="50 60 83 98"
INNER="55.5 56.5 92 94"
EXP33=multires_krsk_33f

mkdir -p "$OUT"
MASTER="$OUT/m33eval_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# run <tag> <exp> <data> <region> <extra...>
run() {
  local tag="$1" exp="$2" data="$3" reg="$4"; shift 4
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  local cmd="python -u scripts/predict.py experiments/$exp --data-dir $data --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save --region $reg --save-sample-metrics $npz $*"
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag: $tag | started: $(date -Iseconds) | host: $(hostname)"
    echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
    echo "# dataset: $data"
    echo "#   info: $(tr -d '\n ' < "$data/dataset_info.json" 2>/dev/null | cut -c1-260)"
    echo "# experiment: experiments/$exp (config md5 $(md5sum "experiments/$exp/config.json"|cut -d' ' -f1))"
    for _a in "$@"; do
      [[ -f "$_a" && "$_a" == *.pth ]] && echo "#   ckpt(--ckpt): $(md5sum "$_a"|cut -d' ' -f1)  $_a"
    done
    [[ "$*" == *--ckpt* ]] || {
      _b="experiments/$exp/best_model.pth"
      [[ -f "$_b" ]] && echo "#   ckpt: $(md5sum "$_b"|cut -d' ' -f1)  $_b"
    }
    echo "# COMMAND:"; echo "#   $cmd"
    echo "##############################################################"; echo
  } > "$lf"
  log "START $tag"
  eval "$cmd" >> "$lf" 2>&1
  local rc=$?
  echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$lf"
  local sk=$(grep -oE "skill=[-0-9.]+%" "$lf" | tail -1)
  local t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-60)
  log "DONE  $tag rc=$rc | $sk | $t2"
}

log "=== 33f EVAL START (commit $GIT_COMMIT) ==="

# ---- подготовка: распаковать последнее состояние (эпоха 29, стадия AR=4) ----
python - <<'PY'
import torch, pathlib
src = pathlib.Path("/workdir/graphcast-lite/experiments/multires_krsk_33f/checkpoint.pth")
dst = pathlib.Path("/workdir/paper_results/krsk33f_last_epoch.pth")
if src.exists():
    ck = torch.load(src, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    torch.save(sd, dst)
    print(f"[prep] epoch={ck.get('epoch','?')} ar={ck.get('ar_steps','?')} keys={len(sd)} -> {dst}")
else:
    print("[prep] нет checkpoint.pth")
PY

# ---- 33 канала: два состояния ----
run m33_best_roi    "$EXP33" "$D33" "$ROI"                                            # эпоха 15, лучшая по val
run m33_last_roi    "$EXP33" "$D33" "$ROI"   --ckpt "$OUT/krsk33f_last_epoch.pth"     # эпоха 29, стадия AR=4
run m33_best_inner  "$EXP33" "$D33" "$INNER"
run m33_last_inner  "$EXP33" "$D33" "$INNER" --ckpt "$OUT/krsk33f_last_epoch.pth"

# ---- 19 каналов на том же окне (контроль на этой же машине) ----
if [[ -f "experiments/multires_nores_freeze6/best_model.pth" ]]; then
  run m19_freeze6_roi   multires_nores_freeze6 "$D19" "$ROI"   --no-residual
  run m19_freeze6_inner multires_nores_freeze6 "$D19" "$INNER" --no-residual
fi
if [[ -f "experiments/multires_merge_freeze6_v2/best_model.pth" ]]; then
  run m19_flagship_roi  multires_merge_freeze6_v2 "$D19" "$ROI"
fi

# ---- длинный горизонт для 33f: проверка устойчивости ----
run m33_last_ar28 "$EXP33" "$D33" "$ROI" --ar-steps 28 --max-samples 200 --ckpt "$OUT/krsk33f_last_epoch.pth"

log "=== ALL DONE ==="
grep -E "DONE " "$MASTER" | tail -12
