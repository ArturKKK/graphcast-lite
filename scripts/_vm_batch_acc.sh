#!/usr/bin/env bash
# Этап 1: пересчёт табл. 1 с широтным весом и настоящим ACC.
#
# Долг перед рецензентами. До сих пор узлы суммировались с равными весами, а
# величина под именем ACC была корреляцией по отклонению от СРЕДНЕГО ПО
# ОБЛАСТИ — не аномальной корреляцией. Здесь те же конфигурации считаются с
# --lat-weight и --climatology.
#
# Постфактум это не пересчитать: широтный вес пространственный, а сохранялся
# MSE, уже осреднённый по узлам. Отсюда и повторный инференс.
#
# Запуск:  bash scripts/_vm_batch_acc.sh        (уходит в фон)
# Лог:     /workdir/paper_results/acc_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/acc_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy          # тяжёлое — в /data: в /workdir квота 8 ГБ
D33=/data/datasets/multires_krsk_33f
D19=/data/datasets/multires_krsk_19f_merge
ROI="50 60 83 98"
INNER="55.5 56.5 92 94"
COEF=$HEAVY/clim_coef_krsk.npz   # около 180 МБ, поэтому не в /workdir

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/acc_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")

log "=== ЭТАП 1: широтный вес и ACC против климатологии (commit $GIT_COMMIT) ==="
[[ -f "$D33/data.npy" ]] || { log "FATAL: нет $D33"; exit 1; }

# ---------- 0. состояния моделей ----------
# Из checkpoint.pth достаётся последнее состояние: отбор «лучшего по val»
# ненадёжен для многошагового прогноза (это результат самой статьи).
prep() {
  local exp="$1" dst="$2"
  [[ -f "$dst" ]] && { log "состояние уже готово: $dst"; return 0; }
  python - "$exp" "$dst" <<'PY'
import sys, pathlib, torch
src = pathlib.Path(f"/workdir/graphcast-lite/experiments/{sys.argv[1]}/checkpoint.pth")
if not src.exists():
    print(f"[prep] нет {src}"); raise SystemExit(0)
ck = torch.load(src, map_location="cpu")
sd = ck.get("model_state_dict", ck)
torch.save(sd, sys.argv[2])
print(f"[prep] {sys.argv[1]}: epoch={ck.get('epoch','?')} ar={ck.get('ar_steps','?')} -> {sys.argv[2]}")
PY
}
prep multires_krsk_33f     "$OUT/krsk33f_last_epoch.pth"
prep multires_krsk_33f_chw "$HEAVY/krsk33f_chw_last.pth"

# ---------- 1. коэффициенты климатологии ----------
# Считаются по ОБУЧАЮЩЕЙ части выборки; тестовые сроки в подгонку не входят.
# Только процессор, около 12 минут (замер 15.08.2026).
if [[ ! -f "$COEF" ]]; then
  S=$(ls -1 "$OUT"/m33_last_roi_samples.npz docs/paper/runs/*/m33_last_roi_samples.npz 2>/dev/null | head -1)
  [[ -n "$S" ]] || { log "FATAL: не нашёл m33_last_roi_samples.npz — без него не взять сроки"; exit 2; }
  log "START климатология (сроки из $S)"
  python -u scripts/paper_climatology.py --data-dir "$D33" --samples "$S" \
      --region 50 60 83 98 --out "$OUT/clim_krsk.npz" --out-coef "$COEF" \
      >> "$OUT/acc_clim.log" 2>&1
  [[ -f "$COEF" ]] || { log "FATAL: коэффициенты не собрались, см. $OUT/acc_clim.log"; exit 3; }
  log "DONE климатология: $(du -sh "$COEF" | cut -f1)"
else
  log "коэффициенты уже есть: $COEF"
fi

# ---------- 2. прогоны ----------
run() {
  local tag="$1" exp="$2" data="$3" reg="$4"; shift 4
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  if [[ -f "$npz" ]]; then log "SKIP $tag (уже посчитан)"; return 0; fi
  log "START $tag"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$data" \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $reg --lat-weight --climatology "$COEF" \
      --save-sample-metrics "$npz" "$@" > "$lf" 2>&1
  local rc=$?
  local sk t2 acc
  sk=$(grep -oE 'skill=[0-9.]+%' "$lf" | tail -1)
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  acc=$(grep -oE 'ACC=[0-9.]+' "$lf" | tail -1)
  log "DONE  $tag rc=$rc | $sk | $acc | $t2"
}

CK33="--ckpt $OUT/krsk33f_last_epoch.pth"
CKCHW="--ckpt $HEAVY/krsk33f_chw_last.pth"

# Строки табл. 1 в том же порядке, что в статье.
run w_m33_base_roi    multires_krsk_33f     "$D33" "$ROI"   $CK33
run w_m33_base_inner  multires_krsk_33f     "$D33" "$INNER" $CK33
run w_m33_chw_roi     multires_krsk_33f_chw "$D33" "$ROI"   $CKCHW
run w_m33_chw_inner   multires_krsk_33f_chw "$D33" "$INNER" $CKCHW
if [[ -f experiments/multires_merge_freeze6_v2/best_model.pth && -f "$D19/data.npy" ]]; then
  run w_m19_roi multires_merge_freeze6_v2 "$D19" "$ROI"
else
  log "19-канальной строки не будет: нет модели или набора $D19"
fi

log "=== ВСЁ ==="
grep -E "DONE  w_" "$MASTER" | tail -8
