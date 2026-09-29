#!/usr/bin/env bash
# Сгущение меша над регионом: опыты и контроль от итоговой модели (28–29.09.2026).
#
#   ref — итоговая модель (dec_long_encres) + меш над регионом сгущён до ~55 км
#         (refine_region в конфиге, src/mesh/create_mesh.py). Процессор видит
#         масштабы 100–220 км, на которых сидит основное отставание от
#         GraphCast. 8 эпох.
#   ctl — та же модель и те же 8 эпох без сгущения. Без контроля не отличить
#         выигрыш сгущения от выигрыша лишних эпох (long дала ~1 % сама по себе).
#   ref2 — сгущение дважды, до ~27 км, почти шаг вставки 0,25° (29.09.2026).
#         Старт и 8 эпох те же, что у ref, так что контролем служит сам ref.
#   ref_s43 — повтор ref с другим начальным значением генератора (порядок
#         примеров). Разброс между повторами показывает, какие разности
#         итоговой модели осмысленны; до сих пор его знали только по одной
#         паре прогонов старой модели (п. 5.1 статьи).
#
# Каждый вариант, кроме обучения и оценки на тесте (t_mesh_*), считает ту же
# оценку на проверочной выборке (v_mesh_*, --split val). Выбор между вариантами
# делается по ней, чтобы не подбирать итоговую модель по тестовой выборке.
# Если обучение уже закончено (ref и ctl 29.09), второй запуск только
# досчитывает то, чего нет: обучение и тест пропускаются.
#
# Очередь. Скрипт ждёт пересчёт статьи (_vm_batch_fin.sh) и другие варианты
# на этой машине, поэтому можно запускать подряд, не дожидаясь конца.
#
# Стартовое состояние берётся из experiments/multires_krsk_33f_chw_dec_long_encres
# (есть только там, где учили эту модель), а если его нет — из
# checkpoints/dec_long_encres.pth в репозитории: его туда кладёт
#     bash scripts/_vm_batch_mesh.sh share     (на машине с моделью)
# после чего  GIT_ASKPASS= git push origin main-arthur.
#
# Запуск:  bash scripts/_vm_batch_mesh.sh ref     (на одной машине)
#          bash scripts/_vm_batch_mesh.sh ctl     (на другой)
#          bash scripts/_vm_batch_mesh.sh ref2
#          bash scripts/_vm_batch_mesh.sh ref_s43
# Лог:     /workdir/paper_results/improve_mesh_<вариант>_master.log
set -uo pipefail
V=${1:-}
[[ "$V" =~ ^(ref|ref2|ref_s43|ctl|share)$ ]] \
  || { echo "вариант: ref, ref2, ref_s43, ctl или share"; exit 1; }

REPO=/workdir/graphcast-lite
BASE_EXP=multires_krsk_33f_chw_dec_long_encres
SHARED=checkpoints/dec_long_encres.pth

# ---------- share: выложить стартовое состояние в репозиторий ----------
if [[ "$V" == "share" ]]; then
  cd "$REPO" || exit 1
  PY=/data/venvs/graphcast/bin/python
  [[ -x "$PY" ]] || PY=python3
  mkdir -p checkpoints
  "$PY" - "experiments/$BASE_EXP/checkpoint.pth" "$SHARED" <<'PY' || exit 1
import sys, pathlib, torch
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print(f"нет {src} — эта машина модель не учила"); raise SystemExit(1)
ck = torch.load(src, map_location="cpu")
sd = ck.get("model_state_dict", ck)
# буфер признаков рёбер процессора — геометрия, её пересчитывает конструктор
sd = {k: v for k, v in sd.items() if not k.startswith("_processing_edge_features")}
torch.save(sd, dst)
print(f"эпоха {ck.get('epoch', '?')} -> {dst} ({pathlib.Path(dst).stat().st_size >> 20} МБ)")
PY
  git add -f "$SHARED"
  git -c user.email="${GIT_AUTHOR_EMAIL:-artur.tabakov1@gmail.com}" \
      -c user.name="${GIT_AUTHOR_NAME:-Artur Tabakov}" \
      commit -qm "стартовое состояние dec_long_encres для опытов с мешем" && \
    echo "закоммичено. осталось:  GIT_ASKPASS= git push origin main-arthur"
  exit 0
fi

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/improve_mesh_${V}_master.log"
  exit 0
fi

VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
ROI="50 60 83 98"
CLIM=$REPO/docs/paper/runs/clim_wb2_nodes.npz
EXP=multires_krsk_33f_chw_mesh_${V}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/improve_mesh_${V}_master.log"
exec 9>"$OUT/.improve_mesh_${V}.lock"
flock -n 9 || { echo "[$(date '+%d.%m %H:%M:%S')] уже идёт" >> "$MASTER"; exit 0; }
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== МЕШ: $V ($(git rev-parse --short HEAD)) ==="

# ---------- очередь: пересчёт статьи и другие варианты на этой машине ----------
for lk in "$OUT"/.fin_*.lock; do
  [[ -e "$lk" ]] || continue
  exec 8>>"$lk"; flock -n 8 || { log "жду $(basename "$lk")"; flock 8; }; exec 8>&-
done
exec 7>>"$OUT/.gpu_queue.lock"
flock -n 7 || { log "жду другой вариант на этой машине"; flock 7; log "дождался"; }

BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# ---------- окружение и датасет ----------
if [[ ! -x "$VENV/bin/python" || ! -f "$D33/data_extra.npy" ]]; then
  log "нет окружения или датасета — восстанавливаю (~40 мин), лог /workdir/paper_logs/restore33f.log"
  DAEMONIZED=1 FREE=1 bash scripts/_vm_restore33f.sh
  log "восстановление rc=$?"
fi
[[ -f "$D33/data_extra.npy" ]] || { log "датасет так и не собрался — стоп"; exit 1; }
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ---------- стартовое состояние ----------
START=$HEAVY/${BASE_EXP}_start.pth
if [[ ! -f "$START" ]]; then
  if [[ -f "experiments/$BASE_EXP/checkpoint.pth" ]]; then
    python - "experiments/$BASE_EXP/checkpoint.pth" "$START" <<'PY' || { log "не извлёк состояние — стоп"; exit 1; }
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
torch.save(ck.get("model_state_dict", ck), sys.argv[2])
print(f"[prep] эпоха {ck.get('epoch', '?')} -> {sys.argv[2]}")
PY
  elif [[ -f "$SHARED" ]]; then
    cp -p "$SHARED" "$START" && log "стартовое состояние из репозитория: $SHARED"
  else
    log "нет стартового состояния: ни experiments/$BASE_EXP, ни $SHARED."
    log "на машине с моделью: bash scripts/_vm_batch_mesh.sh share, затем push; здесь pull — стоп"
    exit 1
  fi
fi

# ---------- конфиг: итоговая архитектура (+ сгущение для ref) ----------
mkdir -p "experiments/$EXP"
python - experiments/multires_krsk_33f_chw/config.json "experiments/$EXP/config.json" "$V" <<'PY'
import json, sys
src, dst, v = sys.argv[1:4]
c = json.load(open(src))
c["pipeline"]["encoder"]["gcn"].update(edge_refine=True, edge_feature_dim=4)
c["pipeline"]["decoder"]["gcn"] = {
    "layer_type": "interaction_net_decoder", "hidden_dims": [128],
    "output_dim": c["pipeline"]["decoder"]["gcn"]["output_dim"],
    "activation": "swish", "edge_feature_dim": 4, "use_layer_norm": True}
if v in ("ref", "ref2", "ref_s43"):
    c["graph"]["refine_region"] = [50.0, 60.0, 83.0, 98.0]
    c["graph"]["refine_buffer_deg"] = 2.0
    c["graph"]["refine_steps"] = 2 if v == "ref2" else 1
if v == "ref_s43":
    c["random_seed"] = 43
if v == "ref2":
    # 29.09: без пересчёта активаций двойное сгущение упало по памяти (79 из 80 ГБ)
    c["pipeline"]["processor"]["gcn"]["grad_checkpoint"] = True
c["num_epochs"] = 8
c["freeze_processor_epochs"] = 0     # одинаково для ref и ctl; процессору надо учиться новым рёбрам
c["finetune_processor_lr_factor"] = 1.0
c["early_stopping_patience"] = 100
c["_comment"] = f"mesh_{v}: от dec_long_encres, см. scripts/_vm_batch_mesh.sh"
json.dump(c, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"[prep] {dst}: сгущение {c['graph'].get('refine_region')} ×{c['graph'].get('refine_steps', 0)}, эпох {c['num_epochs']}")
PY
python - "experiments/$EXP/config.json" <<'PY' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
ExperimentConfig(**json.load(open(sys.argv[1])))
PY

# ---------- обучение ----------
if grep -q "Training finished" "experiments/$EXP/training_log.txt" 2>/dev/null; then
  log "обучение уже закончено — пропускаю"
else
  RESUME=""
  [[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "нашёлся чекпойнт — продолжаю"; }
  log "START обучение $EXP $RESUME"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u -m src.main "experiments/$EXP" --pretrained "$START" $RESUME \
      >> "$OUT/improve_mesh_${V}_train.log" 2>&1
  log "DONE  обучение rc=$?"
  tail -12 "experiments/$EXP/training_log.txt" 2>/dev/null
fi

# ---------- оценка ----------
FIN=$HEAVY/${EXP}_last.pth
[[ -f "$FIN" ]] || python - "experiments/$EXP/checkpoint.pth" "$FIN" <<'PY' || { log "нет чекпойнта — оценки не будет"; exit 1; }
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
torch.save(ck.get("model_state_dict", ck), sys.argv[2])
PY
run() {   # run <тег> <split> [доп. ключи]
  local tag="$1" split="$2"; shift 2
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag (уже посчитан)"; return 0; }
  # поля ошибок (60 МБ) нужны только на тесте: по ним строятся рисунки
  local err=()
  [[ "$split" == "test_only" ]] && err=(--save-region-errors "$OUT/${tag}_errors.npz")
  log "START $tag"
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33" \
      --split "$split" --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $ROI --lat-weight --ckpt "$FIN" \
      --save-sample-metrics "$npz" "${err[@]}" "$@" \
      > "$lf" 2>&1
  local rc=$? t2
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $t2"
}
run t_mesh_${V}_roi test_only --climatology "$CLIM"
run v_mesh_${V}_roi val --climatology "$CLIM"

log "=== ВСЁ ==="
grep -E "DONE  " "$MASTER" | tail -4
