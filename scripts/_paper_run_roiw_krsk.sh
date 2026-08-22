#!/usr/bin/env bash
# Вес области интереса в функции потерь. Дожиг тем же расписанием, что drop8.
#
# Зачем. Узлов вставки 2501 из 133 279 — 1,9 %. Обучение почти целиком идёт по
# глобальному полю 0,7°, а публикуем мы ошибку по вставке 0,25°. Здесь узлам
# вставки даётся вес W при единице у остальных: доля области в целевой функции
# становится 2501·W / (130 778 + 2501·W) — 16 % при W=10, 37 % при W=30.
#
# Контроль — multires_krsk_33f_drop8 (то же расписание, W=1): агрегат по области
# 73,81 %, t2m 1,31/1,52/1,56/1,62 °C. Разброс по сиду около 0,3 %.
#
# ВАЖНО: ошибку на валидации между разными W сравнивать НЕЛЬЗЯ — она сама
# взвешена по-разному. Сравнивать только агрегат по области из оценки на тесте.
#
# Запуск: bash scripts/_paper_run_roiw_krsk.sh 30   (сам уходит в фон)
set -uo pipefail
W=${1:-30}
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне, вес области $W. лог: /workdir/paper_results/roiw${W}_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f; BASE=multires_krsk_33f_drop8; EXP=multires_krsk_33f_roiw$W
mkdir -p "$OUT"; exec >>"$OUT/roiw${W}_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ВЕС ОБЛАСТИ В ЛОССЕ: W=$W ==="
# Сторож ловит и обучение, и инференс: 22.08.2026 чуть не запустили дожиг
# поверх пятисуточной развёртки на той же карте — прежний сторож знал
# только про src.main и predict.py бы не заметил.
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/roiw${W}_krsk_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
if [[ ! -f "$D33R/data.npy" ]]; then
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  [[ ! -f "$GX/coords.npz" && -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]] && \
    cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
  log "собираю региональный датасет (часы CPU)"
  python -u scripts/build_multires_russia_33f.py --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir "$GX" --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/roiw${W}_krsk_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

# Конфиг — копия drop8 с одним изменённым полем, чтобы сравнение было честным.
mkdir -p "experiments/$EXP"
python -u - "experiments/$BASE/config.json" "experiments/$EXP/config.json" "$W" <<'PYEOF'
import json, sys
src, dst, w = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = json.load(open(src))
c['roi_loss_weight'] = w
for k in ('wandb_name', 'experiment_name', 'name'):
    if isinstance(c.get(k), str):
        c[k] = c[k].replace('drop8', f'roiw{int(w)}')
json.dump(c, open(dst, 'w'), indent=2, ensure_ascii=False)
print(f"[prep] конфиг {dst}: roi_loss_weight={w:g}, эпох {c['num_epochs']}, "
      f"AR={c['initial_ar_steps']}, темп {c['learning_rate']:g} {c['lr_schedule']}")
PYEOF
[[ -f "experiments/$EXP/config.json" ]] || { log "конфиг не собрался — стоп"; exit 1; }

# Проверка схемы до обучения: поле roi_loss_weight новое, и если оно не примется,
# лучше узнать за секунды, чем после суток счёта и двух оценок на мусорных весах.
python -u - "experiments/$EXP/config.json" <<'PYEOF' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
cfg = ExperimentConfig(**json.load(open(sys.argv[1])))
print(f"[prep] схема принята, roi_loss_weight={cfg.roi_loss_weight:g}")
PYEOF

# Стартовые веса те же, что у drop8: последняя эпоха основной модели.
mkdir -p /data/paper_heavy
PRETRAINED=/data/paper_heavy/krsk33f_last_epoch.pth
if [[ ! -f "$PRETRAINED" ]]; then
  python -u - "experiments/$SRC/checkpoint.pth" "$PRETRAINED" <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print("[prep] нет чекпойнта основной модели"); sys.exit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
e = ck.get("epoch")
print("[prep] стартуем с эпохи", (e + 1) if isinstance(e, int) else e)
PYEOF
fi
[[ -f "$PRETRAINED" ]] || { log "нет стартовых весов — стоп"; exit 1; }

RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume --reset-patience"; log "нашёлся чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD 2>/dev/null)"
log "START обучения $EXP (8 эпох, AR=4, темп 7.5e-5 с косинусом до нуля) $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/roiw${W}_krsk_train.log" 2>&1
log "DONE обучение rc=$?"
grep -m1 "roi_loss_weight=" "$OUT/roiw${W}_krsk_train.log"
[[ -f "experiments/$EXP/checkpoint.pth" ]] || { log "обучение не оставило чекпойнта — оценку не запускаю"; exit 1; }

python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_roiw${W}_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch")
    print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF

for TAG in last best; do
  CK=""
  [[ "$TAG" == last && -f /data/paper_heavy/krsk_roiw${W}_last.pth ]] && CK="--ckpt /data/paper_heavy/krsk_roiw${W}_last.pth"
  [[ "$TAG" == best && -f "experiments/$EXP/best_model.pth" ]] && CK="--ckpt experiments/$EXP/best_model.pth"
  log "START оценки по области ($TAG) $CK"
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
      --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 $CK \
      --save-sample-metrics "$OUT/roiw${W}_krsk_roi_${TAG}_samples.npz" >> "$OUT/roiw${W}_krsk_roi_$TAG.log" 2>&1
  RC=$?
  AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/roiw${W}_krsk_roi_$TAG.log" | grep -oE 'skill=[-0-9.]+%')
  log "DONE оценка $TAG rc=$RC | агрегат по области $AGG"
  grep -E '^\s+(t2m|msl)\b' "$OUT/roiw${W}_krsk_roi_$TAG.log" | tail -2
done
log "контроль W=1 (drop8): агрегат 73.81 %, t2m 1.31/1.52/1.56/1.62 °C"
log "=== ALL DONE ==="
