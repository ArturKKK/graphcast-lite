#!/usr/bin/env bash
# Шум в авторегрессии поверх рабочей точки по весу области.
#
# Зачем. При обучении модель на каждом шаге получает на вход СВОЙ ЖЕ выход, но
# идеальный — тот, который она только что выдала. В работе на вход идёт выход
# с ошибкой, и за четыре шага ошибка накапливается. Приём известный: перед тем
# как подать выход обратно, подмешать гауссов шум — модель учится не полагаться
# на собственную безошибочность. Лосс считается по ЧИСТОМУ выходу, шум добавляется
# только в то, что уходит на следующий шаг (src/train.py:243 и 249).
#
# Почему это уместно именно сейчас. Развёртка на пять суток 22.08 показала, что
# преимущество веса области затухает со сроком: ошибка накапливается быстро.
# Оценка статьи — четыре шага, то есть накопление входит в публикуемое число.
#
# Масштаб. Данные нормированы, собственная ошибка модели за 6 ч около 0,135 в тех
# же единицах. sigma 0,05 — это примерно треть от неё, 0,10 — две трети.
#
# Контроль — multires_krsk_33f_roiw30: агрегат по области 74,85 %,
# t2m 1,23/1,43/1,47/1,53 °C. Отличие ровно в одном поле конфига.
#
# Запуск: bash scripts/_paper_run_noise_krsk.sh [sigma]   (сам уходит в фон)
set -uo pipefail
SIG=${1:-0.05}
TAG=$(echo "$SIG" | tr -d '.')
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне, sigma $SIG. лог: /workdir/paper_results/noise${TAG}_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f; BASE=multires_krsk_33f_drop8; EXP=multires_krsk_33f_noise$TAG
mkdir -p "$OUT"; exec >>"$OUT/noise${TAG}_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ШУМ В АВТОРЕГРЕССИИ sigma=$SIG ПОВЕРХ ВЕСА ОБЛАСТИ W=30 ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/noise${TAG}_krsk_setup.log" 2>&1
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
      --out-dir "$D33R" >> "$OUT/noise${TAG}_krsk_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

mkdir -p "experiments/$EXP"
python -u - "experiments/$BASE/config.json" "experiments/$EXP/config.json" "$SIG" <<'PYEOF'
import json, sys
src, dst, sig = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = json.load(open(src))
c['roi_loss_weight'] = 30.0        # рабочая точка из перебора 21–22.08
c['noise_sigma'] = sig
c['noise_apply_from_ar_step'] = 1  # шум со второго входа, первый вход — настоящие данные
for k in ('wandb_name', 'experiment_name', 'name'):
    if isinstance(c.get(k), str):
        c[k] = c[k].replace('drop8', f'noise{sig}')
json.dump(c, open(dst, 'w'), indent=2, ensure_ascii=False)
print(f"[prep] конфиг {dst}: вес области 30, шум {sig:g}, эпох {c['num_epochs']}, "
      f"AR={c['initial_ar_steps']}, темп {c['learning_rate']:g} {c['lr_schedule']}")
PYEOF
python -u - "experiments/$EXP/config.json" <<'PYEOF' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
cfg = ExperimentConfig(**json.load(open(sys.argv[1])))
print(f"[prep] схема принята: roi_loss_weight={cfg.roi_loss_weight:g}, "
      f"noise_sigma={cfg.noise_sigma:g}")
PYEOF

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
    >> "$OUT/noise${TAG}_krsk_train.log" 2>&1
log "DONE обучение rc=$?"
grep -m1 "input noise injection" "$OUT/noise${TAG}_krsk_train.log" || log "ВНИМАНИЕ: строки про шум в логе нет — проверь, применился ли он"
[[ -f "experiments/$EXP/checkpoint.pth" ]] || { log "обучение не оставило чекпойнта — оценку не запускаю"; exit 1; }

python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_noise${TAG}_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch")
    print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF
log "START оценки по области"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 \
    --ckpt /data/paper_heavy/krsk_noise${TAG}_last.pth \
    --save-sample-metrics "$OUT/noise${TAG}_krsk_roi_samples.npz" >> "$OUT/noise${TAG}_krsk_roi.log" 2>&1
RC=$?
AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/noise${TAG}_krsk_roi.log" | grep -oE 'skill=[-0-9.]+%')
log "DONE оценка rc=$RC | агрегат по области $AGG"
grep -E '^\s+(t2m|msl)\b' "$OUT/noise${TAG}_krsk_roi.log" | tail -2
log "контроль roiw30: агрегат 74.85 %, t2m 1.23/1.43/1.47/1.53 °C"
log "=== ALL DONE ==="
