#!/usr/bin/env bash
# Веса каналов в лоссе поверх веса области. Расписание то же, что у roiw30.
#
# Зачем. Тот же перекос, что мы починили по узлам, есть и по каналам, и он хуже.
# Нормировка делит на разброс поля, а модель предсказывает приращение; разброс
# приращений отличается на порядки, и доля канала в целевой функции получилась
# случайной. Замер на нашей модели: осадки 34 %, влажность на 250 гПа 11 %,
# а приземная температура — 0,26 %. Пять каналов из таблиц статьи вместе дают
# 8,9 %. Осадки при этом в статью не идут вовсе и предсказываются плохо, то есть
# треть градиента уходит в шум.
#
# Веса считаются из самих данных (scripts/channel_loss_weights.py), не из головы:
# w = (σ_мед / σ_канала)^2 с ограничением, чтобы спокойные каналы вроде
# приземного давления не забрали лосс себе.
#
# Контроль — multires_krsk_33f_roiw30: агрегат по области 74,85 %,
# t2m 1,23/1,43/1,47/1,53 °C. Отличие ровно в одном поле конфига.
#
# ВАЖНО про постановку. Агрегат по области считается как среднее квадрата ошибки
# по всем каналам ПОРОВНУ, то есть он и есть исходный равномерный лосс — для него
# равные веса оптимальны по построению. Веса каналов не чинят баг, а переводят
# точность с одних каналов на другие. Здесь мы целимся в те пять, что стоят в
# таблицах статьи: t2m, 10u, 10v, msl, z@500.
#
# Калибровка по прогону chw (23.08): выигрыш ≈ 2,16·ln(вес) + 0,49, корреляция
# 0,98, перелом на весе 0,80. При множителе 4 к выравненным весам ожидается
# около +4 % в среднем по пяти каналам против +1,4 % у чистого выравнивания.
#
# Запуск: bash scripts/_paper_run_chwb_krsk.sh [множитель]   (сам уходит в фон)
#   множитель 4 — умолчание; 1 равносильно прогону chw.
set -uo pipefail
BOOST=${1:-4}
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне, множитель $BOOST. лог: /workdir/paper_results/chwb_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f; BASE=multires_krsk_33f_drop8; EXP=multires_krsk_33f_chwb
mkdir -p "$OUT"; exec >>"$OUT/chwb_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ВЕСА КАНАЛОВ С УПОРОМ НА ПУБЛИКУЕМЫЕ (множитель $BOOST) ПОВЕРХ W=30 ==="
# Сторож ловит и обучение, и инференс: 22.08.2026 чуть не запустили дожиг
# поверх пятисуточной развёртки на той же карте — прежний сторож знал
# только про src.main и predict.py бы не заметил.
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/chwb_krsk_setup.log" 2>&1
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
      --out-dir "$D33R" >> "$OUT/chwb_krsk_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

# Веса считаем по обучающей выборке — таблица уходит в отдельный лог целиком.
mkdir -p "experiments/$EXP"
log "считаю веса каналов по невязке за 6 ч, множитель $BOOST на t2m/10u/10v/msl/z@500"
python -u scripts/channel_loss_weights.py --exp "experiments/$SRC" --data-dir "$D33R" \
    --boost "$BOOST" --out "$OUT/chwb_weights.json" >> "$OUT/chwb_weights.log" 2>&1
log "веса rc=$?"
[[ -f "$OUT/chwb_weights.json" ]] || { log "веса не посчитались — стоп"; exit 1; }
grep -E "пять каналов|самый тяжёлый|множитель" "$OUT/chwb_weights.log"

python -u - "experiments/$BASE/config.json" "experiments/$EXP/config.json" "$OUT/chwb_weights.json" <<'PYEOF'
import json, sys
src, dst, wf = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
c['roi_loss_weight'] = 30.0          # рабочая точка из перебора 21–22.08
c['channel_loss_weights'] = json.load(open(wf))
for k in ('wandb_name', 'experiment_name', 'name'):
    if isinstance(c.get(k), str):
        c[k] = c[k].replace('drop8', 'chwb')
json.dump(c, open(dst, 'w'), indent=2, ensure_ascii=False)
print(f"[prep] конфиг {dst}: вес области 30, весов каналов "
      f"{len(c['channel_loss_weights'])}, эпох {c['num_epochs']}, "
      f"AR={c['initial_ar_steps']}, темп {c['learning_rate']:g} {c['lr_schedule']}")
PYEOF
python -u - "experiments/$EXP/config.json" <<'PYEOF' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
cfg = ExperimentConfig(**json.load(open(sys.argv[1])))
print(f"[prep] схема принята: roi_loss_weight={cfg.roi_loss_weight:g}, "
      f"каналов с весом {len(cfg.channel_loss_weights)}")
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
    >> "$OUT/chwb_krsk_train.log" 2>&1
log "DONE обучение rc=$?"
grep -m1 "channel_loss_weights applied" "$OUT/chwb_krsk_train.log"
[[ -f "experiments/$EXP/checkpoint.pth" ]] || { log "обучение не оставило чекпойнта — оценку не запускаю"; exit 1; }

# Оцениваем только последнюю эпоху: по четырём прогонам с косинусом до нуля
# лучшая по валидации всегда чуть хуже последней, вторая оценка — потеря 35 минут.
python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_chwb_last.pth <<'PYEOF'
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
    --ckpt /data/paper_heavy/krsk_chwb_last.pth \
    --save-sample-metrics "$OUT/chwb_krsk_roi_samples.npz" >> "$OUT/chwb_krsk_roi.log" 2>&1
RC=$?
AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/chwb_krsk_roi.log" | grep -oE 'skill=[-0-9.]+%')
log "DONE оценка rc=$RC | агрегат по области $AGG"
grep -E '^\s+(t2m|msl)\b' "$OUT/chwb_krsk_roi.log" | tail -2
log "контроль roiw30: агрегат 74.85 %, t2m 1.23/1.43/1.47/1.53 °C"
log "контроль chw (чистое выравнивание): агрегат 74.55 %, t2m 1.18/1.36/1.40/1.46 °C"
log "=== ALL DONE ==="
