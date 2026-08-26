#!/usr/bin/env bash
# Обучение на ТРОЕ СУТОК развёртки (12 шагов) поверх лучшей модели.
#
# Зачем. Всё обучение шло на четыре шага, то есть на сутки. На трёх сутках
# модель работает в режиме свободной экстраполяции: ошибка по области 2,53 °C
# по t2m и успешность 68 %. Никто её этому не учил.
#
# Почему не сделали раньше: при сквозном пробросе градиента память растёт с
# длиной развёртки, и на четырёх шагах уже занято 70 ГБ из 80. Двенадцать не
# влезают. Поэтому ar_detach_steps=True: вход каждого шага отцепляется, градиент
# делается пошагово, память остаётся на уровне одного шага.
#
# Это же — правильная версия приёма, провалившегося 23.08 с шумом. Там мы
# подмешивали гауссов шум, ИЗОБРАЖАЯ ошибки модели; здесь модель получает свои
# настоящие ошибки, накопленные развёрткой. Плата: градиент не течёт между
# шагами, то есть модель не учится готовить состояние для следующего шага.
#
# Стоимость: эпоха примерно втрое дороже (12 шагов вместо 4), около 8 часов.
# Четыре эпохи — это уже в полтора раза больше градиентной работы, чем весь
# восьмиэпошный дожиг на четырёх шагах.
#
# Оценка идёт на 20 шагов по 500 сроков — ровно как krsk5d, чтобы сравнивать
# напрямую с таблицей той развёртки:
#   roiw30 по области: +24 ч 80,22 %, +72 ч 68,21 %, +120 ч 46,25 %
#   t2m:               +24 ч 1,71 °C, +72 ч 2,53 °C, +120 ч 3,79 °C
#
# Запуск: bash scripts/_paper_run_ar12_krsk.sh [эпох]   (сам уходит в фон)
set -uo pipefail
EPOCHS=${1:-4}
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне, эпох $EPOCHS. лог: /workdir/paper_results/ar12_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
SRC=multires_krsk_33f_roiw30; BASE=multires_krsk_33f_drop8; EXP=multires_krsk_33f_ar12
mkdir -p "$OUT"; exec >>"$OUT/ar12_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== РАЗВЁРТКА НА 12 ШАГОВ С ОТЦЕПЛЕНИЕМ, $EPOCHS эпох ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/ar12_krsk_setup.log" 2>&1
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
      --out-dir "$D33R" >> "$OUT/ar12_krsk_build.log" 2>&1
  log "сборка rc=$?"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }

mkdir -p "experiments/$EXP" /data/paper_heavy
python -u - "experiments/$BASE/config.json" "experiments/$EXP/config.json" "$EPOCHS" <<'PYEOF'
import json, sys
src, dst, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(src))
c['roi_loss_weight'] = 30.0        # рабочая точка из перебора 21-22.08
c['num_epochs'] = ep
c['initial_ar_steps'] = 12         # сразу полная развёртка, без разгона
c['max_ar_steps'] = 12
c['val_ar_steps'] = 12             # отбор по среднему за трое суток, а не за сутки
c['ar_detach_steps'] = True        # иначе не хватит памяти карты
# Окно при 12 шагах вдвое с лишним тяжелее, чем при четырёх, и загрузчик с
# четырьмя процессами выедал ОЗУ контейнера: прогон дважды убило сигналом без
# трассировки. Два процесса с предвыборкой 1 снижают аппетит вчетверо.
c['dataloader_workers'] = 2
c['dataloader_prefetch'] = 1
for k in ('wandb_name', 'experiment_name', 'name'):
    if isinstance(c.get(k), str):
        c[k] = c[k].replace('drop8', 'ar12')
json.dump(c, open(dst, 'w'), indent=2, ensure_ascii=False)
print(f"[prep] конфиг {dst}: вес области 30, развёртка {c['max_ar_steps']} шагов, "
      f"отцепление {c['ar_detach_steps']}, валидация на {c['val_ar_steps']} горизонтах, "
      f"эпох {ep}, темп {c['learning_rate']:g} {c['lr_schedule']}")
PYEOF
python -u - "experiments/$EXP/config.json" <<'PYEOF' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
cfg = ExperimentConfig(**json.load(open(sys.argv[1])))
print(f"[prep] схема принята: развёртка {cfg.max_ar_steps}, отцепление {cfg.ar_detach_steps}")
PYEOF

# Стартуем с лучшей модели: roiw30, последняя эпоха.
PRETRAINED=/data/paper_heavy/krsk_ar12_start.pth
python -u - "experiments/$SRC/checkpoint.pth" "$PRETRAINED" <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print("[prep] нет чекпойнта roiw30"); sys.exit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
print("[prep] стартуем с эпохи", (ck.get("epoch", 0) or 0) + 1)
PYEOF
[[ -f "$PRETRAINED" ]] || { log "нет стартовых весов — стоп"; exit 1; }

RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume --reset-patience"; log "нашёлся чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD 2>/dev/null)"
# Расширяемые сегменты убирают фрагментацию распределителя — на длинных
# прогонах это иногда решает, а вреда нет.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "START обучения $EXP (эпоха примерно 8 ч) $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/ar12_krsk_train.log" 2>&1
log "DONE обучение rc=$?"
grep -m1 "развёртка с отцеплением" "$OUT/ar12_krsk_train.log" \
  || log "ВНИМАНИЕ: строки про отцепление нет — проверь, применился ли флаг"
[[ -f "experiments/$EXP/checkpoint.pth" ]] || { log "нет чекпойнта — оценку не запускаю"; exit 1; }

python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_ar12_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
print("[prep] последняя эпоха", (ck.get("epoch", 0) or 0) + 1)
PYEOF
log "START оценки на 20 шагов по 500 сроков (как krsk5d)"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 20 --max-samples 500 --per-channel --no-save --region 50 60 83 98 \
    --ckpt /data/paper_heavy/krsk_ar12_last.pth \
    --save-sample-metrics "$OUT/ar12_krsk_roi_samples.npz" >> "$OUT/ar12_krsk_roi.log" 2>&1
RC=$?
AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/ar12_krsk_roi.log" | grep -oE 'skill=[-0-9.]+%')
log "DONE оценка rc=$RC | агрегат за 5 суток $AGG"
awk '/--- Region /{c=1} c&&/\+(024|072|120)h:/{print "    "$0}' "$OUT/ar12_krsk_roi.log"
grep -E '^\s+t2m\b' "$OUT/ar12_krsk_roi.log" | tail -1
log "контроль roiw30 (та же оценка): агрегат 64.49 %, +24 ч 80.22 %, +72 ч 68.21 %, +120 ч 46.25 %"
log "=== ALL DONE ==="
