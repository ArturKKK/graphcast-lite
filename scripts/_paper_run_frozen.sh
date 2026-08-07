#!/usr/bin/env bash
# Абляция: нужна ли региональная адаптация процессора?
#
# Обучает multires_krsk_33f_frozen — полную копию основной модели, но с
# процессором, замороженным на всё обучение. Учатся только кодировщик и
# декодировщик, то есть интерфейсы между расчётной сеткой и графом.
#
# Что проверяем. Контрольный эксперимент п. 4.4 показал, что РАСПИСАНИЕ
# заморозки не влияет: обе конфигурации при равном бюджете дают одно и то же.
# Но в обеих процессор в итоге обучался. Вопрос «нужно ли его обучать вообще»
# остался открытым, и данные намекают, что нужно: в логе обучения основной
# модели за шесть замороженных эпох val улучшился с 0.01943 до 0.01922, а
# первая же эпоха после разморозки дала 0.01831 — скачок впятеро больший.
#
# Возможные исходы, оба публикуемы:
#   * качество заметно хуже → адаптация динамики необходима, и мы измерили,
#     сколько именно она даёт;
#   * качество сопоставимо → глобальная динамика переносится на регион как
#     есть, и адаптация под новую территорию сводится к обучению интерфейсов.
#     Это сильное утверждение: перенос на любой другой регион становится
#     кратно дешевле.
#
# Сравнение только при РАВНОМ БЮДЖЕТЕ — состояния одинаковых эпох, не «лучшие
# по val». Ровно та ошибка, на которой развалилась первая версия абляции.
#
# Запуск: nohup setsid bash scripts/_paper_run_frozen.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/frozen_master.log
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33=/data/datasets/multires_krsk_33f
MERGE=/data/datasets/multires_krsk_19f_merge
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
REXTRA=/data/datasets/region_krsk_61x41_extra_2010-2020_025deg
EXP=multires_krsk_33f_frozen
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

mkdir -p "$OUT"
MASTER="$OUT/frozen_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== FROZEN ABLATION START ==="

# ── 0. Окружение и датасеты ───────────────────────────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
  log "нет venv — запускаю подготовку (venv + глобальный + Krsk + merge)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/frozen_setup.log" 2>&1
  log "подготовка завершена rc=$?"
fi
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

if [[ ! -f "$D33/data.npy" ]]; then
  log "собираю 33f-датасет"
  [[ -f "$GEXTRA/coords.npz" ]] || python -u scripts/fix_global_extra_coords.py \
      --merge-dir "$MERGE" --extra-dir "$GEXTRA" >> "$OUT/frozen_build.log" 2>&1
  python -u scripts/build_multires_russia_33f.py \
      --multires-dir "$MERGE" --extra-dir "$GEXTRA" \
      --region-extra-dir "$REXTRA" --out-dir "$D33" >> "$OUT/frozen_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33/data.npy" ]] || { log "датасета нет — стоп"; exit 1; }

# ── 1. Провенанс ──────────────────────────────────────────────────────
log "коммит: $(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  log "ВНИМАНИЕ: есть незакоммиченные изменения — модель будет невоспроизводима"
  git status --porcelain | head -5
fi

# ── 2. Обучение ───────────────────────────────────────────────────────
log "START обучения $EXP (процессор заморожен все 32 эпохи)"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" \
    >> "$OUT/frozen_train.log" 2>&1
log "DONE обучение rc=$?"

# ── 3. Оценка на том же окне, что и основная модель ───────────────────
for STATE in best last; do
  CK=""
  if [[ "$STATE" == "last" ]]; then
    python - <<PY
import torch, pathlib
src = pathlib.Path("experiments/$EXP/checkpoint.pth")
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), "/data/paper_heavy/frozen_last.pth")
    print("epoch", ck.get("epoch"))
PY
    CK="--ckpt /data/paper_heavy/frozen_last.pth"
  fi
  tag="frozen_${STATE}_roi"
  log "START $tag"
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33" $CK \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region 50 60 83 98 --save-sample-metrics "$OUT/${tag}_samples.npz" \
      >> "$OUT/${tag}.log" 2>&1
  log "DONE  $tag | $(grep -oE 'skill=[-0-9.]+%' "$OUT/${tag}.log" | tail -1) | $(grep -E '^\s+t2m' "$OUT/${tag}.log" | tail -1 | tr -s ' ' | cut -c1-60)"
done

log "=== ALL DONE ==="
log "сравнивать с multires_krsk_33f: t2m 1.32/1.53/1.59/1.66 °C, успешность 75.31 %"
