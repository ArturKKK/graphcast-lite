#!/usr/bin/env bash
# Глобальная v4: обучение с нуля с полным многоуровневым мешем (0..6).
#
# Зачем с нуля. Региональная попытка сорвалась не из-за самого приёма, а из-за
# того, что веса брались от глобальной v3, обученной на мешe [4, 6]: процессор
# впервые увидел рёбра длиной до 7000 км вместо привычных 440 км и стартовал с
# ошибкой 0.21 против 0.02 у базовой конфигурации. Восемь эпох ушло на
# восстановление, и сравнение оказалось бессмысленным.
#
# Здесь модель с самого начала видит все масштабы, поэтому сравнение с v3
# честное: единственное отличие конфигурации — набор уровней меша.
#
# ⚠️ Запускать ТОЛЬКО одной на видеокарте. Два обучения разом уронили прошлый
# прогон по памяти на переходе AR=2 -> AR=3 и втрое замедлили друг друга.
#
# Порядок: остановка прежнего обучения -> оценка замороженной модели ->
# подготовка глобального 33-канального датасета -> обучение v4.
#
# Запуск: nohup setsid bash scripts/_paper_run_v4_global.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/v4_master.log
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33G=/data/datasets/wb2_512x256_33f_v3      # глобальный 33-канальный
BASE=/data/datasets/wb2_512x256_19f_ar      # базовые 19 каналов
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
D33R=/data/datasets/multires_krsk_33f       # региональный, для оценки заморозки
EXPF=multires_krsk_33f_frozen
EXP4=wb2_512x256_33f_ar_v4_multimesh

mkdir -p "$OUT"
MASTER="$OUT/v4_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== V4 GLOBAL MULTIMESH START ==="

# ── 0. Окружение и базовые датасеты ───────────────────────────────────
# После перезапуска виртуалки /data стирается целиком: ни venv, ни датасетов.
# Подготовку делает _paper_setup_vm.sh (venv + распаковка глобального и
# красноярских датасетов + сборка merge), она же нужна и здесь.
if [[ ! -x "$VENV/bin/python" || ! -d "$BASE" ]]; then
  log "нет окружения или базового датасета — запускаю подготовку (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/v4_setup.log" 2>&1
  log "подготовка завершена rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "venv так и не поднялся — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ── 1. Освободить карту ───────────────────────────────────────────────
if pgrep -f "src.main" > /dev/null; then
  log "останавливаю текущее обучение (заморозка вышла на плато)"
  pkill -f "src.main"; sleep 20
fi
log "GPU свободна: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

# ── 2. Оценка замороженной модели — нужна статье в физических единицах ─
if [[ -f "experiments/$EXPF/best_model.pth" && -f "$D33R/data.npy" ]]; then
  for STATE in best last; do
    CK=""
    if [[ "$STATE" == "last" ]]; then
      python - <<PY
import torch, pathlib
s = pathlib.Path("experiments/$EXPF/checkpoint.pth")
if s.exists():
    ck = torch.load(s, map_location="cpu")
    pathlib.Path("/data/paper_heavy").mkdir(parents=True, exist_ok=True)
    torch.save(ck.get("model_state_dict", ck), "/data/paper_heavy/frozen_last.pth")
    print("[prep] эпоха", ck.get("epoch"))
PY
      [[ -f /data/paper_heavy/frozen_last.pth ]] && CK="--ckpt /data/paper_heavy/frozen_last.pth"
    fi
    tag="frozen_${STATE}_roi"
    log "START $tag"
    python -u scripts/predict.py "experiments/$EXPF" --data-dir "$D33R" $CK \
        --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
        --region 50 60 83 98 --save-sample-metrics "$OUT/${tag}_samples.npz" \
        >> "$OUT/${tag}.log" 2>&1
    log "DONE  $tag | $(grep -oE 'skill=[-0-9.]+%' "$OUT/${tag}.log" | tail -1) | $(grep -E '^\s+t2m' "$OUT/${tag}.log" | tail -1 | tr -s ' ' | cut -c1-60)"
  done
  log "сравнивать с основной моделью: t2m 1.32/1.53/1.59/1.66 °C, успешность 75.31 %"
else
  log "пропускаю оценку заморозки: нет весов или регионального датасета"
fi

# ── 3. Глобальный 33-канальный датасет ────────────────────────────────
if [[ ! -f "$D33G/data.npy" ]]; then
  log "собираю глобальный 33-канальный датасет (часы CPU)"
  for d in "$BASE" "$GEXTRA"; do
    [[ -d "$d" ]] || { log "НЕТ ИСХОДНИКА $d — сборка невозможна"; exit 1; }
  done
  python -u scripts/build_v3_extra_with_time.py \
      --base-dir "$BASE" --extra-dir "$GEXTRA" --out-dir "$D33G" \
      >> "$OUT/v4_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33G" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33G/data.npy" ]] || { log "глобального датасета нет — стоп"; exit 1; }

# ── 4. Провенанс и обучение ───────────────────────────────────────────
log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения"
log "START обучения $EXP4 (с нуля, без pretrained)"
python -u -m src.main "experiments/$EXP4" >> "$OUT/v4_train.log" 2>&1
log "DONE обучение rc=$?"
log "=== ALL DONE ==="
