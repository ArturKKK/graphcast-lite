#!/usr/bin/env bash
# Абляция: косинусный график темпа обучения с разогревом против постоянного.
#
# Гипотеза. До 11.08.2026 темп обучения был постоянным всё обучение. В логах это
# видно прямо: глобальная v3 взяла лучшее значение 0.01868 на 18-й эпохе и за
# следующие восемнадцать эпох улучшила его до 0.01865, то есть ни на сколько.
# Так выглядит блуждание вокруг минимума со слишком крупным шагом, а не выход на
# сходимость. В GraphCast — линейный разогрев 1000 шагов и косинусный спад до нуля.
#
# Почему региональная модель, а не глобальная. Эпоха здесь стоит часы, а не сутки,
# и есть с чем сравнивать: основную модель статьи переобучать не нужно, её лог
# сохранён (experiments/multires_krsk_33f/training_log.txt). Ровно тот протокол,
# которым закрывали вопрос о заморозке процессора.
#
# Единственное отличие конфигурации — три поля lr_*. Сид, датасет, расписание,
# заморозка и веса каналов совпадают.
#
# ⚠️ Только одно обучение на видеокарте. Два разом уже уронили августовский прогон.
#
# Запуск: bash scripts/_paper_run_cosine.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/cosine_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/cosine_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
EXP=multires_krsk_33f_cosine
PRETRAINED=experiments/wb2_512x256_33f_ar_v3/best_model.pth

mkdir -p "$OUT"
MASTER="$OUT/cosine_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== COSINE LR ABLATION START ==="

# ── 1. Карта должна быть свободна ─────────────────────────────────────
if pgrep -f "src.main" >/dev/null; then
  log "на карте уже идёт обучение — останавливаю"
  pkill -f "src.main"; sleep 20
fi
pkill -f "_paper_run_v3v4_compare" 2>/dev/null
log "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

# ── 2. Окружение и датасет ────────────────────────────────────────────
# /data стирается при перезапуске виртуалки, поэтому проверяем, а не полагаемся.
if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "нет окружения или регионального датасета — запускаю подготовку (часы)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/cosine_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "venv не поднялся — стоп"; exit 1; }
export PYTHONPATH="$REPO"

if [[ ! -f "$D33R/data.npy" ]]; then
  # Сборщику нужны координаты глобальной сетки, а у global_extra своих нет.
  # 11.08.2026 на этом потеряли запуск: копирование в setup_vm сидело внутри
  # ветки SLIM, которую здесь не включают. Проверяем сами и чиним двумя путями.
  GX=/data/datasets/global_512x256_extra_2010-2021_07deg
  if [[ ! -f "$GX/coords.npz" ]]; then
    if [[ -f /data/datasets/wb2_512x256_19f_ar/coords.npz ]]; then
      cp -p /data/datasets/wb2_512x256_19f_ar/coords.npz "$GX/coords.npz"
      log "coords.npz скопирован в global_extra из базового датасета"
    else
      log "coords.npz нет и базового датасета нет — восстанавливаю из merge"
      python -u scripts/fix_global_extra_coords.py >> "$OUT/cosine_build.log" 2>&1
      log "восстановление rc=$?"
    fi
  fi
  [[ -f "$GX/coords.npz" ]] || { log "coords.npz так и нет — сборка невозможна"; exit 1; }

  log "нет $D33R — собираю 33-канальный региональный датасет (часы CPU)"
  python -u scripts/build_multires_russia_33f.py \
      --multires-dir /data/datasets/multires_krsk_19f_merge \
      --extra-dir /data/datasets/global_512x256_extra_2010-2021_07deg \
      --region-extra-dir /data/datasets/region_krsk_61x41_extra_2010-2020_025deg \
      --out-dir "$D33R" >> "$OUT/cosine_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "датасета так и нет — стоп"; exit 1; }
[[ -f "$PRETRAINED" ]] || { log "нет весов глобальной v3 ($PRETRAINED) — стоп"; exit 1; }

# ── 3. Провенанс и обучение ───────────────────────────────────────────
log "коммит: $(git rev-parse --short HEAD)"
[[ -n "$(git status --porcelain)" ]] && log "ВНИМАНИЕ: незакоммиченные изменения — модель будет невоспроизводима"
log "START обучения $EXP (косинус, разогрев 1000 шагов)"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" >> "$OUT/cosine_train.log" 2>&1
log "DONE обучение rc=$?"

# ── 4. Оценка на том же окне, что и основная модель ───────────────────
log "START оценки на тестовом окне"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" \
    --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
    --region 50 60 83 98 --save-sample-metrics "$OUT/cosine_roi_samples.npz" \
    >> "$OUT/cosine_roi.log" 2>&1
log "DONE оценка | $(grep -oE 'skill=[-0-9.]+%' "$OUT/cosine_roi.log" | tail -1)"
grep -E '^\s+(t2m|z500|msl)\b' "$OUT/cosine_roi.log" | tail -3
log "сравнивать с multires_krsk_33f: t2m 1.32/1.53/1.59/1.66 °C, успешность 75.31 %"
log "=== ALL DONE ==="
