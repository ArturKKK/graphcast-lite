#!/usr/bin/env bash
# Основная модель статьи, переобученная от УЛУЧШЕННОЙ глобальной.
#
# Зачем. Модель статьи дообучалась от wb2_512x256_33f_ar_v3. Контрольный прогон
# 16.08 показал, что сброс темпа в конце обучения улучшает эту глобальную модель
# заметно: на пяти сутках 33,1 % против 26,0 % по сводному показателю и 2,50
# против 2,97 °C по приземной температуре. Логично стартовать региональное
# дообучение от неё, а не от прежней.
#
# Расписание совпадает с основной моделью (32 эпохи, max_ar=4, заморозка
# процессора 6 эпох, lr 1e-4), добавлен только косинусный спад — приём проверен
# и на глобальной, и на региональной линии.
#
# Цель прикладная: улучшить ПУБЛИКУЕМЫЕ числа, а не измерить очередной эффект.
# Сравнивать с основной моделью: t2m 1.32/1.53/1.59/1.66 °C, агрегатная
# успешность по области 73,41 %.
#
# Запуск: bash scripts/_paper_run_krsk_from_v3lr.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/krsk_v3lr_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
EXP=multires_krsk_33f_v3lr
PRETRAINED=experiments/wb2_512x256_33f_ar_v3_lrdrop/best_model.pth
mkdir -p "$OUT"; exec >>"$OUT/krsk_v3lr_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== РЕГИОНАЛЬНАЯ ОТ УЛУЧШЕННОЙ ГЛОБАЛЬНОЙ ==="
pgrep -f "src.main" >/dev/null && { log "карта занята — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/krsk_v3lr_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }
[[ -f "$PRETRAINED" ]] || { log "НЕТ ВЕСОВ $PRETRAINED — их надо запушить с машины, где считался контроль"; exit 1; }

RESUME=""; [[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "найден чекпойнт — продолжаю"; }
log "коммит: $(git rev-parse --short HEAD)"
log "START обучения $EXP от $PRETRAINED $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$PRETRAINED" $RESUME \
    >> "$OUT/krsk_v3lr_train.log" 2>&1
log "DONE обучение rc=$?"

# Оценивать последнюю эпоху: отбор по одношаговой ошибке систематически
# промахивается мимо дальних горизонтов (п. 4.4 статьи).
mkdir -p /data/paper_heavy
python -u - "experiments/$EXP/checkpoint.pth" /data/paper_heavy/krsk_v3lr_last.pth <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if src.exists():
    ck = torch.load(src, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), dst)
    e = ck.get("epoch"); print("[prep] последняя эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF
CK=""; [[ -f /data/paper_heavy/krsk_v3lr_last.pth ]] && CK="--ckpt /data/paper_heavy/krsk_v3lr_last.pth"
log "START оценки по области $CK"
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
    --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 $CK \
    --save-sample-metrics "$OUT/krsk_v3lr_roi_samples.npz" >> "$OUT/krsk_v3lr_roi.log" 2>&1
log "DONE оценка rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$OUT/krsk_v3lr_roi.log" | tail -1)"
grep -E '^\s+(t2m|msl)\b' "$OUT/krsk_v3lr_roi.log" | tail -2
log "сравнивать с основной моделью: t2m 1.32/1.53/1.59/1.66 °C, агрегат 73.41 %"
log "=== ALL DONE ==="
