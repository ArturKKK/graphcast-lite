#!/usr/bin/env bash
# Ансамбль из уже обученных моделей. Обучать нечего — считаем то, что есть.
#
# Зачем. На дисках лежат несколько региональных моделей, отличающихся весом
# области и весами каналов. Среднее по нескольким прогнозам почти всегда точнее
# любого отдельного: независимые ошибки частично гасят друг друга. Стоит это
# только времени вывода, обучение не нужно.
#
# Оговорка для статьи: участники дообучались от одного родителя, значит их
# ошибки коррелированы и выигрыш будет меньше, чем у независимых моделей. Плюс
# усреднение сглаживает поле — для RMSE хорошо, для спектров и экстремумов
# плохо, и это надо написать честно.
#
# Первым делом прогоняется ОДИН участник через тот же новый код: он обязан
# воспроизвести известное число (roiw30 → 74,85 %). Если не воспроизвёл —
# сломана обвязка, а не идея, и ансамблю верить нельзя.
#
# Запуск: bash scripts/_paper_run_ens_krsk.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/ens_krsk_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
CFG=multires_krsk_33f          # конфиг берём отсюда: архитектура у всех одна
mkdir -p "$OUT"; exec >>"$OUT/ens_krsk_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== АНСАМБЛЬ ИЗ ГОТОВЫХ МОДЕЛЕЙ ==="
BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/ens_krsk_setup.log" 2>&1
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
      --out-dir "$D33R" >> "$OUT/ens_krsk_build.log" 2>&1
  log "сборка rc=$?"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }
mkdir -p /data/paper_heavy

# Собираем веса всех участников, которых нашли. Список — от лучших к слабым;
# берём только тех, чей чекпойнт реально лежит на этой машине.
CAND="multires_krsk_33f_roiw30 multires_krsk_33f_roiw10 multires_krsk_33f_drop8 \
multires_krsk_33f_roiw100 multires_krsk_33f_chw multires_krsk_33f_chwb"
PATHS=(); NAMES=()
for EXP in $CAND; do
  CK="experiments/$EXP/checkpoint.pth"
  [[ -f "$CK" ]] || { log "нет $CK — участник $EXP пропущен"; continue; }
  DST=/data/paper_heavy/ens_${EXP#multires_krsk_33f_}.pth
  python -u - "$CK" "$DST" <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
e = ck.get("epoch")
print("[prep]", dst, "эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF
  [[ -f "$DST" ]] && { PATHS+=("$DST"); NAMES+=("$EXP"); }
done
log "участников найдено: ${#PATHS[@]} — ${NAMES[*]}"
(( ${#PATHS[@]} >= 2 )) || { log "меньше двух участников, ансамбль бессмысленен — стоп"; exit 1; }

roi() {  # roi <тег> <список чекпойнтов через запятую>
  local tag=$1 cks=$2
  log "START $tag"
  python -u scripts/predict.py "experiments/$CFG" --data-dir "$D33R" --split test_only \
      --ar-steps 4 --max-samples 2000 --per-channel --no-save --region 50 60 83 98 \
      --ensemble-ckpt "$cks" \
      --save-sample-metrics "$OUT/ens_${tag}_samples.npz" >> "$OUT/ens_$tag.log" 2>&1
  local rc=$?
  local agg
  agg=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/ens_$tag.log" | grep -oE 'skill=[-0-9.]+%')
  log "DONE $tag rc=$rc | агрегат по области $agg"
  grep -E '^\s+(t2m|msl)\b' "$OUT/ens_$tag.log" | tail -2
}

# 1. Проверка обвязки: один участник новым кодом обязан дать прежнее число.
roi "solo_${NAMES[0]#multires_krsk_33f_}" "${PATHS[0]}"
log "ожидалось для roiw30: 74.85 %, t2m 1.23/1.43/1.47/1.53 — если не совпало, дальше не верить"

# 2. Ансамбль нарастающим составом: видно, окупается ли каждый следующий участник.
CKS="${PATHS[0]}"
for i in $(seq 1 $(( ${#PATHS[@]} - 1 ))); do
  CKS="$CKS,${PATHS[$i]}"
  roi "ens$(( i + 1 ))" "$CKS"
done
log "лучший одиночный результат: roiw30, агрегат 74.85 %, t2m 1.23/1.43/1.47/1.53 °C"
log "=== ALL DONE ==="
