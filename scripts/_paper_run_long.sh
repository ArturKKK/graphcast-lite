#!/usr/bin/env bash
# Горизонты статьи: 7 и 14 суток. Шаг 6 ч против шага 24 ч.
#
# Зачем. Разделы 5.7–5.8 живут на 7 и 14 сутках. При шестичасовом шаге это 28 и
# 56 авторегрессионных применений, при суточном — 7 и 14. На пяти сутках
# суточный шаг уже обходил шестичасовой на 12,5 п.п. по успешности, и разрыв
# рос с горизонтом (−3,8 → +1,1 → +5,4 → +9,3 → +12,5 на +24…+120 ч).
# Здесь проверяем, во что это превращается на горизонтах, которые публикуются.
#
# Три модели: v3 (шаг 6 ч), v5 (шаг 24 ч, 30 эпох) и v5_ext (та же, но с резким
# снижением темпа на 31-й эпохе). Третья заодно отвечает на вопрос, доходит ли
# выигрыш от снижения темпа до дальних сроков — на валидации он 9,6 %, а у
# косинусной абляции похожий выигрыш на прогноз не дошёл вовсе.
#
# Только инференс, обучать ничего не надо. Но /data стирается при выключении
# виртуалки, поэтому глобальный датасет может потребовать пересборки.
#
# Запуск: bash scripts/_paper_run_long.sh   (сам уходит в фон)
# Лог:    /workdir/paper_results/long_master.log
set -uo pipefail

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/long_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33G=/data/datasets/wb2_512x256_33f_v3
BASE=/data/datasets/wb2_512x256_19f_ar
GEXTRA=/data/datasets/global_512x256_extra_2010-2021_07deg
MAXN=${MAXN:-200}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/long_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }

log "=== 7 И 14 СУТОК: шаг 6 ч против 24 ч ==="

if pgrep -f "src.main" >/dev/null; then
  log "на карте идёт обучение — стоп"
  exit 1
fi

# ── Окружение и глобальный датасет ────────────────────────────────────
if [[ ! -x "$VENV/bin/python" || ! -d "$BASE" ]]; then
  log "нет окружения или базового датасета — подготовка (1-2 ч)"
  bash scripts/_paper_setup_vm.sh >> "$OUT/long_setup.log" 2>&1
  log "подготовка rc=$?"
fi
source "$VENV/bin/activate" 2>/dev/null || { log "venv не поднялся — стоп"; exit 1; }
export PYTHONPATH="$REPO"

if [[ ! -f "$D33G/data.npy" ]]; then
  for d in "$BASE" "$GEXTRA"; do
    [[ -d "$d" ]] || { log "НЕТ ИСХОДНИКА $d — стоп"; exit 1; }
  done
  log "собираю глобальный 33-канальный датасет"
  python -u scripts/build_v3_extra_with_time.py \
      --base-dir "$BASE" --extra-dir "$GEXTRA" --out-dir "$D33G" \
      >> "$OUT/long_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33G" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33G/data.npy" ]] || { log "глобального датасета нет — стоп"; exit 1; }

# Последняя эпоха продолжения — отдельным файлом
python - <<PY
import torch, pathlib
s = pathlib.Path("experiments/wb2_512x256_33f_ar_v5_24h_ext/checkpoint.pth")
if s.exists():
    ck = torch.load(s, map_location="cpu")
    torch.save(ck.get("model_state_dict", ck), "$HEAVY/v5ext_last.pth")
    e = ck.get("epoch")
    print("[prep] v5_ext последняя эпоха", (e + 1) if isinstance(e, int) else e)
PY

run() {   # run <тег> <эксперимент> <шагов> [доп. аргументы]
  local tag=$1 exp=$2 steps=$3; shift 3
  [[ -f "$OUT/$tag.log" ]] && { log "$tag уже посчитан, пропускаю"; return; }
  log "START $tag ($steps шагов)"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D33G" \
      --split test_only --ar-steps "$steps" --max-samples "$MAXN" \
      --per-channel --no-save --save-sample-metrics "$OUT/${tag}_samples.npz" \
      "$@" >> "$OUT/$tag.log" 2>&1
  log "DONE $tag rc=$?"
  grep -E '^\s+\+(168|336)h:' "$OUT/$tag.log" | tail -2
}

# ── 7 суток = 168 ч ───────────────────────────────────────────────────
log "--- 7 СУТОК ---"
run d7_v3     wb2_512x256_33f_ar_v3           28
# Шестичасовая со сбросом темпа: на пяти сутках она подтянулась с 26,0 до 33,1 %,
# и без неё сравнение с суточной моделью на горизонтах статьи снова становится
# нечестным — у одной был сброс темпа, у другой нет.
run d7_v3lr   wb2_512x256_33f_ar_v3_lrdrop    28
run d7_v5     wb2_512x256_33f_ar_v5_24h        7
[[ -f "$HEAVY/v5ext_last.pth" ]] && \
  run d7_v5ext wb2_512x256_33f_ar_v5_24h_ext   7 --ckpt "$HEAVY/v5ext_last.pth"

# ── 14 суток = 336 ч ──────────────────────────────────────────────────
log "--- 14 СУТОК ---"
run d14_v3    wb2_512x256_33f_ar_v3           56
run d14_v3lr  wb2_512x256_33f_ar_v3_lrdrop    56
run d14_v5    wb2_512x256_33f_ar_v5_24h       14
[[ -f "$HEAVY/v5ext_last.pth" ]] && \
  run d14_v5ext wb2_512x256_33f_ar_v5_24h_ext 14 --ckpt "$HEAVY/v5ext_last.pth"

log "--- ИТОГ: приземная температура ---"
for t in d7_v3 d7_v3lr d7_v5 d7_v5ext d14_v3 d14_v3lr d14_v5 d14_v5ext; do
  [[ -f "$OUT/$t.log" ]] || continue
  log "$t: $(grep -E '^\s+t2m' "$OUT/$t.log" | tail -1 | tr -s ' ' | cut -c1-150)"
done
log "опорные числа на +120 ч: v3 2.97 °C (26.0 %), v5 2.69 °C (38.5 %)"
log "=== ALL DONE ==="
