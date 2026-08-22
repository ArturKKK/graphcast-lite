#!/usr/bin/env bash
# Прогноз ПО ОБЛАСТИ на пять суток: держится ли выигрыш от веса области дальше 24 ч.
#
# Зачем. Всё, что мы мерили по области, обрывалось на сутках. Вес области поднял
# долю вставки в лоссе, но чуть просадил глобальное поле (0,44 % при W=30), а
# прогноз во вставке живёт притоком извне. На сутках это не проявилось. Если
# выигрыш доживает до трёх-пяти суток — результат крепкий; если рассыпается —
# это надо знать и написать.
#
# Развёртка 20 шагов по 6 ч. Датасет пересобирать не нужно: predict.py строит
# окно по --ar-steps.
#
# Запуск: bash scripts/_paper_run_krsk_5day.sh   (сам уходит в фон)
set -uo pipefail
if [[ "${DAEMONIZED:-}" != "1" ]]; then
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. лог: /workdir/paper_results/krsk5d_master.log"; exit 0
fi
REPO=/workdir/graphcast-lite; VENV=/data/venvs/graphcast; OUT=/workdir/paper_results
D33R=/data/datasets/multires_krsk_33f
STEPS=20; MAXN=500
mkdir -p "$OUT"; exec >>"$OUT/krsk5d_master.log" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ПРОГНОЗ ПО ОБЛАСТИ НА $((STEPS*6/24)) СУТОК, $STEPS шагов, $MAXN сроков ==="
pgrep -f "src.main" >/dev/null && { log "карта занята обучением — стоп"; exit 1; }

if [[ ! -x "$VENV/bin/python" || ! -f "$D33R/data.npy" ]]; then
  log "подготовка окружения и датасета"
  bash scripts/_paper_setup_vm.sh >> "$OUT/krsk5d_setup.log" 2>&1
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
      --out-dir "$D33R" >> "$OUT/krsk5d_build.log" 2>&1
  log "сборка rc=$? размер $(du -sh "$D33R" 2>/dev/null | cut -f1)"
fi
[[ -f "$D33R/data.npy" ]] || { log "нет $D33R — стоп"; exit 1; }
mkdir -p /data/paper_heavy

# Веса берём из чекпойнтов в /workdir — он переживает перезапуск виртуалки,
# в отличие от /data, где после рестарта пусто.
for pair in "multires_krsk_33f:main" "multires_krsk_33f_drop8:w1" \
            "multires_krsk_33f_roiw10:w10" "multires_krsk_33f_roiw30:w30"; do
  EXP=${pair%%:*}; TAG=${pair##*:}
  CKPT="experiments/$EXP/checkpoint.pth"
  DST=/data/paper_heavy/krsk5d_$TAG.pth
  if [[ ! -f "$CKPT" ]]; then log "ПРОПУСК $TAG: нет $CKPT"; continue; fi
  python -u - "$CKPT" "$DST" <<'PYEOF'
import sys, torch, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
e = ck.get("epoch")
print("[prep]", dst, "эпоха", (e + 1) if isinstance(e, int) else e)
PYEOF
  [[ -f "$DST" ]] || { log "ПРОПУСК $TAG: веса не извлеклись"; continue; }
  log "START $TAG ($EXP)"
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33R" --split test_only \
      --ar-steps "$STEPS" --max-samples "$MAXN" --per-channel --no-save \
      --region 50 60 83 98 --ckpt "$DST" \
      --save-sample-metrics "$OUT/krsk5d_${TAG}_samples.npz" >> "$OUT/krsk5d_$TAG.log" 2>&1
  RC=$?
  AGG=$(awk '/--- Region /{getline; l=$0} END{print l}' "$OUT/krsk5d_$TAG.log" | grep -oE 'skill=[-0-9.]+%')
  log "DONE $TAG rc=$RC | агрегат по области за $((STEPS*6/24)) суток $AGG"
  # t2m по суточным отсечкам, чтобы форма кривой была видна прямо в мастер-логе
  awk '/Per-horizon per-channel RMSE — REGION/{c=1} c&&/^ +t2m /{print "    t2m:", $0; exit}' "$OUT/krsk5d_$TAG.log"
done
log "=== ALL DONE ==="
