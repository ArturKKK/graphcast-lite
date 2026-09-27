#!/usr/bin/env bash
# Пересчёт статьи на итоговой модели: всё, что не даёт оценка t_<опыт>_roi.
#
# Зачем (27.09.2026). Табл. 5 (стык) и рис. 1 были посчитаны на старой
# 19-канальной модели multires_merge_freeze6_v2, а текст подавал их как
# результаты основной. Для внутренней зоны табл. 1 нужен отдельный прогон.
# Этот батч считает для выбранного опыта:
#   p_<опыт>_inner  — вся тестовая выборка, внутренняя зона (табл. 1), с
#                     широтным весом и ACC против климатологии WB2;
#   p_<опыт>_s100   — первые 100 сроков, область, без широтного веса, с
#                     сохранением полных полей в /data (для стыка);
#   p_<опыт>_seam_* — профиль ошибки по расстоянию до границы вставки (табл. 5)
#                     и данные для карты стыка (рис. 1). Тяжёлые поля (~14 ГБ)
#                     после этого удаляются.
# Табл. 2 не пересчитывается: для неё нужна глобальная 33-канальная модель на
# глобальных данных, а их на виртуалке нет (удаляются при восстановлении ради
# места). Она остаётся на 19-канальной линии и так подписана в статье.
#
# Запуск:  bash scripts/_vm_batch_fin.sh dec_long      — сразу
#          bash scripts/_vm_batch_fin.sh dec_long w    — дождаться всех идущих
#                                                        на машине батчей improve
# Лог:     /workdir/paper_results/fin_<опыт>_master.log
set -uo pipefail
V=${1:-}
WAIT=${2:-}
[[ "$V" =~ ^(dec|dec_long|dec_encres|dec_long_encres|long)$ ]] \
  || { echo "опыт: dec, dec_long, dec_encres, dec_long_encres или long"; exit 1; }
[[ -z "$WAIT" || "$WAIT" == "w" ]] || { echo "второй аргумент — только w (ждать)"; exit 1; }

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/fin_${V}_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
ROI="50 60 83 98"
INNER="55.5 56.5 92 94"
CLIM=$REPO/docs/paper/runs/clim_wb2_nodes.npz
EXP=multires_krsk_33f_chw_${V}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/fin_${V}_master.log"
exec 9>"$OUT/.fin_${V}.lock"
flock -n 9 || { echo "[$(date '+%d.%m %H:%M:%S')] уже идёт" >> "$MASTER"; exit 0; }
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== ПЕРЕСЧЁТ СТАТЬИ: $V ($(git rev-parse --short HEAD)) ==="

# Очередь: ждём все батчи improve на этой машине. Каждый держит свою
# блокировку до конца обучения и оценки; у законченных она свободна сразу.
if [[ "$WAIT" == "w" ]]; then
  for lk in "$OUT"/.improve_*.lock; do
    [[ -e "$lk" ]] || continue
    log "жду $(basename "$lk")"
    exec 8>>"$lk"; flock 8; exec 8>&-
  done
  log "все батчи на машине закончились — начинаю"
  sleep 60
fi

BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# После перезапуска машины /data пуст: восстанавливаем окружение и датасет
# тем же скриптом, что и батчи improve (~40 мин).
if [[ ! -x "$VENV/bin/python" || ! -f "$D33/data_extra.npy" ]]; then
  log "нет окружения или датасета — восстанавливаю (~40 мин), лог /workdir/paper_logs/restore33f.log"
  DAEMONIZED=1 FREE=1 bash scripts/_vm_restore33f.sh
  log "восстановление rc=$?"
fi
[[ -f "$D33/data_extra.npy" ]] || { log "нет датасета $D33 — стоп"; exit 1; }
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ---------- чекпойнт ----------
CK=$HEAVY/${EXP}_last.pth
if [[ ! -f "$CK" ]]; then
  python - "experiments/$EXP/checkpoint.pth" "$CK" <<'PY' || { log "нет чекпойнта $EXP — стоп"; exit 1; }
import sys, pathlib, torch
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print(f"[prep] нет {src}"); raise SystemExit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
print(f"[prep] {src.parent.name}: эпоха {ck.get('epoch','?')} -> {dst}")
PY
fi
log "чекпойнт: $CK"

run() {   # run <тег> <макс. сроков> <область> [доп. ключи]
  local tag="$1" n="$2" reg="$3"; shift 3
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag (уже посчитан)"; return 0; }
  log "START $tag"
  # shellcheck disable=SC2086
  python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33" \
      --split test_only --ar-steps 4 --max-samples "$n" --per-channel \
      --region $reg --ckpt "$CK" --save-sample-metrics "$npz" "$@" > "$lf" 2>&1
  local rc=$? t2
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $t2"
  return $rc
}

# ---------- 1. внутренняя зона, вся тестовая выборка (табл. 1) ----------
run "p_${V}_inner" 2000 "$INNER" --no-save --lat-weight --climatology "$CLIM"

# ---------- 2. стык: первые 100 сроков, полные поля ----------
PRED=$HEAVY/p_${V}_s100.pt
if [[ ! -f "$OUT/p_${V}_seam_profile.md" ]]; then
  if run "p_${V}_s100" 100 "$ROI" --save "$PRED"; then
    SEAM=$OUT/p_${V}_seam
    mkdir -p "$SEAM"
    log "START диагностика стыка"
    python -u scripts/paper_seam_diagnostic.py --predictions "$PRED" \
        --data-dir "$D33" --out "$SEAM" > "$OUT/p_${V}_seam.log" 2>&1
    log "DONE  диагностика стыка rc=$?"
    cp -p "$SEAM/seam_profile.md" "$OUT/p_${V}_seam_profile.md" 2>/dev/null
    cp -p "$SEAM/seam_map_data.npz" "$OUT/p_${V}_seam_map_data.npz" 2>/dev/null
    grep -E "^\| вставка|^\| глобальная" "$OUT/p_${V}_seam_profile.md" 2>/dev/null
  fi
  rm -f "$PRED" && log "полные поля удалены (место в /data)"
fi

log "=== ВСЁ ==="
grep -E "DONE  " "$MASTER" | tail -4
