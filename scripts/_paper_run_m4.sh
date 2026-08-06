#!/usr/bin/env bash
# M4 — абляционная матрица «зачем нужен единый граф».
#
# Отвечает на первый вопрос рецензента: что даёт вставка 0.25° по сравнению
# с интерполяцией прогноза глобальной модели в те же узлы. Все строки считаются
# на ОДНОМ окне (полный test_only) и против ОДНОЙ истины (реальный ERA5 0.25°
# в узлах вставки), иначе сравнение некорректно.
#
# Строки матрицы:
#   1. инерционный прогноз      — считается внутри predict.py (base= в логах)
#   2. глобальная v2 + интерполяция в ROI
#   3. региональная GNN standalone (без глобального контекста)
#   4. мультимасштабная модель  — уже посчитана (m19_flagship_roi, Skill 70.26 %)
#
# ⚠️ ГЛАВНАЯ ЛОВУШКА: wb2_512x256_19f_ar_v2 обучена с use_residual=false, а
# predict.py по умолчанию складывает X_last + delta. Без --no-residual получается
# мусор (t2m RMSE под 130 °C). Именно на этом упал прогон сравнения v2/v3.
#
# Предпосылки: отработавший scripts/_paper_setup_vm.sh (venv + global + merge).
# Запуск: nohup setsid bash scripts/_paper_run_m4.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/m4_master.log
set -uo pipefail

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results          # логи и метрики — лёгкое, сюда можно
HEAVY=/data/paper_heavy             # тяжёлые .pt — ТОЛЬКО сюда (см. предупреждение ниже)
GLOBAL=/data/datasets/wb2_512x256_19f_ar
MERGE=/data/datasets/multires_krsk_19f_merge
REGION=/data/datasets/region_krsk_61x41_19f_2010-2020_025deg
ROI="50 60 83 98"

# ⚠️ ОГРАНИЧЕНИЕ ПАМЯТИ: predict.py --save копит предсказания И истину по всей
# глобальной сетке 512x256 в оперативке до самого конца прогона. На полном тесте
# это 1607 x 4 x 19 x 131072 x 4 байта x2 ≈ 128 ГБ — гарантированный OOM.
# Поэтому вся матрица M4 считается на подвыборке MAXN сроков (первые MAXN
# сроков тестового окна, одни и те же для всех строк — сравнение остаётся
# корректным). 200 сроков ≈ 16 ГБ, помещается с запасом.
# ⚠️ ДИСКИ: /workdir переживает перезапуск, но квота всего ~8 ГБ (df врёт про
# терабайт!) — при превышении платформа ГАСИТ виртуалку. Поэтому тяжёлый .pt
# идёт в /data, где места много, но всё стирается при рестарте и действует
# лимит ~240 ГБ. В /workdir остаются только логи и .npz.
# 100 сроков → .pt около 8 ГБ. Больше не ставить не подумав.
MAXN=100

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/m4_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

header() {  # header <logfile> <tag> <command>
  {
    echo "### PROVENANCE ###############################################"
    echo "# tag: $2 | started: $(date -Iseconds) | host: $(hostname)"
    echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "# COMMAND:"; echo "#   $3"
    echo "##############################################################"; echo
  } > "$1"
}

log "=== M4 START (commit $GIT_COMMIT), MAXN=$MAXN ==="
log "диск /data: $(df -h /data | tail -1 | awk '{print $3" занято, "$4" свободно"}')"
log "диск /workdir: $(df -h /workdir | tail -1 | awk '{print $4" свободно"}')"

for d in "$GLOBAL" "$MERGE" "$REGION"; do
  [[ -d "$d" ]] || { log "НЕТ ДАТАСЕТА: $d — сначала scripts/_paper_setup_vm.sh"; exit 1; }
done

# ---- 0. SMOKE: 5 сэмплов, проверить что --no-residual даёт вменяемый t2m ----
log "SMOKE глобальной v2 (ожидаем t2m порядка 1-2 °C, НЕ десятки)"
python -u scripts/predict.py experiments/wb2_512x256_19f_ar_v2 \
  --data-dir "$GLOBAL" --ar-steps 4 --max-samples 5 --per-channel \
  --no-residual --no-save 2>&1 | grep -E "^\s+t2m|skill=" | tail -3

# ---- 1. Глобальная v2 на полном тесте, с сохранением предсказаний ----
TAG=m4_global_v2
LF="$OUT/$TAG.log"; PRED="$HEAVY/m4_global_preds.pt"
CMD="python -u scripts/predict.py experiments/wb2_512x256_19f_ar_v2 --data-dir $GLOBAL --split test_only --ar-steps 4 --max-samples $MAXN --per-channel --no-residual --save $PRED"
header "$LF" "$TAG" "$CMD"
log "START $TAG"
eval "$CMD" >> "$LF" 2>&1
log "DONE  $TAG rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$LF" | tail -1)"

# ---- 2. Интерполяция глобального прогноза в узлы 0.25° + метрики ----
TAG=m4_interp_to_roi
LF="$OUT/$TAG.log"
CMD="python -u scripts/interpolate_to_region.py --predictions $PRED --global-data $GLOBAL --region-data $REGION --per-channel"
header "$LF" "$TAG" "$CMD"
log "START $TAG"
eval "$CMD" >> "$LF" 2>&1
log "DONE  $TAG rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$LF" | tail -1)"

# ---- 3. Региональная GNN standalone (имя чекпойнта с пробелом!) ----
# use_residual этого конфига не подтверждён — сначала смотрим smoke в логе.
TAG=m4_regional_standalone
LF="$OUT/$TAG.log"
CKPT="experiments/region_krsk_cds_19f/best_model (18).pth"
CMD="python -u scripts/predict.py experiments/region_krsk_cds_19f --ckpt \"$CKPT\" --data-dir $REGION --split test_only --ar-steps 4 --max-samples $MAXN --per-channel --no-save --save-sample-metrics $OUT/${TAG}_samples.npz"
header "$LF" "$TAG" "$CMD"
log "START $TAG"
eval "$CMD" >> "$LF" 2>&1
log "DONE  $TAG rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$LF" | tail -1)"

# ---- 4. Флагман на том же окне — контроль, что окно совпало с прошлым прогоном ----
# На MAXN сроках числа будут отличаться от полнотестовых (Skill 70.26 %,
# t2m 1.39/1.66/1.74/1.84) — это нормально, окно другое. Строка нужна как
# внутренний эталон матрицы M4, все строки которой считаются на одних сроках.
TAG=m4_multires_control
LF="$OUT/$TAG.log"
CMD="python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir $MERGE --split test_only --ar-steps 4 --max-samples $MAXN --per-channel --no-save --region $ROI"
header "$LF" "$TAG" "$CMD"
log "START $TAG"
eval "$CMD" >> "$LF" 2>&1
log "DONE  $TAG rc=$? | $(grep -oE 'skill=[-0-9.]+%' "$LF" | tail -1)"

log "=== M4 DONE ==="
grep -E "DONE " "$MASTER" | tail -6
