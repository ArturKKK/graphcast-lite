#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Build Russia merge multires dataset:
#   1. Download fine 0.25° Russia region from WB2 1440x721
#   2. Merge with global coarse 0.7° → multires (real fine + coarse outside)
#
# Использование:
#   bash scripts/setup_russia_merge.sh
#   (поправь переменные ниже под свою машину/период)
#
# Локально на ноуте 34 GB свободно — НЕ ВЛЕЗЕТ для 12 лет.
# Рекомендации:
#   - 5 лет (~28 GB)   → START_YEAR=2017 END_YEAR=2022
#   - VM (есть место)  → START_YEAR=2010 END_YEAR=2022 (~66 GB)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Конфиг ──────────────────────────────────────────────────────
PY="${PY:-.venv/bin/python}"

START_YEAR="${START_YEAR:-2017}"
END_YEAR="${END_YEAR:-2022}"      # exclusive

LON_MIN="${LON_MIN:-19.0}"
LON_MAX="${LON_MAX:-180.0}"
LAT_MIN="${LAT_MIN:-41.0}"
LAT_MAX="${LAT_MAX:-82.0}"

GLOBAL_DIR="${GLOBAL_DIR:-data/datasets/global_512x256_19f_2010-2021_07deg}"

REGION_DIR="${REGION_DIR:-data/datasets/region_russia_645x165_19f_${START_YEAR}-$((END_YEAR-1))_025deg}"
MULTIRES_DIR="${MULTIRES_DIR:-data/datasets/multires_russia_19f_merge_${START_YEAR}-$((END_YEAR-1))}"

LOG="${LOG:-russia_merge_setup.log}"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Russia merge multires setup ==="
log "  Period:    ${START_YEAR}-$((END_YEAR-1))"
log "  Bbox:      lon [${LON_MIN},${LON_MAX}], lat [${LAT_MIN},${LAT_MAX}]"
log "  Global:    $GLOBAL_DIR"
log "  Region:    $REGION_DIR"
log "  Multires:  $MULTIRES_DIR"

# ── 0. Проверки ─────────────────────────────────────────────────
if [ ! -f "$GLOBAL_DIR/dataset_info.json" ]; then
    log "ERROR: $GLOBAL_DIR/dataset_info.json не найден"
    exit 1
fi
if [ ! -f "$PY" ]; then
    log "ERROR: python не найден: $PY"
    exit 1
fi

# ── 1. Скачать мелкую сетку России ─────────────────────────────
if [ -f "$REGION_DIR/dataset_info.json" ] && [ ! -f "$REGION_DIR/progress.json" ]; then
    log "SKIP region download (уже готов): $REGION_DIR"
else
    log "=== STEP 1: Download fine 0.25° Russia ==="
    RESUME_FLAG=""
    if [ -f "$REGION_DIR/progress.json" ]; then
        log "  resume detected"
        RESUME_FLAG="--resume"
    fi
    $PY -u scripts/build_region_russia_19f.py \
        --out-dir "$REGION_DIR" \
        --start-year "$START_YEAR" --end-year "$END_YEAR" \
        --lon-min "$LON_MIN" --lon-max "$LON_MAX" \
        --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" \
        --static-from "$GLOBAL_DIR" \
        $RESUME_FLAG 2>&1 | tee -a "$LOG"
    log "DONE region download"
fi

# ── 2. Собрать multires merge ──────────────────────────────────
if [ -f "$MULTIRES_DIR/data.npy" ]; then
    log "SKIP multires merge (уже готов): $MULTIRES_DIR"
else
    log "=== STEP 2: Build multires merge ==="
    $PY -u scripts/build_multires_dataset.py \
        --global-dir "$GLOBAL_DIR" \
        --region-dir "$REGION_DIR" \
        --roi "$LAT_MIN" "$LAT_MAX" "$LON_MIN" "$LON_MAX" \
        --mode merge \
        --out-dir "$MULTIRES_DIR" 2>&1 | tee -a "$LOG"
    log "DONE multires merge"
fi

log ""
log "=== Готово ==="
log "  Region:    $REGION_DIR"
log "  Multires:  $MULTIRES_DIR"
log ""
log "Дальше — обучить fine-tune от multires_russia_19f:"
log "  $PY -u src/main.py experiments/multires_russia_19f_merge \\"
log "      --pretrained experiments/multires_russia_19f/best_model.pth"
