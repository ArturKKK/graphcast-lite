#!/usr/bin/env bash
# Быстрая проверка, что датасет и модель на месте и считают осмысленное.
#
# Пять сроков, развёртка на 4 шага, область Красноярска. Занимает секунды и
# бережёт часы: полный прогон по 1607 срокам идёт 36 минут, и узнать на 35-й,
# что чекпойнт не подходит к датасету, обидно.
#
# Ожидаемое: ошибка приземной температуры 1-2 °C по горизонтам. Если вышло
# 5-10 °C — не сошлись каналы или порядок узлов; если падает на загрузке
# весов — чекпойнт от другой конфигурации.
#
# Запуск:  bash scripts/_vm_smoke.sh [опыт] [чекпойнт]
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
D33=/data/datasets/multires_krsk_33f
EXP=${1:-multires_krsk_33f}
CK=${2:-/workdir/paper_results/krsk33f_last_epoch.pth}
cd "$REPO" || exit 1

[[ -f "$D33/data.npy" ]]     || { echo "нет датасета $D33 — сначала scripts/_vm_restore33f.sh"; exit 1; }
[[ -d "experiments/$EXP" ]]  || { echo "нет experiments/$EXP"; exit 1; }
[[ -x "$VENV/bin/python" ]]  || { echo "нет venv $VENV"; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"

CK_ARG=""
[[ -f "$CK" ]] && CK_ARG="--ckpt $CK" || echo "чекпойнта $CK нет — беру лучший из опыта"

echo "опыт: $EXP | датасет: $(du -sh "$D33" | cut -f1) | $CK_ARG"
echo
python -u scripts/predict.py "experiments/$EXP" --data-dir "$D33" \
    --split test_only --ar-steps 4 --max-samples 5 \
    --region 50 60 83 98 --per-channel --no-save $CK_ARG
