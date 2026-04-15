#!/bin/bash
# nores_freeze6: обучался на глобальном 512x256 датасете → тестируем на нём же,
#                метрики по региону через --region (bbox) или is_regional mask
# real_freeze6:  обучался на merge 133K датасете → тестируем на merge
set -e
export PYTHONPATH=/workdir/graphcast-lite
PYTHON=/data/venv/bin/python
GLOBAL=/data/datasets/wb2_512x256_19f_ar
MERGE=/data/datasets/multires_krsk_19f_merge
OUTDIR=/data/merge_eval_results
cd /workdir/graphcast-lite

echo "=== 1/4 nores_freeze6 global+region (on global dataset) ===" | tee $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir $GLOBAL --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 50 60 83 98 > $OUTDIR/nores_region.log 2>&1
echo "DONE 1/4" >> $OUTDIR/status.log

echo "=== 2/4 nores_freeze6 city (on global dataset) ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir $GLOBAL --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 55.5 56.5 92 94 > $OUTDIR/nores_city.log 2>&1
echo "DONE 2/4" >> $OUTDIR/status.log

echo "=== 3/4 real_freeze6 region (on merge dataset) ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual > $OUTDIR/real_region.log 2>&1
echo "DONE 3/4" >> $OUTDIR/status.log

echo "=== 4/4 real_freeze6 city (on merge dataset) ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 55.5 56.5 92 94 > $OUTDIR/real_city.log 2>&1
echo "DONE 4/4 ALL COMPLETE" >> $OUTDIR/status.log
