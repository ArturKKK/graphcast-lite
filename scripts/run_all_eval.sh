#!/bin/bash
set -e
export PYTHONPATH=/workdir/graphcast-lite
PYTHON=/data/venv/bin/python
MERGE=/data/datasets/multires_krsk_19f_merge
OUTDIR=/data/merge_eval_results
cd /workdir/graphcast-lite

echo "=== 1/4 nores_freeze6 region ===" | tee $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual > $OUTDIR/nores_region.log 2>&1
echo "DONE 1/4" >> $OUTDIR/status.log

echo "=== 2/4 nores_freeze6 city ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 55.5 56.5 92 94 > $OUTDIR/nores_city.log 2>&1
echo "DONE 2/4" >> $OUTDIR/status.log

echo "=== 3/4 real_freeze6 region ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual > $OUTDIR/real_region.log 2>&1
echo "DONE 3/4" >> $OUTDIR/status.log

echo "=== 4/4 real_freeze6 city ===" | tee -a $OUTDIR/status.log
$PYTHON -u scripts/predict.py experiments/multires_real_freeze6 --data-dir $MERGE --max-samples 200 --ar-steps 4 --per-channel --no-residual --region 55.5 56.5 92 94 > $OUTDIR/real_city.log 2>&1
echo "DONE 4/4 ALL COMPLETE" >> $OUTDIR/status.log
