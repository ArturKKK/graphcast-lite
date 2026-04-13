#!/bin/bash
# DA v3 verification: reproduce key v2 experiments on new dataset + MOS sweep
set -e
cd /workdir/graphcast-lite
export LD_LIBRARY_PATH=/home/mlcore/conda/lib:$LD_LIBRARY_PATH
PY=".venv/bin/python -u"
EXP="experiments/multires_nores_freeze6"
BF="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42"

echo "========================================"
echo "DA v3 verification + MOS sweep at $(date)"
echo "========================================"

# 1) No-DA baseline with same 200 samples
echo "--- [1/7] Baseline (no DA, 200 samples) ---"
$PY scripts/predict.py $EXP $BF > /data/v3_baseline_200.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_baseline_200.log

# 2) OI 10% c=50km (was 82.13% in v2)
echo "--- [2/7] OI 10% c=50km ---"
$PY scripts/predict.py $EXP $BF --obs-sparsity 0.1 --assim-method oi --oi-corr-len 50000 --oi-sigma-o 0.5 > /data/v3_oi_s01_c50000_sig0.5.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_oi_s01_c50000_sig0.5.log

# 3) OI 10% c=100km (was 83.79% in v2 — the best!)
echo "--- [3/7] OI 10% c=100km ---"
$PY scripts/predict.py $EXP $BF --obs-sparsity 0.1 --assim-method oi --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/v3_oi_s01_c100000_sig0.5.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_oi_s01_c100000_sig0.5.log

# 4) OI 10% c=150km (was 83.39% in v2)
echo "--- [4/7] OI 10% c=150km ---"
$PY scripts/predict.py $EXP $BF --obs-sparsity 0.1 --assim-method oi --oi-corr-len 150000 --oi-sigma-o 0.5 > /data/v3_oi_s01_c150000_sig0.5.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_oi_s01_c150000_sig0.5.log

# 5) OI 1% c=100km (was 75.05% in v2)
echo "--- [5/7] OI 1% c=100km ---"
$PY scripts/predict.py $EXP $BF --obs-sparsity 0.01 --assim-method oi --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/v3_oi_s001_c100000_sig0.5.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_oi_s001_c100000_sig0.5.log

# 6) OI 1% c=150km (was 75.84% in v2)
echo "--- [6/7] OI 1% c=150km ---"
$PY scripts/predict.py $EXP $BF --obs-sparsity 0.01 --assim-method oi --oi-corr-len 150000 --oi-sigma-o 0.5 > /data/v3_oi_s001_c150000_sig0.5.log 2>&1
grep -A5 "Per-horizon (region)" /data/v3_oi_s001_c150000_sig0.5.log

# 7) MOS/IDW parameter sweep
echo "--- [7/7] MOS/IDW sweep (50 samples) ---"
$PY scripts/mos_idw_sweep.py --max-samples 50 --ar-steps 4 > /data/v3_mos_sweep.log 2>&1
tail -60 /data/v3_mos_sweep.log

echo "========================================"
echo "All verification + MOS experiments done at $(date)"
echo "========================================"
