#!/bin/bash
# GraphCast-lite: full forecast pipeline (cron every 6h)
# Crontab: 0 1,7,13,19 * * * /opt/graphcast-lite/website/cron_forecast.sh >> /var/log/graphcast-forecast.log 2>&1
set -euo pipefail

BASEDIR=/opt/graphcast-lite
VENV=/opt/graphcast-venv/bin/python
RESULTS=$BASEDIR/results
OUT=$RESULTS/live_latest
CACHE=/tmp/gdas_cache

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
echo "[1/5] Disk check..."
df -h / | tail -1

# ── Rotation: keep only 2 latest forecasts ──
echo "[2/5] Rotating forecasts..."
rm -rf "$RESULTS/live_old"
[ -d "$RESULTS/live_previous" ] && mv "$RESULTS/live_previous" "$RESULTS/live_old"
[ -d "$OUT" ] && mv "$OUT" "$RESULTS/live_previous"

# ── Run forecast ──
echo "[3/5] Running forecast..."
cd "$BASEDIR"
$VENV scripts/live_gdas_forecast.py \
  --experiment-dir experiments/multires_nores_freeze6 \
  --runtime-bundle live_runtime_bundle \
  --learned-mos live_runtime_bundle/learned_mos_t2m_wind_19st.joblib \
  --wind-scale live_runtime_bundle/wind_monthly_scale.json \
  --spatial-idw \
  --selective \
  --lapse-target-elevation 287 \
  --ar-steps 12 \
  --out-dir "$OUT" \
  --cache-dir "$CACHE"

# ── Cleanup GDAS cache ──
echo "[4/5] Cleaning cache..."
rm -rf "$CACHE"

# ── Delete oldest forecast ──
rm -rf "$RESULTS/live_old"

# ── Parse forecast.pt → forecast.json ──
echo "[5/5] Generating forecast.json..."
$VENV "$BASEDIR/website/forecast_parser.py" \
  --input "$OUT/forecast.pt" \
  --output "$BASEDIR/website/static/forecast.json"

echo "[DONE] Pipeline complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
df -h / | tail -1
