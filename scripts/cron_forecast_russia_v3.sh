#!/bin/bash
# Russia v3 production pipeline on city.sscc.ru (every 6h)
#
# Stage 1: 5-day GDAS-initialized inference with multires_russia_33f_v3_noroi
# Stage 2: neural_postproc_v3 applied at 689 stations
# Stage 3: build russia_forecast.json for the website
#
# Outputs:
#   ~/graphcast-lite/results/live_russia_33f_5d/forecast.pt
#   ~/graphcast-lite/results/live_russia_33f_5d/station_postproc_v3.json
#   ~/graphcast-lite/website/static/russia_forecast.json
set -euo pipefail

BASEDIR=$HOME/graphcast-lite
VENV=$HOME/graphcast-russia/venv/bin/python
RESULTS=$BASEDIR/results/live_russia_33f_5d
CACHE=$HOME/tmp/gdas_cache_russia
mkdir -p "$RESULTS" "$CACHE" "$BASEDIR/website/static"
cd "$BASEDIR"

echo "===== RUSSIA v3 PIPELINE $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
echo "[1/4] Disk + memory..."
df -h "$BASEDIR" | tail -1
free -h | head -2

# ── Stage 1: inference (5-day, 20 AR steps)
# Retry the GDAS download up to 3 times — NOAA NOMADS occasionally times out.
# `--max-lookback-cycles 4` lets the script fall back to a 24h-older cycle if the
# very latest one is missing on the server (common during the 3-4h posting lag).
echo "[2/4] GDAS → forecast.pt (v3 33f, 5 days)..."
INFER_OK=0
for attempt in 1 2 3; do
    if $VENV scripts/live_gdas_forecast.py \
            --experiment-dir experiments/multires_russia_33f_v3_noroi \
            --runtime-bundle live_runtime_bundle_russia_33f \
            --ar-steps 20 \
            --selective \
            --timeout 300 \
            --max-lookback-cycles 4 \
            --out-dir "$RESULTS" \
            --cache-dir "$CACHE"; then
        INFER_OK=1
        echo "  inference succeeded on attempt $attempt"
        break
    else
        echo "  inference attempt $attempt failed; cleaning cache and waiting before retry..."
        rm -rf "$CACHE"
        sleep 180
    fi
done

# Cleanup GDAS cache after inference (large grib2 files)
rm -rf "$CACHE"

if [ "$INFER_OK" -ne 1 ]; then
    echo "[ABORT] All inference attempts failed — keeping the previous russia_forecast.json on the site."
    exit 1
fi

# ── Stage 2: neural_postproc_v3 (689 stations)
echo "[3/4] neural_postproc_v3 (689 stations)..."
$VENV scripts/postproc_apply_v3_russia.py \
    --forecast "$RESULTS/forecast.pt" \
    --stations data/russia_mos_stations.json \
    --postproc-dir experiments/neural_postproc_v3 \
    --out "$RESULTS/station_postproc_v3.json"

# ── Stage 3: build site-facing russia_forecast.json
echo "[4/4] Build russia_forecast.json..."
$VENV scripts/build_russia_forecast_json.py \
    --forecast "$RESULTS/forecast.pt" \
    --postproc-json "$RESULTS/station_postproc_v3.json" \
    --out "$BASEDIR/website/static/russia_forecast.json"

ls -la "$BASEDIR/website/static/russia_forecast.json"

# ── Cleanup huge intermediate forecast.pt (1.8 GB) — keep only the final JSON
echo "[5/5] Cleaning intermediate tensors..."
rm -f "$RESULTS/forecast.pt" "$RESULTS/input_normalized.npy"

# ── Stage 6: spatial overlays for the map
echo "[6/6] gen_overlays for Russia map..."
$VENV scripts/gen_overlays.py --input "$BASEDIR/website/static/russia_forecast.json" --output-dir "$BASEDIR/website/static/overlays_russia"

echo "[DONE] Russia v3 pipeline complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
df -h "$BASEDIR" | tail -1
