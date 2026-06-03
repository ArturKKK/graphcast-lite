#!/bin/bash
# GraphCast-lite: Krasnoyarsk forecast pipeline (cron every 6h)
# Crontab (city.sscc.ru, user tabakov_2026):
#   0 1,7,13,19 * * * $HOME/graphcast-lite/website/cron_forecast.sh \
#     >> $HOME/graphcast-lite/logs/cron_krsk.log 2>&1
set -euo pipefail

# Allow overriding paths via env vars; defaults work on city.sscc.ru.
BASEDIR=${BASEDIR:-$HOME/graphcast-lite}
VENV=${VENV:-$HOME/graphcast-russia/venv/bin/python}
RESULTS=${RESULTS:-$BASEDIR/results}
OUT=${OUT:-$RESULTS/live_latest}
CACHE=${CACHE:-$HOME/tmp/gdas_cache}
mkdir -p "$RESULTS" "$BASEDIR/website/static" "$(dirname "$CACHE")"

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
echo "[1/5] Disk check..."
df -h / | tail -1

# ── Rotation: keep only 2 latest forecasts ──
echo "[2/5] Rotating forecasts..."
rm -rf "$RESULTS/live_old"
[ -d "$RESULTS/live_previous" ] && mv "$RESULTS/live_previous" "$RESULTS/live_old"
[ -d "$OUT" ] && mv "$OUT" "$RESULTS/live_previous"

# ── Run forecast (retry up to 3x on NOMADS failures) ──
echo "[3/5] Running forecast..."
cd "$BASEDIR"
INFER_OK=0
for attempt in 1 2 3; do
    if $VENV scripts/live_gdas_forecast.py \
            --experiment-dir experiments/multires_nores_freeze6 \
            --runtime-bundle live_runtime_bundle \
            --learned-mos live_runtime_bundle/learned_mos_t2m_wind_19st.joblib \
            --wind-scale live_runtime_bundle/wind_monthly_scale.json \
            --spatial-idw \
            --selective \
            --timeout 300 \
            --max-lookback-cycles 4 \
            --lapse-target-elevation 287 \
            --ar-steps 12 \
            --out-dir "$OUT" \
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

if [ "$INFER_OK" -ne 1 ]; then
    echo "[ABORT] All inference attempts failed — keeping the previous forecast.json on the site."
    rm -rf "$CACHE"
    exit 1
fi

# ── Cleanup GDAS cache ──
echo "[4/5] Cleaning cache..."
rm -rf "$CACHE"

# ── Delete oldest forecast ──
rm -rf "$RESULTS/live_old"

# ── Parse forecast.pt → forecast.json ──
echo "[5/6] Generating forecast.json..."
$VENV "$BASEDIR/website/forecast_parser.py" \
  --input "$OUT/forecast.pt" \
  --output "$BASEDIR/website/static/forecast.json"

# ── Spatial overlays for the map (temp, wind, precip, pressure tiles) ──
echo "[6/6] Generating overlays..."
$VENV "$BASEDIR/scripts/gen_overlays.py" \
  --input "$BASEDIR/website/static/forecast.json" \
  --output-dir "$BASEDIR/website/static/overlays"

echo "[DONE] Pipeline complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
df -h / | tail -1
