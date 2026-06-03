#!/bin/bash
# Watchdog — runs frequently and only kicks the full pipeline if the latest
# russia_forecast.json on the site is stale (initial cycle older than
# STALE_THRESHOLD_HOURS). Intended as a safety net for transient NOAA NOMADS
# outages that knock out the scheduled 6-hour cron.
#
# Suggested crontab entry (every 2 hours, offset by 1h from the main cron):
#   0 3,9,15,21 * * * /home/tabakov_2026/graphcast-lite/scripts/cron_forecast_russia_v3_watchdog.sh \
#       >> /home/tabakov_2026/graphcast-lite/logs/watchdog.log 2>&1
set -euo pipefail

BASEDIR=$HOME/graphcast-lite
FORECAST_JSON=$BASEDIR/website/static/russia_forecast.json
STALE_THRESHOLD_HOURS=9

if [ ! -f "$FORECAST_JSON" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] watchdog: no russia_forecast.json yet — kicking pipeline"
    exec "$BASEDIR/cron_forecast_russia_v3.sh"
fi

# russia_forecast.json exposes the initial cycle as `last_cycle` (ISO 8601 UTC).
LAST_CYCLE=$(python3 -c "
import json, sys
try:
    d = json.load(open('$FORECAST_JSON'))
    print(d.get('last_cycle') or d.get('init_cycle') or d.get('cycle') or '')
except Exception as e:
    sys.stderr.write(f'parse error: {e}\n')
")

if [ -z "$LAST_CYCLE" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] watchdog: cannot read last init cycle from JSON — kicking pipeline"
    exec "$BASEDIR/cron_forecast_russia_v3.sh"
fi

AGE_HOURS=$(python3 -c "
from datetime import datetime, timezone
t = datetime.fromisoformat('$LAST_CYCLE'.replace('Z','+00:00'))
delta = datetime.now(timezone.utc) - t
print(int(delta.total_seconds() // 3600))
")

if [ "$AGE_HOURS" -ge "$STALE_THRESHOLD_HOURS" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] watchdog: last cycle ${LAST_CYCLE} is ${AGE_HOURS}h old (>= ${STALE_THRESHOLD_HOURS}h) — kicking pipeline"
    exec "$BASEDIR/cron_forecast_russia_v3.sh"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] watchdog: forecast is fresh (${AGE_HOURS}h old) — nothing to do"
fi
