#!/usr/bin/env bash
# Rsync data_extra.npy + metadata from city.sscc.ru to local datasets dir.
# Scheduled run: executed with a 90-min delay from schedule_rsync_extra.sh.
set -euo pipefail

LOG="/tmp/rsync_extra_$(date +%Y%m%d_%H%M%S).log"
DEST="/Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/global_512x256_extra_2010-2021_07deg"
SRC="tabakov_2026@city.sscc.ru:/home/tabakov_2026/graphcast-russia/data/global_512x256_19f_2010-2021_07deg/"
KEY="/Users/a.s.tabakov/Developer/graphcast-lite/.ssh-city/id_city"
PROXY="nc -X 5 -x 192.168.1.1:1080 %h %p"

mkdir -p "$DEST"

echo "[$(date)] === rsync_extra_from_city START ===" | tee -a "$LOG"
echo "[$(date)] DEST: $DEST" | tee -a "$LOG"
df -h "$DEST" 2>/dev/null | tee -a "$LOG"

rsync -av --progress \
  -e "ssh -o ProxyCommand=\"$PROXY\" -i \"$KEY\"" \
  "$SRC" \
  "$DEST/" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] === rsync DONE (exit $?) ===" | tee -a "$LOG"
echo "Log saved to: $LOG"
