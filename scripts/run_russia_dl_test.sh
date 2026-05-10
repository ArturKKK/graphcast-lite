#!/bin/bash
set -e
cd /Users/a.s.tabakov/Developer/graphcast-lite
source .venv/bin/activate
export ALL_PROXY='socks5h://192.168.1.1:1080'
export HTTP_PROXY="$ALL_PROXY"
export HTTPS_PROXY="$ALL_PROXY"
mkdir -p logs
exec python -u scripts/build_region_russia_19f.py \
  --out-dir data/datasets/region_russia_645x165_19f_2010-2021_025deg \
  --start-year 2010 --end-year 2022 \
  --lon-min 19 --lon-max 180 --lat-min 41 --lat-max 82 \
  --time-chunk 200 \
  --resume
