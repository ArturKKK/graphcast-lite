# Presentation Context

## Purpose

This file stores the current working context for the Krasnoyarsk presentation-refresh workflow so future sessions can recover state quickly.

Current goal:
- rebuild and extend the presentation artifacts for the diploma/presentation around GraphCast-lite over Krasnoyarsk;
- regenerate comparison plots and summary assets from already trained models;
- include longer-horizon offline data assimilation and a separate global baseline block;
- keep heavy artifacts under `/data` so the VM/workdir disk is not exhausted.

## What We Are Doing Right Now

We are running / preparing to run `scripts/refresh_presentation_artifacts.py` to rebuild the presentation outputs.

This run is intended to produce artifacts for a refreshed presentation with:
- multires model comparison: `multires_nores_freeze6` vs `multires_nores_nofreeze`;
- offline DA on a longer horizon: 9 days (`--ar-steps 36`);
- separate global baseline evaluation for 3 days (`--global-ar-steps 12`);
- optional WRF/Jan-2023 comparison if the January dataset and WRF JSON are present;
- optional staging of already-computed live 3-day outputs, but live mode is not rerun inside the batch script by default.

## Why This Presentation Exists

The presentation is meant to show:
- that the multires Krasnoyarsk model improves local forecast quality relative to the plain global baseline;
- that `freeze6` is the main winning multires variant, while `nofreeze` is kept as an ablation / comparison;
- that offline DA gives clearer benefit on longer horizons than on the original short 24h setup;
- that the project has both regional high-resolution value and an interpretable evaluation pipeline.

## Current Best-Known Model Choices

Main regional/multires models:
- `experiments/multires_nores_freeze6` — main winner / preferred baseline for slides.
- `experiments/multires_nores_nofreeze` — keep for comparison in slides.

Global baseline for the presentation refresh script:
- `experiments/wb2_512x256_19f_ar_v2`

Important note:
- the old `64x32` baseline was mistakenly wired into `refresh_presentation_artifacts.py` during one edit;
- this was fixed, and the script now defaults to the `512x256 v2` global baseline instead.

## Key Datasets And Paths

Main multires dataset:
- `/data/datasets/multires_krsk_19f`

Global baseline dataset:
- `/data/datasets/global_512x256_19f_2010-2021_07deg`

Expected Jan-2023 multires dataset for WRF/presentation comparison:
- `/data/datasets/multires_krsk_19f_jan2023_interp`

Expected WRF JSON:
- `/workdir/graphcast-lite/aaaa/wrf_krasnoyarsk/wrf_d03_jan2023.json`

Heavy artifact store:
- `/data/graphcast-lite/presentation_cache/presentation_refresh_9day/artifacts`

Main output directory for the refreshed presentation run:
- `results/presentation_refresh_9day`

## Command We Intend To Run

Minimal current command:

```bash
python scripts/refresh_presentation_artifacts.py \
  --main-data /data/datasets/multires_krsk_19f \
  --out-dir results/presentation_refresh_9day \
  --artifact-store-dir /data/graphcast-lite/presentation_cache/presentation_refresh_9day/artifacts \
  --ar-steps 36 \
  --artifact-samples 4 \
  --global-ar-steps 12 \
  --global-max-samples 200
```

Full command if Jan-2023 and WRF inputs are available:

```bash
python scripts/refresh_presentation_artifacts.py \
  --main-data /data/datasets/multires_krsk_19f \
  --jan-data /data/datasets/multires_krsk_19f_jan2023_interp \
  --wrf-json /workdir/graphcast-lite/aaaa/wrf_krasnoyarsk/wrf_d03_jan2023.json \
  --out-dir results/presentation_refresh_9day \
  --artifact-store-dir /data/graphcast-lite/presentation_cache/presentation_refresh_9day/artifacts \
  --ar-steps 36 \
  --artifact-samples 4 \
  --global-ar-steps 12 \
  --global-max-samples 200
```

## Script Behavior That Matters

`scripts/refresh_presentation_artifacts.py` now:
- defaults to `wb2_512x256_19f_ar_v2` as the global baseline;
- resolves relative `data/datasets/...` paths via `/data` when those datasets do not exist inside the repo checkout;
- does not reuse old outputs unless `--reuse-existing` is explicitly passed;
- does not stage live outputs unless `--include-live` is explicitly passed;
- writes large `.pt` bundles into the external artifact store and symlinks them into the output tree.

## Live Forecast Context

There is a separate already-computed live 3-day run under:
- `results/live_gdas_run_3day`

Important facts from earlier debugging:
- Open-Meteo comparison was matched in UTC, not in Krasnoyarsk local time;
- the local-time column in the comparison table is display-only for the same UTC instant;
- live-mode rerun is intentionally not part of the batch presentation refresh by default.

## Current Branch And Relevant Fixes

Working branch:
- `main-arthur`

Relevant recent commits:
- `33f0115` — extend presentation refresh pipeline;
- `2bfc68b` — fix global baseline defaults for presentation refresh (switch from old 64x32 to `wb2_512x256_19f_ar_v2`, add `/data` dataset fallback).

## Current Open Questions

1. Whether `/data/datasets/multires_krsk_19f_jan2023_interp` exists on the cluster.
2. Whether `/workdir/graphcast-lite/aaaa/wrf_krasnoyarsk/wrf_d03_jan2023.json` exists on the cluster.
3. How well the live forecasts match local station observations in the current cold/warm event.

## Operational Interpretation For City Temperatures

If local station observations show around `+3 C` and our live city forecast is materially colder, then for that specific city-surface operational case:
- Yandex / IBM is currently doing better;
- our forecast is missing something important for that event.

But that does not automatically prove the entire model is useless, because point-station values and grid-cell forecasts are not the same object. A proper conclusion requires comparison of:
- the same valid UTC time;
- the same location / station or a defensible city aggregate;
- archived observations rather than a single portal snapshot;
- enough cases, not just one event.

Still, if several trusted station posts consistently say `+3 C` and our model is well below that, the honest operational takeaway is that our current live product is not yet reliable enough as a city-temperature service for that situation.