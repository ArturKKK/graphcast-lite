# Манифест прогонов — graphcast-v3-328274966, 2026-07-25T19:43:16+03:00

## Репозиторий
```
commit: 6d48c3209fe19a66654ff1d3a1dd4cb4dfb423a4
short:  6d48c32
branch: main-arthur
subject: add M1+M5 runner: long-horizon DA (AR-28) + honest OI protocol (val tuning, test with 3 obs seeds) with per-sample metrics
date:    2026-07-25 19:36:38 +0300
--- uncommitted (если есть) ---
?? data/postproc/
?? experiments/neural_postproc_v1_train_artifacts/
?? experiments/neural_postproc_v2/eval_per_station_nsk.json
?? experiments/neural_postproc_v2/eval_per_station_nsk.md
?? run_v2.sh
```

## Окружение
```
gpu:    NVIDIA H100 80GB HBM3, 580.126.16
torch:  2.2.2+cu121 cuda True
python: Python 3.10.20
```

## Датасеты
### multires_krsk_19f_merge  (76G)
```json
{  "time_start": "2010-01-01",  "time_end": "2021-12-31",  "n_time": 16072,  "n_nodes": 133279,  "n_feat": 19,  "flat": true,  "n_global_kept": 130778,  "n_regional": 2501,  "roi": [    50.0,    60.0,    83.0,    98.0  ],  "variables": [    "t2m",    "10u",    "10v",    "msl",    "tp",    "sp",    "tcwv",    "z_surf",    "lsm",    "t@850",    "u@850",    "v@850",    "z@850",    "q@850",    "t@500",    "u@500",    "v@500",    "z@500",    "q@500"  ],  "dtype": "float16",  "file": "data.npy",  "sou

```
### region_krsk_61x41_19f_2010-2020_025deg  (1.5G)
```json
{  "time_start": "2010-01-01",  "time_end": "2020-12-31",  "n_time": 16072,  "n_lon": 61,  "n_lat": 41,  "n_feat": 19,  "variables": [    "t2m",    "10u",    "10v",    "msl",    "tp",    "sp",    "tcwv",    "z_surf",    "lsm",    "t@850",    "u@850",    "v@850",    "z@850",    "q@850",    "t@500",    "u@500",    "v@500",    "z@500",    "q@500"  ],  "dtype": "float16",  "file": "data.npy",  "size_gb": 1.527,  "source": "CDS ERA5 API (regional, 0.25 deg) + time features (sin/cos hour, doy) | slice

```
### region_krsk_61x41_extra_2010-2020_025deg  (767M)
```json
{  "time_start": "2010-01-01",  "time_end": "2020-12-31",  "n_time": 16072,  "n_lon": 61,  "n_lat": 41,  "n_feat_extra": 10,  "variables_extra": [    "z@250",    "t@250",    "u@250",    "v@250",    "q@250",    "z@1000",    "t@1000",    "u@1000",    "v@1000",    "q@1000"  ],  "dtype": "float16",  "file": "data_extra.npy",  "size_gb": 0.749,  "source": "CDS ERA5 reanalysis-era5-pressure-levels (regional, 0.25°)",  "region": {    "lon_min": 83.0,    "lon_max": 98.0,    "lat_min": 50.0,    "lat_max

```
### wb2_512x256_19f_ar  (82G)
```json
{  "time_start": "2010-01-01",  "time_end": "2021-12-31",  "n_time": 17532,  "n_lon": 512,  "n_lat": 256,  "n_feat": 19,  "variables": [    "t2m",    "10u",    "10v",    "msl",    "tp",    "sp",    "tcwv",    "z_surf",    "lsm",    "t@850",    "u@850",    "v@850",    "z@850",    "q@850",    "t@500",    "u@500",    "v@500",    "z@500",    "q@500"  ],  "dtype": "float16",  "file": "data.npy",  "size_gb": 81.3}

```

## Чекпойнты (md5)
```
6ad5ade3178d63776ce79a12069270ad  28M  experiments/multires_merge_freeze6_v2/best_model.pth
c42ba49fa635c12c65fa3eb2e3813b7c  28M  experiments/multires_nores_freeze6/best_model.pth
c4401b4290a467f71dc93dc6d3c2f6a2  74M  experiments/multires_nores_freeze6/checkpoint.pth
bd518528f5b37e4e5f439b679c1da3d9  28M  experiments/multires_nores_nofreeze/best_model.pth
c025da723fa8e014584c67d7653f2709  74M  experiments/multires_nores_nofreeze/checkpoint.pth
173a07aa97c3ea3a021a5f6250b1bb6e  28M  experiments/multires_russia_19f_noroi/best_model.pth
b32fab5395bf178388a6f74aa697ac57  74M  experiments/multires_russia_19f_noroi/checkpoint.pth
3f5d6753a6062061e2ca87997493cdf2  28M  experiments/multires_russia_33f_v3_noroi/best_model.pth
89d5e88f934306e018a50a299d4d0ca8  74M  experiments/multires_russia_33f_v3_noroi/checkpoint.pth
941ca6707fe2a87f331a01d896103e61  92K  experiments/neural_postproc_v1_train_artifacts/best_model.pth
86c4fbaf0424b3aaeaf418ef66681c90  140K  experiments/neural_postproc_v2/best_model.pth
52679c8f5a6ea43674e61951d5c0dca1  508K  experiments/neural_postproc_v3/best_model.pth
8e9ca7c9e51f8e3bc2e7b0b00c2350e9  4.2M  experiments/region_krsk_cds_19f/best_model (18).pth
1a841c3c9420f94be69a23d0dcecc090  28M  experiments/wb2_512x256_19f_ar_v2/best_model.pth
22397f67ef83a01a50f83e035179656a  28M  experiments/wb2_512x256_33f_ar_v3/best_model.pth
```

## Логи прогонов
| файл | размер | изменён | команда (из PROVENANCE) |
|---|---|---|---|
| m1m5_master.log | 4.0K | 19:36 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m1_noda_ar28.log | 8.0K | 19:42 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m2_flagship_inner.log | 12K | 17:03 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_flagship_roi.log | 12K | 15:53 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_freeze6_inner.log | 12K | 17:26 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_freeze6_roi.log | 12K | 16:17 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_master.log | 4.0K | 18:36 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_nofreeze_inner.log | 12K | 17:49 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m2_nofreeze_roi.log | 12K | 16:40 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m3_multires_nores_freeze6_ep16.log | 12K | 18:13 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m3_multires_nores_nofreeze_ep16.log | 12K | 18:36 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |

## Итоговые строки (DONE) из мастер-логов
```
[15:53:29] DONE  m2_flagship_roi rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[16:17:26] DONE  m2_freeze6_roi rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[16:40:23] DONE  m2_nofreeze_roi rc=0 | skill=61.94% |  t2m K 1.36°C 1.65°C 1.81°C 1.96°C
[17:03:11] DONE  m2_flagship_inner rc=0 | skill=71.40% |  t2m K 1.30°C 1.58°C 1.69°C 1.82°C
[17:26:04] DONE  m2_freeze6_inner rc=0 | skill=65.54% |  t2m K 1.23°C 1.47°C 1.57°C 1.66°C
[17:49:19] DONE  m2_nofreeze_inner rc=0 | skill=63.65% |  t2m K 1.28°C 1.55°C 1.70°C 1.83°C
[18:13:12] DONE  m3_multires_nores_freeze6_ep16 rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[18:36:01] DONE  m3_multires_nores_nofreeze_ep16 rc=0 | skill=62.15% |  t2m K 1.35°C 1.62°C 1.72°C 1.82°C
[18:36:01] === ALL DONE ===
[15:53:29] DONE  m2_flagship_roi rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[16:17:26] DONE  m2_freeze6_roi rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[16:40:23] DONE  m2_nofreeze_roi rc=0 | skill=61.94% |  t2m K 1.36°C 1.65°C 1.81°C 1.96°C
[17:03:11] DONE  m2_flagship_inner rc=0 | skill=71.40% |  t2m K 1.30°C 1.58°C 1.69°C 1.82°C
[17:26:04] DONE  m2_freeze6_inner rc=0 | skill=65.54% |  t2m K 1.23°C 1.47°C 1.57°C 1.66°C
[17:49:19] DONE  m2_nofreeze_inner rc=0 | skill=63.65% |  t2m K 1.28°C 1.55°C 1.70°C 1.83°C
[18:13:12] DONE  m3_multires_nores_freeze6_ep16 rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[18:36:01] DONE  m3_multires_nores_nofreeze_ep16 rc=0 | skill=62.15% |  t2m K 1.35°C 1.62°C 1.72°C 1.82°C
```
