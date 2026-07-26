# Манифест прогонов — graphcast-v3-750057806, 2026-07-26T13:43:34+03:00

## Репозиторий
```
commit: d62475f8e7b62c79ab2c5458a86a3cfa324a8f00
short:  d62475f
branch: main-arthur
subject: paper scripts: write logs/results to /workdir (persistent), not /data (wiped on VM restart)
date:    2026-07-26 12:24:17 +0300
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
| m1m5_master.log | 4.0K | 13:36 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m1_noda_ar28.log | 20K | 13:03 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m1_oi10_ar28.log | 20K | 13:36 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m1_oi10_first4_ar28.log | 8.0K | 13:42 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |

## Итоговые строки (DONE) из мастер-логов
```
[13:03:24] DONE  m1_noda_ar28 rc=0 | skill=18.86% |  t2m K 1.44°C 1.83°C 1.99°C 2.11°C 2.29°C 2.42°C 2.49°C 2.59°C
[13:36:29] DONE  m1_oi10_ar28 rc=0 | skill=78.31% |  t2m K 0.84°C 1.02°C 1.08°C 1.12°C 1.21°C 1.26°C 1.29°C 1.31°C
```
