# Манифест прогонов — graphcast-v3-3994606755, 2026-07-26T23:24:39+03:00

## Репозиторий
```
commit: ffdc74c670328de3637dcc732ce334ea9fdb3b29
short:  ffdc74c
branch: main-arthur
subject: TODO: add v2 vs v3 global comparison plan (commands, disk budget, interpretation caveats)
date:    2026-07-26 15:27:46 +0300
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
torch:  
python: scripts/_paper_collect_logs.sh: line 35: /data/venvs/graphcast/bin/python: No such file or directory
```

## Датасеты
### region_russia_645x165_19f_2010-2021_025deg  (23G)
```json
{  "time_start": "2010-01-01",  "time_end": "2021-12-31",  "n_time": 17532,  "n_lon": 645,  "n_lat": 165,  "n_feat": 19,  "variables": [    "t2m",    "10u",    "10v",    "msl",    "tp",    "sp",    "tcwv",    "z_surf",    "lsm",    "t@850",    "u@850",    "v@850",    "z@850",    "q@850",    "t@500",    "u@500",    "v@500",    "z@500",    "q@500"  ],  "dtype": "float16",  "file": "data.npy",  "size_gb": 66.0,  "bbox": {    "lon_min": 19.0,    "lon_max": 180.0,    "lat_min": 41.0,    "lat_max": 82

```
### region_russia_645x165_extra_2010-2021_025deg  (22G)
```json
{  "time_start": "2010-01-01",  "time_end": "2021-12-31",  "n_time": 17532,  "n_lon": 645,  "n_lat": 165,  "n_feat_extra": 10,  "variables_extra": [    "z@250",    "t@250",    "u@250",    "v@250",    "q@250",    "z@1000",    "t@1000",    "u@1000",    "v@1000",    "q@1000"  ],  "dtype": "float16",  "file": "data_extra.npy",  "size_gb": 34.754,  "source": "CDS ERA5 reanalysis-era5-pressure-levels (regional, 0.25°)",  "region": {    "lon_min": 19.0,    "lon_max": 180.0,    "lat_min": 41.0,    "lat

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
| m1_dipl_noda_ar28.log | 20K | 14:39 | `python -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples ` |
| m1_dipl_oi10_ar28.log | 20K | 15:13 | `python -u scripts/predict.py experiments/multires_nores_freeze6 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples ` |
| m1m5_master.log | 8.0K | 20:13 | `(без header — см. scripts/_paper_run_*.sh на коммите выше)` |
| m1_noda_ar28.log | 20K | 13:03 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m1_oi10_ar28.log | 20K | 13:36 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m1_oi10_first4_ar28.log | 20K | 13:58 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m1_oi1_ar28.log | 20K | 14:20 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-sampl` |
| m5_test_baseline.log | 16K | 16:37 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_nudge_a0.5.log | 16K | 17:01 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_nudge_a0.7.log | 16K | 17:26 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_nudge_a0.9.log | 16K | 17:50 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_oi10_seed42.log | 16K | 18:55 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_oi10_seed43.log | 16K | 19:34 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_oi10_seed44.log | 16K | 20:13 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_test_oi1.log | 16K | 18:17 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-sample` |
| m5_val_oi_L100000_s0.3.log | 12K | 15:33 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L100000_s0.5.log | 12K | 15:38 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L100000_s1.0.log | 12K | 15:43 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L150000_s0.3.log | 12K | 15:48 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L150000_s0.5.log | 12K | 15:53 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L150000_s1.0.log | 12K | 15:58 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L200000_s0.3.log | 12K | 16:03 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L200000_s0.5.log | 12K | 16:08 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L200000_s1.0.log | 12K | 16:13 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L50000_s0.3.log | 12K | 15:18 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L50000_s0.5.log | 12K | 15:23 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |
| m5_val_oi_L50000_s1.0.log | 12K | 15:28 | `python -u scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 ` |

## Итоговые строки (DONE) из мастер-логов
```
[16:08:27] DONE  m5_val_oi_L200000_s0.5 rc=0 | skill=85.53% |  t2m K 0.96°C 1.12°C 1.15°C 1.18°C
[16:13:31] DONE  m5_val_oi_L200000_s1.0 rc=0 | skill=85.19% |  t2m K 0.99°C 1.15°C 1.19°C 1.22°C
[16:13:31] === M5 VAL SWEEP DONE — лучшие (L,sigma) выбираются из логов выше ===
[16:37:16] DONE  m5_test_baseline rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[17:01:51] DONE  m5_test_nudge_a0.5 rc=0 | skill=71.79% |  t2m K 1.34°C 1.59°C 1.66°C 1.75°C
[17:26:11] DONE  m5_test_nudge_a0.7 rc=0 | skill=72.10% |  t2m K 1.33°C 1.58°C 1.65°C 1.74°C
[17:50:30] DONE  m5_test_nudge_a0.9 rc=0 | skill=72.29% |  t2m K 1.32°C 1.57°C 1.64°C 1.73°C
[18:17:01] DONE  m5_test_oi1 rc=0 | skill=77.24% |  t2m K 1.16°C 1.36°C 1.39°C 1.42°C
[18:55:53] DONE  m5_test_oi10_seed42 rc=0 | skill=84.65% |  t2m K 0.82°C 0.93°C 0.93°C 0.94°C
[19:34:43] DONE  m5_test_oi10_seed43 rc=0 | skill=84.89% |  t2m K 0.82°C 0.94°C 0.93°C 0.94°C
[20:13:50] DONE  m5_test_oi10_seed44 rc=0 | skill=85.08% |  t2m K 0.81°C 0.93°C 0.92°C 0.93°C
[20:13:50] === ALL DONE ===
[13:03:24] DONE  m1_noda_ar28 rc=0 | skill=18.86% |  t2m K 1.44°C 1.83°C 1.99°C 2.11°C 2.29°C 2.42°C 2.49°C 2.59°C
[13:36:29] DONE  m1_oi10_ar28 rc=0 | skill=78.31% |  t2m K 0.84°C 1.02°C 1.08°C 1.12°C 1.21°C 1.26°C 1.29°C 1.31°C
[13:58:00] DONE  m1_oi10_first4_ar28 rc=0 | skill=18.69% |  t2m K 0.84°C 1.02°C 1.08°C 1.12°C 1.70°C 2.02°C 2.20°C 2.36°C
[14:20:13] DONE  m1_oi1_ar28 rc=0 | skill=68.88% |  t2m K 1.22°C 1.53°C 1.65°C 1.74°C 1.87°C 1.96°C 2.01°C 2.05°C
[14:39:42] DONE  m1_dipl_noda_ar28 rc=0 | skill=-90.51% |  t2m K 1.40°C 1.77°C 1.97°C 2.12°C 2.35°C 2.49°C 2.67°C 2.83°C
[15:13:04] DONE  m1_dipl_oi10_ar28 rc=0 | skill=-60.44% |  t2m K 0.84°C 1.02°C 1.09°C 1.15°C 1.22°C 1.27°C 1.29°C 1.32°C
[15:18:06] DONE  m5_val_oi_L50000_s0.3 rc=0 | skill=86.13% |  t2m K 0.89°C 1.06°C 1.10°C 1.13°C
[15:23:08] DONE  m5_val_oi_L50000_s0.5 rc=0 | skill=85.76% |  t2m K 0.93°C 1.09°C 1.13°C 1.17°C
[15:28:12] DONE  m5_val_oi_L50000_s1.0 rc=0 | skill=83.83% |  t2m K 1.07°C 1.25°C 1.28°C 1.32°C
[15:33:16] DONE  m5_val_oi_L100000_s0.3 rc=0 | skill=86.12% |  t2m K 0.87°C 1.03°C 1.07°C 1.11°C
[15:38:19] DONE  m5_val_oi_L100000_s0.5 rc=0 | skill=86.55% |  t2m K 0.87°C 1.02°C 1.05°C 1.08°C
[15:43:18] DONE  m5_val_oi_L100000_s1.0 rc=0 | skill=85.98% |  t2m K 0.94°C 1.08°C 1.11°C 1.14°C
[15:48:18] DONE  m5_val_oi_L150000_s0.3 rc=0 | skill=85.86% |  t2m K 0.92°C 1.08°C 1.11°C 1.14°C
[15:53:17] DONE  m5_val_oi_L150000_s0.5 rc=0 | skill=86.06% |  t2m K 0.92°C 1.07°C 1.10°C 1.13°C
[15:58:20] DONE  m5_val_oi_L150000_s1.0 rc=0 | skill=85.70% |  t2m K 0.95°C 1.11°C 1.14°C 1.16°C
[16:03:24] DONE  m5_val_oi_L200000_s0.3 rc=0 | skill=85.45% |  t2m K 0.95°C 1.12°C 1.15°C 1.18°C
[16:08:27] DONE  m5_val_oi_L200000_s0.5 rc=0 | skill=85.53% |  t2m K 0.96°C 1.12°C 1.15°C 1.18°C
[16:13:31] DONE  m5_val_oi_L200000_s1.0 rc=0 | skill=85.19% |  t2m K 0.99°C 1.15°C 1.19°C 1.22°C
[16:13:31] === M5 VAL SWEEP DONE — лучшие (L,sigma) выбираются из логов выше ===
[16:37:16] DONE  m5_test_baseline rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[17:01:51] DONE  m5_test_nudge_a0.5 rc=0 | skill=71.79% |  t2m K 1.34°C 1.59°C 1.66°C 1.75°C
[17:26:11] DONE  m5_test_nudge_a0.7 rc=0 | skill=72.10% |  t2m K 1.33°C 1.58°C 1.65°C 1.74°C
[17:50:30] DONE  m5_test_nudge_a0.9 rc=0 | skill=72.29% |  t2m K 1.32°C 1.57°C 1.64°C 1.73°C
[18:17:01] DONE  m5_test_oi1 rc=0 | skill=77.24% |  t2m K 1.16°C 1.36°C 1.39°C 1.42°C
[18:55:53] DONE  m5_test_oi10_seed42 rc=0 | skill=84.65% |  t2m K 0.82°C 0.93°C 0.93°C 0.94°C
[19:34:43] DONE  m5_test_oi10_seed43 rc=0 | skill=84.89% |  t2m K 0.82°C 0.94°C 0.93°C 0.94°C
[20:13:50] DONE  m5_test_oi10_seed44 rc=0 | skill=85.08% |  t2m K 0.81°C 0.93°C 0.92°C 0.93°C
[20:13:50] === ALL DONE ===
```
