# k8smr-macos-build--graphcast-v4-3510779436-task-0  —  2026-08-06T11:01:48+00:00

## Железо и диски
```
NVIDIA H100 80GB HBM3, 81559 MiB
Filesystem                                                                                        Size  Used Avail Use% Mounted on
a802a1-ml-m5-vs0.storage.tcsbank.ru:/jupyter_persistent_storage_m5_sm12/macos-build-infra/452042  1.0T  498G  527G  49% /workdir
overlay                                                                                           8.7T  5.4T  2.9T  65% /
227G	/data/datasets
```

## Git
```
290ea46 add v2 vs v3 global comparison runner (venv+unpack+build 33f dataset+both runs, disk-safe)
?? experiments/multires_krsk_33f/training_log.txt
?? experiments/wb2_512x256_33f_ar_v3/config.json.bak
```

## Датасеты
```
--- archives  692M
bash: /data/datasets/archives/dataset_info*.json: No such file or directory

--- global_512x256_extra_2010-2021_07deg  43G
bash: "$d"dataset_info*.json: ambiguous redirect

--- invest  5.4G
bash: /data/datasets/invest/dataset_info*.json: No such file or directory

--- isd_lite_russia  215M
bash: /data/datasets/isd_lite_russia/dataset_info*.json: No such file or directory

--- krasnoyarks-nedelya  133M
bash: /data/datasets/krasnoyarks-nedelya/dataset_info*.json: No such file or directory

--- krasnoyars-1440  5.5M
bash: /data/datasets/krasnoyars-1440/dataset_info*.json: No such file or directory

--- region_russia_645x165_19f_2010-2021_025deg  67G
{"time_start":"2010-01-01","time_end":"2021-12-31","n_time":17532,"n_lon":645,"n_lat":165,"n_feat":19,"variables":["t2m","10u","10v","msl","tp","sp","tcwv","z_surf","lsm","t@850","u@850","v@850","z@850","q@850","t@500","u@500","v@500","z@500","q@500"],"dtype":

--- region_russia_645x165_extra_2010-2021_025deg  35G
{"time_start":"2010-01-01","time_end":"2021-12-31","n_time":17532,"n_lon":645,"n_lat":165,"n_feat_extra":10,"variables_extra":["z@250","t@250","u@250","v@250","q@250","z@1000","t@1000","u@1000","v@1000","q@1000"],"dtype":"float16","file":"data_extra.npy","size

```

## Чекпойнты
```
experiments/multires_krsk_33f/best_model.pth	27828 KB
experiments/multires_krsk_33f/checkpoint.pth	75276 KB
experiments/multires_merge_freeze6_v2/best_model.pth	27796 KB
experiments/multires_nores_freeze6/best_model.pth	27796 KB
experiments/multires_nores_freeze6/checkpoint.pth	75172 KB
experiments/multires_nores_nofreeze/best_model.pth	27796 KB
experiments/multires_nores_nofreeze/checkpoint.pth	75172 KB
experiments/multires_russia_33f_v3_noroi/best_model.pth	27828 KB
experiments/multires_russia_33f_v3_noroi/checkpoint.pth	75276 KB
experiments/region_krsk_cds_19f/best_model (18).pth	4212 KB
experiments/wb2_512x256_19f_ar_v2/best_model.pth	27796 KB
experiments/wb2_512x256_33f_ar_v3/best_model.pth	27828 KB
experiments/wb2_512x256_33f_ar_v3/checkpoint.pth	75276 KB
```

## Результаты
```
=== /workdir/paper_results
total 48952
drwxr-xr-x 2 mlcore mlcore     4096 Jul 27 15:15 .
drwxrwxrwx 6 root   root       4096 Aug  6 10:54 ..
-rw-r--r-- 1 mlcore mlcore 28375319 Jul 27 11:01 krsk33f_last_epoch.pth
-rw-r--r-- 1 mlcore mlcore    12330 Jul 27 14:39 m19_flagship_roi.log
-rw-r--r-- 1 mlcore mlcore  2028992 Jul 27 14:39 m19_flagship_roi_samples.npz
-rw-r--r-- 1 mlcore mlcore    12336 Jul 27 14:13 m19_freeze6_inner.log
-rw-r--r-- 1 mlcore mlcore  2025561 Jul 27 14:13 m19_freeze6_inner_samples.npz
-rw-r--r-- 1 mlcore mlcore    12330 Jul 27 13:48 m19_freeze6_roi.log
-rw-r--r-- 1 mlcore mlcore  2020474 Jul 27 13:48 m19_freeze6_roi_samples.npz
-rw-r--r-- 1 mlcore mlcore    14540 Jul 27 12:46 m33_best_inner.log
-rw-r--r-- 1 mlcore mlcore  3148588 Jul 27 12:46 m33_best_inner_samples.npz
-rw-r--r-- 1 mlcore mlcore    14534 Jul 27 11:36 m33_best_roi.log
-rw-r--r-- 1 mlcore mlcore  3132877 Jul 27 11:36 m33_best_roi_samples.npz
-rw-r--r-- 1 mlcore mlcore    21979 Jul 27 15:15 m33_last_ar28.log
-rw-r--r-- 1 mlcore mlcore  2729619 Jul 27 15:15 m33_last_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore    14601 Jul 27 13:22 m33_last_inner.log
-rw-r--r-- 1 mlcore mlcore  3147919 Jul 27 13:22 m33_last_inner_samples.npz
-rw-r--r-- 1 mlcore mlcore    14595 Jul 27 12:12 m33_last_roi.log
-rw-r--r-- 1 mlcore mlcore  3132092 Jul 27 12:12 m33_last_roi_samples.npz
-rw-r--r-- 1 mlcore mlcore     1932 Jul 27 15:15 m33eval_master.log
    +138h: RMSE=0.398378 | base=0.684749 | skill=41.82% | ACC=0.6015 (base 0.3175)
    +144h: RMSE=0.416428 | base=0.680306 | skill=38.79% | ACC=0.5848 (base 0.3170)
    +150h: RMSE=0.436195 | base=0.675923 | skill=35.47% | ACC=0.5614 (base 0.3104)
    +156h: RMSE=0.459475 | base=0.666415 | skill=31.05% | ACC=0.5368 (base 0.3128)
    +162h: RMSE=0.483405 | base=0.652567 | skill=25.92% | ACC=0.5150 (base 0.3244)
    +168h: RMSE=0.503720 | base=0.637064 | skill=20.93% | ACC=0.4987 (base 0.3426)
  +06h: RMSE=0.125360 | base=0.363277 | skill=65.49% | ACC=0.9921 (base 0.9343)
  +12h: RMSE=0.156780 | base=0.514900 | skill=69.55% | ACC=0.9876 (base 0.8677)
  +18h: RMSE=0.184867 | base=0.605222 | skill=69.45% | ACC=0.9825 (base 0.8171)
  +24h: RMSE=0.203909 | base=0.662380 | skill=69.22% | ACC=0.9784 (base 0.7808)
RMSE=0.110393 | base=0.433576 | skill=74.54%
    +06h: RMSE=0.092154 | base=0.274631 | skill=66.44% | ACC=0.6786 (base 0.5363)
    +12h: RMSE=0.105211 | base=0.403502 | skill=73.93% | ACC=0.6216 (base 0.3928)
    +18h: RMSE=0.115339 | base=0.481172 | skill=76.03% | ACC=0.5949 (base 0.3342)
    +24h: RMSE=0.126021 | base=0.531217 | skill=76.28% | ACC=0.5737 (base 0.3116)
  +06h: RMSE=0.125360 | base=0.363277 | skill=65.49% | ACC=0.9921 (base 0.9343)
  +12h: RMSE=0.156780 | base=0.514900 | skill=69.55% | ACC=0.9876 (base 0.8677)
  +18h: RMSE=0.184867 | base=0.605222 | skill=69.45% | ACC=0.9825 (base 0.8171)
  +24h: RMSE=0.203909 | base=0.662380 | skill=69.22% | ACC=0.9784 (base 0.7808)
RMSE=0.111964 | base=0.421000 | skill=73.41%
    +06h: RMSE=0.093113 | base=0.268288 | skill=65.29% | ACC=0.9351 (base 0.7685)
    +12h: RMSE=0.107345 | base=0.392209 | skill=72.63% | ACC=0.9176 (base 0.5945)
    +18h: RMSE=0.117438 | base=0.466936 | skill=74.85% | ACC=0.9067 (base 0.4955)
    +24h: RMSE=0.127118 | base=0.514906 | skill=75.31% | ACC=0.8968 (base 0.4398)
[14:36:29] DONE  m33_best_roi rc=0 | skill=74.62% |  t2m K 1.30°C 1.54°C 1.63°C 1.71°C
[15:12:08] DONE  m33_last_roi rc=0 | skill=75.31% |  t2m K 1.32°C 1.53°C 1.59°C 1.66°C
[15:46:56] DONE  m33_best_inner rc=0 | skill=75.72% |  t2m K 1.19°C 1.41°C 1.53°C 1.61°C
[16:22:55] DONE  m33_last_inner rc=0 | skill=76.28% |  t2m K 1.21°C 1.41°C 1.47°C 1.53°C
[16:48:13] DONE  m19_freeze6_roi rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[17:13:23] DONE  m19_freeze6_inner rc=0 | skill=65.54% |  t2m K 1.23°C 1.47°C 1.57°C 1.66°C
[17:39:12] DONE  m19_flagship_roi rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[18:15:06] === ALL DONE ===
[14:36:29] DONE  m33_best_roi rc=0 | skill=74.62% |  t2m K 1.30°C 1.54°C 1.63°C 1.71°C
[15:12:08] DONE  m33_last_roi rc=0 | skill=75.31% |  t2m K 1.32°C 1.53°C 1.59°C 1.66°C
[15:46:56] DONE  m33_best_inner rc=0 | skill=75.72% |  t2m K 1.19°C 1.41°C 1.53°C 1.61°C
[16:22:55] DONE  m33_last_inner rc=0 | skill=76.28% |  t2m K 1.21°C 1.41°C 1.47°C 1.53°C
[16:48:13] DONE  m19_freeze6_roi rc=0 | skill=63.75% |  t2m K 1.34°C 1.59°C 1.71°C 1.82°C
[17:13:23] DONE  m19_freeze6_inner rc=0 | skill=65.54% |  t2m K 1.23°C 1.47°C 1.57°C 1.66°C
[17:39:12] DONE  m19_flagship_roi rc=0 | skill=70.26% |  t2m K 1.39°C 1.66°C 1.74°C 1.84°C
[18:15:06] === ALL DONE ===
```

## tmux
```
none
```
