# k8smr-macos-build--graphcast-v3-506905250-task-0  —  2026-08-06T11:01:45+00:00

## Железо и диски
```
NVIDIA H100 80GB HBM3, 81559 MiB
Filesystem                                                                                        Size  Used Avail Use% Mounted on
a802a1-ml-m5-vs0.storage.tcsbank.ru:/jupyter_persistent_storage_m5_sm12/macos-build-infra/451742  1.0T  498G  527G  49% /workdir
overlay                                                                                           8.7T  5.6T  2.8T  68% /
227G	/data/datasets
```

## Git
```
290ea46 add v2 vs v3 global comparison runner (venv+unpack+build 33f dataset+both runs, disk-safe)
?? data/postproc/
?? experiments/neural_postproc_v1_train_artifacts/
?? experiments/neural_postproc_v2/eval_per_station_nsk.json
?? experiments/neural_postproc_v2/eval_per_station_nsk.md
?? run_v2.sh
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
experiments/multires_merge_freeze6_v2/best_model.pth	27796 KB
experiments/multires_nores_freeze6/best_model.pth	27796 KB
experiments/multires_nores_freeze6/checkpoint.pth	75172 KB
experiments/multires_nores_nofreeze/best_model.pth	27796 KB
experiments/multires_nores_nofreeze/checkpoint.pth	75172 KB
experiments/multires_russia_19f_noroi/best_model.pth	27796 KB
experiments/multires_russia_19f_noroi/checkpoint.pth	75172 KB
experiments/multires_russia_33f_v3_noroi/best_model.pth	27828 KB
experiments/multires_russia_33f_v3_noroi/checkpoint.pth	75276 KB
experiments/neural_postproc_v1_train_artifacts/best_model.pth	92 KB
experiments/neural_postproc_v2/best_model.pth	140 KB
experiments/neural_postproc_v3/best_model.pth	508 KB
experiments/region_krsk_cds_19f/best_model (18).pth	4212 KB
experiments/wb2_512x256_19f_ar_v2/best_model.pth	27796 KB
experiments/wb2_512x256_33f_ar_v3/best_model.pth	27828 KB
```

## Результаты
```
=== /workdir/paper_results
total 29988
drwxr-xr-x 2 mlcore mlcore    8192 Jul 27 12:02 .
drwxrwxrwx 7 root   root      4096 Aug  6 10:50 ..
-rw-r--r-- 1 mlcore mlcore   13019 Jul 26 20:24 MANIFEST.md
-rw-r--r-- 1 mlcore mlcore   19694 Jul 26 11:39 m1_dipl_noda_ar28.log
-rw-r--r-- 1 mlcore mlcore 1761080 Jul 26 11:39 m1_dipl_noda_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore   19907 Jul 26 12:13 m1_dipl_oi10_ar28.log
-rw-r--r-- 1 mlcore mlcore 1761611 Jul 26 12:13 m1_dipl_oi10_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore   19661 Jul 26 10:03 m1_noda_ar28.log
-rw-r--r-- 1 mlcore mlcore 1765796 Jul 26 10:03 m1_noda_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore   19968 Jul 26 10:36 m1_oi10_ar28.log
-rw-r--r-- 1 mlcore mlcore 1762699 Jul 26 10:36 m1_oi10_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore   20039 Jul 26 10:58 m1_oi10_first4_ar28.log
-rw-r--r-- 1 mlcore mlcore 1767984 Jul 26 10:57 m1_oi10_first4_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore   19963 Jul 26 11:20 m1_oi1_ar28.log
-rw-r--r-- 1 mlcore mlcore 1763117 Jul 26 11:20 m1_oi1_ar28_samples.npz
-rw-r--r-- 1 mlcore mlcore    6749 Jul 26 17:13 m1m5_master.log
-rw-r--r-- 1 mlcore mlcore   12329 Jul 26 13:37 m5_test_baseline.log
-rw-r--r-- 1 mlcore mlcore 2029062 Jul 26 13:37 m5_test_baseline_samples.npz
-rw-r--r-- 1 mlcore mlcore   12615 Jul 26 14:01 m5_test_nudge_a0.5.log
-rw-r--r-- 1 mlcore mlcore 2028791 Jul 26 14:01 m5_test_nudge_a0.5_samples.npz
-rw-r--r-- 1 mlcore mlcore   12615 Jul 26 14:26 m5_test_nudge_a0.7.log
-rw-r--r-- 1 mlcore mlcore 2028756 Jul 26 14:26 m5_test_nudge_a0.7_samples.npz
-rw-r--r-- 1 mlcore mlcore   12615 Jul 26 14:50 m5_test_nudge_a0.9.log
-rw-r--r-- 1 mlcore mlcore 2028544 Jul 26 14:50 m5_test_nudge_a0.9_samples.npz
-rw-r--r-- 1 mlcore mlcore   12615 Jul 26 15:17 m5_test_oi1.log
-rw-r--r-- 1 mlcore mlcore   12648 Jul 26 15:55 m5_test_oi10_seed42.log
-rw-r--r-- 1 mlcore mlcore 2028586 Jul 26 15:55 m5_test_oi10_seed42_samples.npz
-rw-r--r-- 1 mlcore mlcore   12648 Jul 26 16:34 m5_test_oi10_seed43.log
-rw-r--r-- 1 mlcore mlcore 2028633 Jul 26 16:34 m5_test_oi10_seed43_samples.npz
-rw-r--r-- 1 mlcore mlcore   12648 Jul 26 17:13 m5_test_oi10_seed44.log
-rw-r--r-- 1 mlcore mlcore 2028395 Jul 26 17:13 m5_test_oi10_seed44_samples.npz
-rw-r--r-- 1 mlcore mlcore 2029040 Jul 26 15:17 m5_test_oi1_samples.npz
-rw-r--r-- 1 mlcore mlcore   11562 Jul 26 12:33 m5_val_oi_L100000_s0.3.log
-rw-r--r-- 1 mlcore mlcore  257000 Jul 26 12:33 m5_val_oi_L100000_s0.3_samples.npz
-rw-r--r-- 1 mlcore mlcore   11562 Jul 26 12:38 m5_val_oi_L100000_s0.5.log
-rw-r--r-- 1 mlcore mlcore  256951 Jul 26 12:38 m5_val_oi_L100000_s0.5_samples.npz
-rw-r--r-- 1 mlcore mlcore   11562 Jul 26 12:43 m5_val_oi_L100000_s1.0.log
-rw-r--r-- 1 mlcore mlcore  256805 Jul 26 12:43 m5_val_oi_L100000_s1.0_samples.npz
-rw-r--r-- 1 mlcore mlcore   11562 Jul 26 12:48 m5_val_oi_L150000_s0.3.log
    +12h: RMSE=0.075767 | base=0.421250 | skill=82.01% | ACC=0.9619 (base 0.6642)
    +18h: RMSE=0.080384 | base=0.507277 | skill=84.15% | ACC=0.9578 (base 0.5681)
    +24h: RMSE=0.083393 | base=0.563100 | skill=85.19% | ACC=0.9556 (base 0.5103)
  +06h: RMSE=0.118900 | base=0.358192 | skill=66.81% | ACC=0.9926 (base 0.9343)
  +12h: RMSE=0.152535 | base=0.499293 | skill=69.45% | ACC=0.9877 (base 0.8723)
  +18h: RMSE=0.193214 | base=0.580548 | skill=66.72% | ACC=0.9799 (base 0.8275)
  +24h: RMSE=0.220234 | base=0.632293 | skill=65.17% | ACC=0.9735 (base 0.7956)
RMSE=0.069428 | base=0.456140 | skill=84.78%
    +06h: RMSE=0.057677 | base=0.283533 | skill=79.66% | ACC=0.9757 (base 0.8183)
    +12h: RMSE=0.067265 | base=0.421250 | skill=84.03% | ACC=0.9682 (base 0.6642)
    +18h: RMSE=0.073032 | base=0.507277 | skill=85.60% | ACC=0.9640 (base 0.5681)
    +24h: RMSE=0.078078 | base=0.563100 | skill=86.13% | ACC=0.9610 (base 0.5103)
  +06h: RMSE=0.118917 | base=0.358192 | skill=66.80% | ACC=0.9926 (base 0.9343)
  +12h: RMSE=0.152550 | base=0.499293 | skill=69.45% | ACC=0.9877 (base 0.8723)
  +18h: RMSE=0.193228 | base=0.580548 | skill=66.72% | ACC=0.9799 (base 0.8275)
  +24h: RMSE=0.220250 | base=0.632293 | skill=65.17% | ACC=0.9735 (base 0.7956)
RMSE=0.071295 | base=0.456140 | skill=84.37%
    +06h: RMSE=0.059499 | base=0.283533 | skill=79.02% | ACC=0.9744 (base 0.8183)
    +12h: RMSE=0.068994 | base=0.421250 | skill=83.62% | ACC=0.9669 (base 0.6642)
    +18h: RMSE=0.074848 | base=0.507277 | skill=85.25% | ACC=0.9626 (base 0.5681)
    +24h: RMSE=0.080183 | base=0.563100 | skill=85.76% | ACC=0.9596 (base 0.5103)
  +06h: RMSE=0.118999 | base=0.358192 | skill=66.78% | ACC=0.9926 (base 0.9343)
  +12h: RMSE=0.152631 | base=0.499293 | skill=69.43% | ACC=0.9877 (base 0.8723)
  +18h: RMSE=0.193306 | base=0.580548 | skill=66.70% | ACC=0.9799 (base 0.8275)
  +24h: RMSE=0.220336 | base=0.632293 | skill=65.15% | ACC=0.9735 (base 0.7956)
RMSE=0.080750 | base=0.456140 | skill=82.30%
    +06h: RMSE=0.067717 | base=0.283533 | skill=76.12% | ACC=0.9686 (base 0.8183)
    +12h: RMSE=0.077869 | base=0.421250 | skill=81.51% | ACC=0.9601 (base 0.6642)
    +18h: RMSE=0.084495 | base=0.507277 | skill=83.34% | ACC=0.9554 (base 0.5681)
    +24h: RMSE=0.091069 | base=0.563100 | skill=83.83% | ACC=0.9520 (base 0.5103)
  +06h: RMSE=1.019288 | base=0.367597 | skill=-177.28% | ACC=0.9760 (base 0.9335)
  +12h: RMSE=2.527003 | base=0.513983 | skill=-391.65% | ACC=0.9318 (base 0.8704)
  +18h: RMSE=4.184597 | base=0.598747 | skill=-598.89% | ACC=0.8802 (base 0.8247)
  +24h: RMSE=5.998064 | base=0.652403 | skill=-819.38% | ACC=0.8355 (base 0.7924)
[15:02:26] DONE  v2_global rc=0 |  |  t2m K 19.14°C 50.81°C 89.89°C 133.83°C
[15:02:38] DONE  v3_global rc=1 |  | 
[15:02:38] === ALL DONE ===
[15:02:26] DONE  v2_global rc=0 |  |  t2m K 19.14°C 50.81°C 89.89°C 133.83°C
[15:02:38] DONE  v3_global rc=1 |  | 
[15:02:38] === ALL DONE ===
```

## tmux
```
none
```
