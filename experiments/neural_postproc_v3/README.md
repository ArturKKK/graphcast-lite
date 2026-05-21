# Neural Postprocessor v3 — RU all-stations + per-station bias head

## Цель
Заменить v2 (50 станций) на полноценный пост-проц по **всем 689 станциям РФ**
из каталога `data/russia_mos_stations.json`, чтобы сайт прогноза покрывал всю
страну, а не только аэропорты с лучшим покрытием.

## Главное архитектурное изменение vs v2

Модель: **`StationLeadBiasResidualMLP`** (src/postprocessing/neural/models.py).

```
ŷ_t2m = gnn_t2m + Δ_mlp(features, station, lead) + b_station[t2m]
ŷ_u   = gnn_u10 + Δ_mlp(features, station, lead) + b_station[u]
ŷ_v   = gnn_v10 + Δ_mlp(features, station, lead) + b_station[v]
```

Новый параметр `bias_emb = nn.Embedding(N_stations, 3)`, инициализируется в 0,
обучается градиентом. Это **чистый аддитивный per-station сдвиг** — то, чего
не хватало v2 (там station влияет только через FiLM-модуляцию через 16-d emb).

## Почему именно это даёт буст

В v2 наблюдается остаточный bias: например, на Новосибирске GNN раз был −0.41°C
по T2m, после v2 остался −0.07°C, но в **некоторых режимах** (ночь зимой+
инверсия) остаточный bias всё ещё −1.6°C при сыром −1.9°C. Причина:
1. residual-MLP режет общий bias через MSE-shrinkage, но не до конца в редких
   режимах;
2. нет чисто-станционного аддитивного канала, только косвенное влияние через
   embedding+FiLM.

Bias-head адресует эту проблему напрямую: он выучивает «на этой конкретной
станции в среднем нужно прибавить XX°C к выходу MLP».

## Изменения относительно v2

| параметр | v2 | v3 |
|---|---|---|
| модель | StationLeadAwareResidualMLP | **StationLeadBiasResidualMLP** |
| per-station bias head | — | **Embedding(N,3)** |
| station_emb_dim | 16 | **32** |
| hidden | [128,128] | **[192,192,128]** |
| film_hidden | 32 | 64 |
| station coverage | 50 (top obs/day) | **689 (все)** |
| epochs | 30 | 40 |
| corpus | corpus_v2_*.parquet | **corpus_v3_*.parquet** |
| features | те же 27 (см. ниже) | те же 27 (без изменений) |

## Фичи (27, такие же как в v2 — НЕ меняем!)

Уже включены в v2, не путать с предыдущими ложными утверждениями:
- **GNN-snapshot (13)**: gnn_t2m, gnn_u10/v10, gnn_msl, gnn_sp, gnn_t850, gnn_t500,
  gnn_q850, gnn_z500, gnn_u850/v850, gnn_u1000/v1000
- **Derived (3)**: lapse_t850_1000 (термический градиент!), dewpoint_depression,
  solar_zen
- **Static станции/модели (5)**: lat, lon, elev (реальная высота станции),
  z_surf (модельная высота), lsm
- **Calendar (4)**: sin_hour, cos_hour, sin_doy, cos_doy ← **час и день года уже есть**
- **Lead (1)**: lead_norm = lead_h / 120

Lapse-rate −6.5°C/км и поправка на высоту между станцией и модельным рельефом
модель видит через `elev`, `z_surf`, `gnn_t2m`, `gnn_t850`, `lapse_t850_1000`.

## Корпус v3

Команда сборки:
```bash
python scripts/postproc/build_corpus.py \
    --experiment-dir experiments/multires_russia_33f_v3_noroi \
    --multires-dir   /data/datasets/multires_russia_33f \
    --global-base    /data/datasets/wb2_512x256_19f_ar \
    --regional-base  /data/datasets/region_russia_645x165_19f_2010-2021_025deg \
    --global-extra   /data/datasets/global_512x256_extra_2010-2021_07deg \
    --regional-extra /data/datasets/region_russia_645x165_extra_2010-2021_025deg \
    --stations-json  data/russia_mos_stations.json \
    --isd-dir        /data/datasets/isd_lite_russia \
    --top-stations 689 \
    --years 2018 2020 \
    --init-hours 0 12 \
    --leads-h 6 12 18 24 36 48 72 96 120 \
    --out-parquet data/postproc/corpus_v3_full.parquet \
    --device cuda
```

Затем split по годам:
```bash
python scripts/postproc/split_corpus.py \
    --input data/postproc/corpus_v3_full.parquet \
    --out-dir data/postproc \
    --prefix corpus_v3 \
    --train-years 2018 2019 \
    --val-years 2020
```

Ожидаемый размер: ~12-15M строк train + ~6-8M val (×14 vs v2, но леды сократили
до 9 точек вместо 20 — итоговый множитель ~7).

## Команда тренировки

```bash
python scripts/postproc/train_neural_postproc_v3.py \
    --train-parquet data/postproc/corpus_v3_train.parquet \
    --val-parquet   data/postproc/corpus_v3_val.parquet \
    --out-dir       experiments/neural_postproc_v3 \
    --epochs 40 --batch-size 8192 --station-emb-dim 32 \
    --hidden 192,192,128
```

## Baseline (v2 для сравнения)
- val rmse_t2m = **2.407 °C**
- val vec_rmse_wind = **2.521 m/s**
- val bias_t2m (per-lead): −0.02..−0.29 °C

## Ожидаемые результаты v3
- val rmse_t2m: **2.2-2.3 °C** (−0.10..−0.20 за счёт bias-head + больше станций)
- val bias_t2m: **|bias| < 0.05 °C на каждой станции** (главный буст)
- val vec_rmse_wind: **2.4-2.5 m/s** (умеренное улучшение)

## Идеи на будущее (НЕ в v3)

- Lapse-rate проверка: добавить `elev_diff = elev - z_surf` явно (сейчас через
  две раздельные фичи, модель должна сама вычесть, но явный канал может помочь).
- t925, t1000 как фичи (lapse-rate в нижнем км) — нужны в исходных GNN.
- Probabilistic head (μ, log σ) + CRPS loss — для калибровки uncertainty.
- Fine-tune на 2021-2024 после переезда корпуса.
- Recent-obs anchor (последнее наблюдение станции на init) — самая мощная
  фича, требует переделки live-пайплайна.

## Логика именования

- `v1` = MultiTaskResidualMLP, без station_emb, 50 станций, leads 6-24
- `v2` = StationLeadAwareResidualMLP, station_emb=16+FiLM, 50 станций, leads 6-120
- `v3` = **StationLeadBiasResidualMLP**, +per-station bias-head, **689 станций**,
  leads 6-120 (9 точек)
