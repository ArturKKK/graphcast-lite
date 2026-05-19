# RFC-001: Neural Postprocessing of GNN Weather Forecasts at Station Locations

**Status:** Draft v1 — пишется параллельно со сборкой honest 0.25° Russia 33f мерджа и запуском fine-tune на VM `graphcast_v4-hts83x`.
**Scope:** train post-processing models that turn raw GraphCast-Lite (multires Russia 33f) forecasts at station coordinates into observation-faithful predictions of:
  * **t2m** (2 m air temperature, °C) — улучшить текущий learned-MOS (`learned_mos_t2m_19stations.joblib`) и закрыть оставшийся UHI/inversion bias;
  * **wind10m** (u10, v10, m/s) — закрыть полностью отсутствующий wind post-processor.
**Out of scope (для этого RFC):** precipitation, gusts, radiation, humidity. Их следует делать второй итерацией.
**Target downstream:** `scripts/live_gdas_forecast.py` (production live-forecast pipeline) и оффлайн-валидация для диплома.

> Это документ-аргумент. Цель — зафиксировать **проблему, идею, архитектуру и план эксперимента** так, чтобы можно было кинуть в ревью внешним моделям/научнику.
>
> **Главная идея в одной строчке:** обучить компактный neural post-processor f(GNN-форкаст; station-context; lead-time) → (t2m_corr, u10_corr, v10_corr), используя пары `<GNN-предсказание @ станции>` ↔ `<реальное наблюдение станции>` за 2010-2021, чтобы аддитивная коррекция была хорошо обусловлена обстановкой (день/ночь, сезон, тип станции, lead-time, фоновая стратификация атмосферы), а не глобальной таблицей bias[month][hour].

---

## 1. Контекст и проблема

### 1.1. Что у нас уже есть
Архитектурная цепочка GraphCast-Lite (см. memory `/memories/repo/graphcast_architecture.md`):

```
ERA5 reanalysis (truth)
   │
   ▼
multires_russia_33f (T×N×33, flat irregular grid: 0.25° внутри Russia bbox, 0.7° снаружи)
   │
   ▼ encode-process-decode GNN  (interaction-net, 12 MP steps, 256-dim)
   │         pretrained: experiments/wb2_512x256_33f_ar_v3 (global, ACC 0.9807, t2m RMSE 1.07°C @200 samples)
   │         fine-tuned on Russia → experiments/multires_russia_33f_v3_noroi (LAUNCH PENDING)
   ▼
prediction (T×N×33) — t2m, 10u, 10v, msl, tp, sp, z_surf, lsm, ...
   │
   ▼ scripts/extract_gnn_*_at_stations.py  (bilinear sample @ station lon/lat)
   │
   ▼ live_gdas_forecast.py  +  apply_mos_t2m  (look-up bias[month][hour_utc] for ONE station, applied to ALL forecasts)
   │
   ▼ live forecast UI (slides, demo)
```

### 1.2. Что не работает / known biases
1. **t2m, UHI и инверсии.** ERA5 surface энергобаланс — это не город. На станции UNKL (Krasnoyarsk) bias =`obs − ERA5` достигает **+7°C зимой ночью** и **−6°C летом днём** (см. `docs/mos_correction.md`). Модель верно репродуцирует ERA5, поэтому в живом прогнозе мы получаем «холодную зимнюю ночь» и «горячий летний день», когда станция показывает наоборот.
2. **Скоростной ветер вообще не корректируется.** Сейчас живой пайплайн отдаёт u10/v10 из GNN as-is. Известная проблема ERA5-derived ветра на 10 м: **систематический underestimate** в холмистой местности и **смазанная направленность** возле орографии. Никакого нейронного post-processor для ветра в проекте нет.
3. **Текущий learned-MOS закрывает только t2m и только лукапом** `bias[month][hour]`:
   * не использует динамические признаки (фактическая стратификация атмосферы, ветер, облачность);
   * не зависит от **lead-time** — а front passage за 36 ч ошибается совсем не так, как nowcast;
   * не зависит от **station-context** (lat, lon, elevation, urban/rural, distance-to-coast);
   * не обучен на парах с GNN, обучен на парах ERA5↔станция → переносит свой response на GNN, что **double-corrects** там, где GNN уже отстреливает ERA5.
4. **ERA5 surface fields сами биасены.** Если делать «GNN → ERA5 truth» (как сейчас), модель учится плохой истине. Правильная истина для post-processor — **сами станционные наблюдения** (см. рассуждение в `docs/era5_urban_problems_and_postprocessing.md`).
5. **Wind как 2D vector обработать сложно.** Direction циклична → MSE на (u,v) или жесткая декомпозиция в magnitude + circular angle. Текущая инфраструктура учитывает только compмпонентный MSE.
6. **Качество станционных obs неоднородно.** ISD-Lite пропуски в часах, иногда suspect-флаги, разное оборудование. Без data cleaning нельзя.

### 1.3. Почему нужна именно нейронка, а не «улучшим таблицу»
* Bias **зависит от состояния атмосферы**, не только от часа/месяца (например, в антициклон зимой инверсия +10°C, в адвекцию +1°C — оба «январь, 03 UTC»). Таблица усредняет → теряет signal.
* GNN сам уже понемногу учит **систематическую часть** ERA5↔truth — наш post-processor должен корректировать остаток, а не дублировать. Нелинейная модель умеет это разделить.
* Wind direction → MLP/GNN/transformer тривиально учит circular loss, таблица — нет.
* Future-proof: легко добавить новые признаки (radar, sat, lightning) без перепроектирования.

---

## 2. Доступные данные

### 2.1. Источник истины: станции (target Y)
| Набор | Покрытие | Гранулярность | Переменные | Где лежит |
|---|---|---|---|---|
| **NOAA ISD-Lite (RU)** | ≈19 русских станций (см. `scripts/extract_gnn_wind_at_stations.py`, можно расширить до ~200 ISD-RU), 2010-2024 | hourly | t2m, dewpoint, wind speed+direction (→ u10/v10), pressure, cloud cover | `scripts/build_learned_mos.py` качает on-demand с `https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz` |
| **krasecology** | Красноярский край, локальные станции | sub-hourly | t2m, RH | `data/datasets/krasecology/` |
| **CDS-meteorological reports** (опционально) | глобально, BUFR/SYNOP | 3 h/6 h | широкий набор | можно дотянуть с CDS если ISD-Lite окажется тесным |

ISD-Lite — основной target. Известные проблемы: пропуски (`-9999`), wind=0 в полуночный час, неконсистентные часовые пояса. **Чистка обязательна.**

### 2.2. Источник предиктора: GNN forecasts (input X)
Два режима генерации обучающих пар `<X = GNN forecast @ station, Y = station obs @ same time>`:

**Режим A — Hindcast (offline, для тренировки):**
* Берём pretrained Russia 33f (только что запускаем fine-tune; используем именно эту модель).
* Прогоняем autoregressive прогноз с инициализацией из ERA5 в момент t0 (например, каждый день 00 UTC, лет 2010-2020).
* Сохраняем прогноз на каждые +6h, +12h, ..., +120h (lead-time как axis).
* Bilinear-сэмплим в координаты станции → `gnn_t2m, gnn_u10, gnn_v10` для каждой пары `(station_usaf, init_time, lead_time)`.
* Это даёт корпус ~10⁶ примеров (10 лет × 4 init/день × 20 lead-times × 19+ станций).

**Режим B — Live archive (более реалистичный):**
* Используем сохранённые live-прогнозы GDAS из `results/live_gdas_apr*` (2024-2025).
* Меньше данных, но точно отражает inference-distribution.
* Использовать как **out-of-time test set**, не для тренировки.

Уже есть скрипты экстракции: `scripts/extract_gnn_at_stations.py`, `scripts/extract_gnn_wind_at_stations.py`. Их нужно расширить:
* поддержка lead-time axis (сейчас они работают на single-step);
* пакетный режим (10⁶ примеров за разумное время);
* сохранение в Parquet/HDF5.

### 2.3. Контекстные признаки (input X-context)
* **Static per-station:** lat, lon, elevation (m), distance to coast (km), urban_flag, land_cover (бор/степь/тайга), z_surf из ERA5 invariants. Computed once.
* **Temporal forcing:** sin/cos hour, sin/cos doy, lead-time (часы или дни от init).
* **Forecast snapshot (помимо целевой переменной):**
  * **Локальные ERA5/GNN признаки** на той же точке: t850, t500, q850, u/v на 250 и 1000 hPa (это уже в нашем 33-ch GNN!), msl, sp, tp.
  * **Стратификация / lapse rate:** `t850 − t1000` как proxy инверсии — критично для t2m bias зимой.
  * **Wind context:** wind at 850/1000 hPa, в дополнение к 10m.
  * **Solar geometry:** zenith angle (Spencer 1971, уже есть в `build_learned_mos.py`).
  * **Прошлое наблюдение** (опц.) — если в продакшне обновляем post-processor по последнему наблюдению ⇒ persistence baseline для краткосрочки.

### 2.4. Train/Val/Test split
* **By time (preferred):** 2010-01-01 .. 2019-12-31 train, 2020 val, 2021 test. Гарантирует отсутствие temporal leak.
* **By station (бонус-сплит):** leave-stations-out — обучаем на 14 из 19, тестируем на 5. Проверяет, обобщается ли модель на новые точки (важно для масштабирования на всю РФ).
* **By season (sanity):** холодный сезон / тёплый сезон отдельно — bias-profile сильно различается.

---

## 3. Архитектурные варианты

> Подход: **сначала простой baseline, потом усложняем только если упираемся**. Жёстко требуем правильный baseline-цикл (предобработка → обучение → ablation → честный test).

### 3.1. Вариант A — Per-variable MLP head **(MUST-HAVE baseline)**
* Один маленький MLP на каждую целевую переменную (t2m / u10 / v10).
* **Вход:** все признаки из §2.3 + сама GNN-prediction той же переменной, конкатенированы в плоский вектор.
* **Архитектура:** Linear(D, 128) → GELU → LN → Linear(128, 128) → GELU → Dropout(0.1) → Linear(128, output_dim).
* **Выход:** *residual* — `Δ = MLP(x)`, итог `y_pred = y_gnn + Δ`. Это критически важно: модель учится только КОРРЕКЦИИ, а не предсказанию с нуля; быстрее сходимость и стабильнее на длинных lead-times.
* **Loss:**
  * `t2m`: weighted Smooth-L1 (Huber, δ=1.0) на bias, плюс per-station bias-regularization `λ·|mean(y_pred − y_true) per station|` (чтобы не выехать в общий offset).
  * `(u10, v10)`: единая MLP с output_dim=2, лосс — Euclidean MSE на (u,v) **плюс** `α·circular_mae(angle)` (см. §3.5).
* **Размер модели:** <50K параметров. На laptop CPU тренируется за минуты.
* **Зачем нужен:** минимально-жизнеспособный пайплайн, эталон для всех сложных вариантов.

### 3.2. Вариант B — Lead-time-aware Temporal Transformer
* **Идея:** один прогноз — это последовательность по lead-time (0h, 6h, 12h, ..., 120h). Bias эволюционирует **смыслово непрерывно** (трэнд, autocorrelation). 1-D Transformer по lead-axis может это учесть.
* **Вход на каждой позиции:** snapshot признаков на конкретном lead-time.
* **Архитектура:** sinusoidal lead-time embedding + 2-4 transformer blocks (d=128, heads=4) → linear head per timestep.
* Окно: всё forecast от 0 до +5d.
* **Loss:** усреднённый Huber + per-station bias-reg.
* **Польза:** ловит дрейф bias по lead-time (например, начало прогноза почти точное, через 3 дня bias накапливается).
* **Когда брать:** если baseline (A) даёт большие residual-RMSE на дальних lead-times.

### 3.3. Вариант C — Spatial GNN postprocessor (**stretch-goal для wind**)
* **Идея:** строим граф над станциями (k-NN, k=4-6, веса = exp(−d/σ)). На каждом узле — все признаки из §2.3 + GNN-prediction.
* **Архитектура:** mini-GNN из 2-3 GraphSAGE/GAT-слоёв (d=128) → MLP-head на каждый узел.
* **Польза для wind:** соседние станции «видят» один и тот же фронт, ветер пространственно когерентен. GNN-постпроцессор регуляризует через соседей, ловит ошибки которые на одну станцию не видны.
* **Тренировочный batch:** один граф = один (init_time, lead_time), узлы = все станции.
* **Доказательство пользы:** ablation `MLP vs Spatial-GNN` → если RMSE_wind падает на ≥10%, мердж в production.

### 3.4. Вариант D — Probabilistic / Quantile MLP **(добавляется поверх A)**
* Вместо одного значения предсказываем 3 квантиля (q10, q50, q90) либо параметры распределения (μ, σ для нормальной; μ, σ, λ для Skew-Normal у wind).
* **Loss:** pinball / CRPS-gaussian.
* **Польза:**
  * метрика **CRPS** становится возможной → стандарт в постпроцессинге (EMOS literature);
  * для wind можно показывать «вероятность ветра >10 m/s» — полезно для диплома и для будущих safety apps.
* **Сложность:** +1 час, окупается.

### 3.5. Wind-specific: representation и loss
Главная ловушка ветра — direction циклична. Варианты:

| Repr | Loss | Плюсы | Минусы |
|---|---|---|---|
| (u, v) decoupled | MSE на (u, v) | Простой, дифференцируем, симметричный | Лосс по углу неравномерный (low-wind часы доминируют по углу, но не по магнитуде) |
| (mag, angle) | mag-MSE + Von-Mises NLL | Прямой контроль над dir error | Singularity при mag→0; нужно clamp |
| **гибрид (рекомендую)** | `MSE(u,v) + α·mag·(1−cos(Δθ))` | Балансирует, не разваливается при штиле | На 1 гиперпараметр больше |

Выбираем **гибрид** с α=0.5 (подбирается на val).

### 3.6. Сводная таблица: что когда брать
| Need | Pick |
|---|---|
| Сделать что-то работающее за день | A (MLP residual) для t2m и для (u10,v10) с гибридным loss |
| Улучшить на дальних lead-times | + B (transformer по lead) |
| Улучшить wind, ловить фронты | + C (spatial GNN) |
| Probabilistic для диплома | + D (CRPS) |

Стратегия: **A → A+D → B → C**, ablation на каждом шаге.

---

## 4. Datasets pipeline (детализация)

### 4.1. Stage-0: фиксация registry станций
* Создать `data/postproc/stations_registry.json`:
  ```json
  { "284935": { "name": "Yemelyanovo", "lat": 56.173, "lon": 92.493, "elev": 287, "land_cover": "urban", "isd_wban": "99999" }, ... }
  ```
* Расширить набор за пределы 19 (по умолчанию подтянуть всю РФ из ISD-history.txt с lat∈[41,82], lon∈[19,180]) — будет ~120-200 станций. Записать также для каждой `dist_to_coast`, `urban_flag` (по NDVI/LCZ либо ручная разметка).

### 4.2. Stage-1: assemble (station_obs, GNN_forecast) hindcast корпус
Скрипт `scripts/postproc/build_hindcast_corpus.py`:
* Параметры: `--start-year 2010 --end-year 2020 --stations all --leads 6h,12h,...,120h --init-hours 00,12`.
* Шаги для каждого `init_time`:
  1. Сформировать input окно (2 obs) из ERA5/multires_russia_33f.
  2. Прогнать модель autoregressive до max lead.
  3. На каждом lead — bilinear-сэмпл в координаты станций.
  4. Сохранить row `(station_usaf, init_time, lead_time, gnn_t2m, gnn_u10, gnn_v10, contextual_features...)` в **Parquet** (партиционирование по году).
* В параллельном процессе — фетч ISD-Lite obs (`scripts/build_learned_mos.py` уже умеет качать).
* Join по `(station, valid_time = init_time + lead_time)` → одна большая таблица 10⁶+ строк.
* **Sanity QC:** удалить пары где `|gnn_t2m − obs_t2m| > 30°C` (видимо broken obs) и где `wind_speed > 60 m/s` без подтверждения фронта в ERA5.

### 4.3. Stage-2: feature engineering
* Конвертация obs wind direction (degrees, ISD coding) → (u, v) в m/s.
* Solar zenith angle (Spencer 1971).
* `lapse_t = t@1000 − t@850`, `dewpoint_depression = t2m − dewpoint`.
* Sin/cos hour, sin/cos doy.
* Lead-time в нормированных днях.
* Per-station static features: lat, lon, elevation (нормировать), urban flag, dist_to_coast.
* z_surf и lsm из static fields (`live_runtime_bundle/static_fields.npz`).
* Сохранить `features_train.parquet`, `features_val.parquet`, `features_test.parquet`.

### 4.4. Stage-3: scalers
* Mean/std каждого признака считается на train. Сохранить в `data/postproc/scalers_v1.npz`.
* Лучше всего хранить отдельные scalers и **передавать в модель non-normalized targets**, нормализовать вход только. Это уменьшает риск ошибок при инференсе.

---

## 5. Implementation

### 5.1. Структура кода (новые файлы)
```
src/
└── postprocessing/
    ├── mos_correction.py            (существует — оставить как fallback)
    ├── neural/
    │   ├── __init__.py
    │   ├── dataset.py               # PyTorch Dataset поверх Parquet
    │   ├── features.py              # feature engineering
    │   ├── models/
    │   │   ├── mlp_residual.py      # Variant A
    │   │   ├── temporal_tfm.py      # Variant B
    │   │   ├── spatial_gnn.py       # Variant C
    │   │   └── probabilistic.py     # Variant D heads
    │   ├── losses.py                # Huber+bias, hybrid-wind, CRPS, Von-Mises
    │   ├── train.py                 # train loop
    │   ├── eval.py                  # metrics + plots
    │   └── apply.py                 # production inference: GNN forecast → corrected
scripts/
└── postproc/
    ├── build_hindcast_corpus.py
    ├── build_features.py
    ├── train_neural_postproc.py     # CLI на src.postprocessing.neural.train
    └── eval_neural_postproc.py
docs/
└── postprocessing_rfc.md            # этот файл
```

### 5.2. Конфиг (Pydantic, как остальные experiment configs)
```jsonc
// experiments/neural_postproc_t2m_v1/config.json
{
  "target_var": "t2m",
  "model_type": "mlp_residual",
  "model_args": { "hidden": [128, 128], "dropout": 0.1 },
  "features": {
    "static": ["lat","lon","elev","urban_flag","dist_to_coast","z_surf","lsm"],
    "temporal": ["sin_hour","cos_hour","sin_doy","cos_doy","lead_h"],
    "snapshot": ["gnn_t2m","gnn_u10","gnn_v10","gnn_msl","gnn_sp","gnn_t850","gnn_t500","gnn_q850","gnn_z500","lapse_t","dewpoint_depression","solar_zen"],
    "history": []  // на будущее
  },
  "loss": { "kind": "huber", "delta": 1.0, "bias_reg_lambda": 0.05 },
  "optimizer": { "kind": "adamw", "lr": 1e-3, "weight_decay": 1e-4 },
  "scheduler": { "kind": "cosine", "warmup_steps": 1000 },
  "epochs": 30,
  "batch_size": 4096,
  "split": { "train_years": [2010,2019], "val_year": 2020, "test_year": 2021 }
}
```

### 5.3. Тренировка
* PyTorch Lightning **не** нужен — у нас уже свой `train.py` стиль. Просто аккуратный цикл на одной GPU/CPU.
* Mixed precision (fp16) для batch_size=4096 если будет тяжело.
* Сохраняем `best_model.pth` по val-RMSE.
* Логи метрик per-epoch: train_loss, val_huber, val_rmse_overall, val_rmse_perlead, val_bias_perstation.
* TensorBoard опционально, json-логи обязательно (стиль `losses.json` у нас уже есть).

### 5.4. Inference integration
* В `scripts/live_gdas_forecast.py` добавить флаг `--neural-postproc PATH/best_model.pth`.
* После денормализации GNN-prediction вычисляем features → applies model → корректируем (t2m, u10, v10) в physical space.
* Старый `--mos-table` остаётся как fallback, **выключается** автоматически если задан neural.

### 5.5. Wind direction inference invariance
При выходе нейронки в (u, v) **не** прибегаем к polar conversion — сразу пишем corrected (u, v). При выходе в (mag, angle) — конвертим в (u, v) перед отдачей пользователю.

---

## 6. Метрики и acceptance criteria

### 6.1. Per-variable, per-lead-time
* **t2m:** RMSE (°C), MAE, mean bias, MAPE при `|obs| > 2°C`, CRPS (если probabilistic). Цель: **снизить RMSE@24h на 30% относительно raw GNN, на 15% относительно учебного learned-MOS** на тестовом 2021 году.
* **wind10m:** vector-RMSE = √((u−û)² + (v−v̂)²), magnitude RMSE, direction MAE (°), CRPS, `hit_rate(|ws−wŝ| ≤ 2 m/s)`. Цель: **снизить vector-RMSE@24h на 25% относительно raw GNN** (baseline-а у нас нет, считаем raw GNN). Direction MAE@24h < 25°.
* **Skill score** относительно raw GNN и относительно climatology: `SS = 1 − RMSE_model / RMSE_baseline`.

### 6.2. Reliability
* Reliability diagram на квантиль-предикторе (вариант D).
* PIT histogram для probabilistic — должен быть плоский ±5%.

### 6.3. Per-station breakdown
* Heatmap RMSE_post-RMSE_raw по (station × lead) — должно быть **всюду ≤0** (нигде не хуже raw, иначе модель «накосячила» на каком-то типе станции).

### 6.4. Generalisation test (leave-stations-out)
* Обучаем без 5 случайных станций → проверяем на них.
* Допустимо ухудшение RMSE на ≤15% относительно in-sample станций — это значит модель не переобучилась на station-id.

### 6.5. Production smoke
* На последних 30 днях GDAS-live прогнозов: проверить что corrected forecast не отъезжает катастрофически (no `|Δ| > 15°C` for t2m, `|Δu|+|Δv| > 25 m/s` for wind).
* Sanity: monotonicity hour-on-hour не нарушается, никаких NaN.

---

## 7. Риски и known gotchas

| Риск | Митигейшен |
|---|---|
| **ISD-Lite obs шум** (особенно wind dir, особенно при штиле) | Robust loss (Huber, не MSE), фильтр `mag<0.5 m/s → исключить direction` |
| **Double-correction**: GNN после fine-tune уже сам частично откорректирует ERA5↔obs | Учим **residual** относительно GNN, не относительно ERA5 |
| **Distribution shift live vs train** (новый месяц GDAS init может выйти OOD) | Train на live archive как valid; добавить heavy regularization (dropout 0.1+) |
| **Leak ERA5 t2m через признак t850** | OK: t850 не равен t2m, leak пренебрежим; и нам нужен этот признак для UHI |
| **Urban vs rural дисбаланс** в loss | Per-station-weighted sampling, чтобы urban stations не доминировали (их больше) |
| **Long forecast leads статистически реже** в обучающем corpus | Stratified sampling по lead-time, либо weight ∝ 1/n(lead) |
| **q@250 = 0** в global extra scalers (float16 underflow) | Не критично для post-processor: q@250 не входит в feature-set; если войдёт — clamp σ ≥ 1e-8 |
| **Computational cost корпуса**: 10 лет × 4 × 20 × 200 станций = 16M примеров | Хранить в Parquet с zstd; ленивый load в DataLoader; на старте делать ablation на ¼ corpus |
| **Wind direction discontinuity** | См. §3.5 — использовать гибрид MSE+cos-loss |

---

## 8. Roadmap

| Stage | Что делаем | Зависимости |
|---|---|---|
| **S0** | Зафиксировать station registry, скачать ISD-Lite hourly 2010-2021 для всех ru-stations | — |
| **S1** | Запустить hindcast corpus (200 stations × 10 lat × 20 leads × 4 inits) | Russia 33f fine-tuned model готов (в работе) |
| **S2** | Feature engineering + scalers | S1 |
| **S3** | **Baseline MLP residual** (t2m + wind) → fit → eval | S2 |
| **S4** | Probabilistic-head (CRPS) | S3 |
| **S5** | Lead-time transformer (если на дальних лидах S3 проигрывает) | S3 |
| **S6** | Spatial GNN postprocessor (wind) | S3 |
| **S7** | Production integration в `live_gdas_forecast.py` | S3+ |
| **S8** | Leave-stations-out generalisation test | S3 |
| **S9** | Diploma write-up: разделы про postprocessing + plots | S3-S8 |

Минимально-приличный результат на диплом = **S0..S4 + S7 + S8**. Остальное — bonus.

---

## 9. Open questions (нужен ответ научника / самого автора)

1. **Сколько станций реально подтянуть из ISD-Lite RU?** Если >100, имеет ли смысл сразу делать leave-stations-out (S8) на основном бенчмарке, а не как side-check?
2. **Inits per day:** 00/12 UTC достаточно или нужно 4 inits (00/06/12/18) чтобы покрыть suburban diurnal?
3. **Хранилище corpus:** Parquet locally или ChMlflow? Объём ~30 GB Parquet ожидаем.
4. **Lead-time max:** до +5 days хватит, или нужно +10 days (для рассмотрения accumulated bias)?
5. **Хотим ли вообще probabilistic-output** в дипломе, или достаточно point-estimate с приличными метриками?
6. **Wind gust (10m max)** — добавляем во вторую итерацию или это уже стрэтч за пределами diploma scope?
7. **Sequence (LSTM-вариант)** — отбрасываем в пользу Transformer (Variant B)? Я бы отбросил, transformer выигрывает.
8. **Cross-station weighting:** balance by urban/rural или по N_obs_total? Я бы балансировал по urban-flag.

---

## 10. Краткое резюме (TL;DR)

* **Проблема:** GNN-форкаст хорошо репродуцирует ERA5, но не станции. Текущий MOS — табличный, не учитывает динамику, не работает на ветер.
* **Идея:** обучить compact neural residual post-processor f(GNN-snapshot + station-context + lead-time) → Δ(t2m, u10, v10), на парах GNN↔ISD-Lite, target = реальные obs, baseline = raw GNN.
* **Архитектура (поэтапно):**
  1. Per-variable MLP residual (Variant A) — MUST.
  2. + Probabilistic head (D) — для CRPS и слайдов диплома.
  3. + Lead-time Transformer (B) — если RMSE на +72/+120 h не падает достаточно.
  4. + Spatial GNN (C) — для wind, ловить пространственную когерентность.
* **Данные:** ISD-Lite RU 2010-2024 (target), GNN hindcast от fine-tuned Russia 33f модели (predictor), плюс контекст (lapse, solar, stratification).
* **Метрики:** RMSE/MAE/CRPS per variable per lead, skill score vs raw GNN и vs табличного MOS, reliability, leave-stations-out.
* **Acceptance:** −30% RMSE t2m@24h, −25% vector-RMSE wind@24h, leave-stations-out ≤+15%.
* **План:** S0..S8 по разделу 8, минимум для диплома S0-S4+S7+S8.

