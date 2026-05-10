# Data Assimilation Experiments — Full Results (v2 Batch)

**Дата запуска:** 13 апреля 2026  
**VM:** MLC `graphcast_v2-bjgvt7` (NVIDIA A100-SXM4-80GB)  
**Ветка:** `main-arthur`, коммит `d7c2305`  
**Батч-скрипт:** `scripts/da_experiments_v2.sh`

---

## 1. Окружение и конфигурация

```
VM path:           /workdir/graphcast-lite
venv:              .venv → /data/venv
LD_LIBRARY_PATH:   /home/mlcore/conda/lib:$LD_LIBRARY_PATH
PYTHONPATH:        /workdir/graphcast-lite
```

### Модель

- **Эксперимент/чекпойнт:** `experiments/multires_nores_freeze6`
- MD5 чекпойнта (`best_model.pth`): `c42ba49fa635c12c65fa3eb2e3813b7c`
- Архитектура: multires GNN, без residual learning, 6 замороженных слоёв
- 19 динамических переменных + 2 статических (z_surf, lsm) + 4 форсинга (sin/cos hour/doy)
- Сетка: multires flat grid, **2501 ROI-узлов** (внутренняя зона ~Красноярский край)

### Датасет

- Источник: WeatherBench 2 (ERA5), подмножество Красноярск-Центральная Сибирь
- Каталог данных на VM: `/data/wb2_64x32_15f_4pred/` (обучение); для инференса тот же формат
- Переменные (19 динамических): `t2m, 10u, 10v, msl, tp, sp, tcwv, t@850, u@850, v@850, z@850, q@850, t@500, u@500, v@500, z@500, q@500` + ещё 2 surface
- Статические каналы [7,8]: z_surf (орография), lsm (маска суша/море) — **исключены из метрик**
- Форсинговые каналы [19–22]: sin/cos hour, sin/cos doy — **исключены из метрик**

### Общие параметры запуска (для всех 26 экспериментов)

```bash
BASE_FLAGS="--max-samples 200 --ar-steps 4 --per-channel --no-residual --obs-roi-only --obs-seed 42"
```

| Флаг | Значение |
|------|----------|
| `--max-samples 200` | 200 тестовых примеров (200 × 4 горизонта = 800 предсказаний) |
| `--ar-steps 4` | Авторегрессионный прогноз на 4 шага (+6h, +12h, +18h, +24h) |
| `--per-channel` | Метрики считаются по каждому каналу |
| `--no-residual` | Без residual learning (так обучена freeze6 модель) |
| `--obs-roi-only` | Станции наблюдений только внутри ROI |
| `--obs-seed 42` | Фиксированный seed для воспроизводимости выбора станций |

### Метрики

- **Skill** — нормализованная RMSE: `1 - RMSE(model) / RMSE(climatology)`, в процентах
- **ACC** — Anomaly Correlation Coefficient
- **RMSE** — Root Mean Square Error (для t2m в °C)
- Метрики считаются для **inner zone** (региональная маска `is_regional`)

---

## 2. Baseline (без Data Assimilation)

```bash
.venv/bin/python -u scripts/predict.py experiments/multires_nores_freeze6 \
    --max-samples 20 --ar-steps 4 --per-channel --no-residual
```

**Лог:** `/data/v2_sanity.log`

| Метрика | Значение |
|---------|----------|
| Global Skill (+6h) | 66.52% |
| Region t2m RMSE +6h | 1.05°C |
| Region t2m RMSE +12h | 1.21°C |
| Region t2m RMSE +18h | 1.29°C |
| Region t2m RMSE +24h | 1.33°C |

---

## 3. OI — Optimal Interpolation

### Формула

Коррекция: `x_a = x_b + K(y - Hx_b)`, где:
- `K = BH^T(HBH^T + R)^{-1}` — gain matrix
- `B = σ_b² · exp(-d²/L²)` — ковариация ошибок прогноза (Гауссово ядро)
- `R = σ_o² · I` — ковариация ошибок наблюдений
- `L` = `corr_len` (радиус корреляции)
- `σ_o` = `sigma_o`
- `σ_b = 1.0` (по умолчанию)

Код: `src/assimilation/optimal_interpolation.py`

### Команда запуска (шаблон)

```bash
.venv/bin/python -u scripts/predict.py experiments/multires_nores_freeze6 \
    $BASE_FLAGS \
    --obs-sparsity <SPARSITY> --assim-method oi \
    --oi-corr-len <CORR_LEN_METERS> --oi-sigma-o <SIGMA>
```

---

### 3.1 OI — Sweep corr_len (10% станций, σ=0.5)

~250 станций наблюдения (10% из 2501 ROI-узлов)

| corr_len | Лог-файл | +6h Skill | +6h ACC | Global Skill |
|----------|----------|-----------|---------|-------------|
| 10 km | `v2_oi_s01_c10000_sig0.5.log` | 73.07% | 0.9720 | 66.95% |
| 25 km | `v2_oi_s01_c25000_sig0.5.log` | 76.75% | 0.9787 | 66.98% |
| 50 km | `v2_oi_s01_c50000_sig0.5.log` | 82.13% | 0.9873 | 67.00% |
| 100 km | `v2_oi_s01_c100000_sig0.5.log` | **83.79%** | **0.9899** | 67.01% |
| 150 km | `v2_oi_s01_c150000_sig0.5.log` | 83.39% | 0.9891 | 67.01% |

**Вывод:** Оптимум — 100 км. При 150 км skill незначительно падает (на 0.4%). Дальнейшее увеличение L нецелесообразно — см. обсуждение в §7.

---

### 3.2 OI — Sweep σ_o (10% станций)

| corr_len | σ=0.3 | σ=0.5 | σ=1.0 | Лог-файлы |
|----------|-------|-------|-------|-----------|
| 10 km | 73.22% | 73.07% | 72.51% | `v2_oi_s01_c10000_sig{0.3,0.5,1.0}.log` |
| 50 km | 82.66% | 82.13% | 79.49% | `v2_oi_s01_c50000_sig{0.3,0.5,1.0}.log` |
| 100 km | 83.57% | 83.79% | 82.54% | `v2_oi_s01_c100000_sig{0.3,0.5,1.0}.log` |

**Вывод:** σ_o слабо влияет (~1% разницы между σ=0.3 и σ=1.0). Можно фиксировать σ=0.5.

---

### 3.3 OI — 1% станций (≈25 наблюдательных точек, σ=0.5)

Реалистичный сценарий: малое число станций метеонаблюдений.

| corr_len | +6h Skill | +12h Skill | +18h Skill | +24h Skill | Лог-файл |
|----------|-----------|-----------|-----------|-----------|----------|
| 10 km | 71.60% | 76.17% | 76.19% | 75.24% | `v2_oi_s001_c10000_sig0.5.log` |
| 50 km | 73.10% | 77.67% | 78.01% | 77.35% | `v2_oi_s001_c50000_sig0.5.log` |
| 100 km | 75.05% | 79.79% | 80.72% | 80.66% | `v2_oi_s001_c100000_sig0.5.log` |
| 150 km | 75.84% | — | — | — | `v2_oi_s001_c150000_sig0.5.log` |

**t2m RMSE (°C, регион inner zone):**

| corr_len | +6h | +12h | +18h | +24h |
|----------|-----|------|------|------|
| 10 km | 1.14 | 1.47 | 1.65 | 1.79 |
| 100 km | 0.95 | 1.20 | 1.33 | 1.43 |

**Вывод:** Даже 25 станций (1%) дают +9% skill vs baseline на +6h и снижают t2m RMSE с 1.05°C до 0.95°C.

---

## 4. Nudging

### Формула

Коррекция: `x_a = x_b + α · H^T(y - Hx_b)`
- `α` — коэффициент релаксации (0..1)
- Sequential — коррекция применяется на каждом AR-шаге
- Offline — коррекция только на начальном шаге

### Команда запуска (шаблон)

```bash
.venv/bin/python -u scripts/predict.py experiments/multires_nores_freeze6 \
    $BASE_FLAGS \
    --obs-sparsity <SPARSITY> --assim-method nudging \
    --nudging-alpha <ALPHA> --nudging-mode <sequential|offline>
```

---

### 4.1 Nudging — 10% станций

| α | mode | +6h Skill | +12h | +18h | +24h | Лог-файл |
|---|------|-----------|------|------|------|----------|
| 0.3 | sequential | 72.18% | 76.73% | 76.82% | 75.92% | `v2_nudge_s01_a0.3_sequential.log` |
| 0.5 | sequential | 72.53% | 77.05% | 77.16% | 76.29% | `v2_nudge_s01_a0.5_sequential.log` |
| 0.7 | sequential | 72.76% | — | — | — | `v2_nudge_s01_a0.7_sequential.log` |
| 0.3 | offline | 72.18% | — | — | — | `v2_nudge_s01_a0.3_offline.log` |

---

### 4.2 Nudging — 1% станций

| α | +6h Skill | Лог-файл |
|---|-----------|----------|
| 0.3 | 71.52% | `v2_nudge_s001_a0.3_sequential.log` |
| 0.5 | 71.55% | `v2_nudge_s001_a0.5_sequential.log` |

---

### Сравнение OI vs Nudging

| Метод | 10% +6h | 1% +6h |
|-------|---------|--------|
| OI (best) | **83.79%** (c=100km) | **75.84%** (c=150km) |
| Nudging (best) | 72.76% (α=0.7) | 71.55% (α=0.5) |
| Baseline | 66.52% | 66.52% |
| **Δ OI vs Nudge** | **+11.03%** | **+4.29%** |

**Вывод:** OI значительно превосходит Nudging. Offline nudging = sequential при одинаковом α (ожидаемо — идентичная коррекция на 1-м шаге).

---

## 5. Variable Groups (частичное наблюдение)

Какие каналы наблюдать? Реалистичный сценарий: метеостанция измеряет только часть переменных.

Конфигурация: OI, corr_len=10km, σ=0.5, 10% станций.

### Команда (шаблон)

```bash
.venv/bin/python -u scripts/predict.py experiments/multires_nores_freeze6 \
    $BASE_FLAGS \
    --obs-sparsity 0.1 --assim-method oi \
    --oi-corr-len 10000 --oi-sigma-o 0.5 \
    --obs-channels "<CHANNEL_LIST>"
```

### Результаты

| Группа | Наблюдаемые каналы | +6h | +12h | +18h | +24h | Лог-файл |
|--------|-------------------|-----|------|------|------|----------|
| t2m only | t2m | 71.49% | 76.07% | 76.08% | 75.11% | `v2_oi_ch_t2m_only_s0.1.log` |
| t + ветер | t2m, 10u, 10v | 71.66% | 76.21% | 76.20% | 75.24% | `v2_oi_ch_t_wind_s0.1.log` |
| surface | t2m, 10u, 10v, msl | 71.70% | 76.25% | 76.27% | 75.32% | `v2_oi_ch_surface_s0.1.log` |
| surface + верх | t2m, 10u, 10v, msl, t@850, t@500 | 71.74% | 76.31% | 76.35% | 75.41% | `v2_oi_ch_surface_tup_s0.1.log` |
| все динамические | 17 переменных | **73.07%** | **77.55%** | **77.71%** | **76.89%** | `v2_oi_ch_all_dynamic_s0.1.log` |

**Вывод:** Наблюдение только t2m даёт 97.8% эффекта surface-группы. Полный набор (17 каналов) даёт **+1.6%** vs t2m-only. Для метеостанций, измеряющих только температуру, DA всё равно полезен.

---

## 6. Файлы логов на VM (для скачивания)

Все логи находятся в `/data/` на VM.

### Обязательные к скачиванию

```
# Sanity (baseline)
/data/v2_sanity.log

# OI корреляция sweep (10%)
/data/v2_oi_s01_c10000_sig0.5.log
/data/v2_oi_s01_c25000_sig0.5.log
/data/v2_oi_s01_c50000_sig0.5.log
/data/v2_oi_s01_c100000_sig0.5.log
/data/v2_oi_s01_c150000_sig0.5.log

# OI sigma sweep (10%)
/data/v2_oi_s01_c10000_sig0.3.log
/data/v2_oi_s01_c10000_sig1.0.log
/data/v2_oi_s01_c50000_sig0.3.log
/data/v2_oi_s01_c50000_sig1.0.log
/data/v2_oi_s01_c100000_sig0.3.log
/data/v2_oi_s01_c100000_sig1.0.log

# OI 1% станций
/data/v2_oi_s001_c10000_sig0.5.log
/data/v2_oi_s001_c50000_sig0.5.log
/data/v2_oi_s001_c100000_sig0.5.log
/data/v2_oi_s001_c150000_sig0.5.log

# Nudging 10%
/data/v2_nudge_s01_a0.3_sequential.log
/data/v2_nudge_s01_a0.5_sequential.log
/data/v2_nudge_s01_a0.7_sequential.log
/data/v2_nudge_s01_a0.3_offline.log

# Nudging 1%
/data/v2_nudge_s001_a0.3_sequential.log
/data/v2_nudge_s001_a0.5_sequential.log

# Variable groups
/data/v2_oi_ch_t2m_only_s0.1.log
/data/v2_oi_ch_t_wind_s0.1.log
/data/v2_oi_ch_surface_s0.1.log
/data/v2_oi_ch_surface_tup_s0.1.log
/data/v2_oi_ch_all_dynamic_s0.1.log

# Батч-лог
/data/v2_batch_master.log
```

### Команда для скачивания всех логов одним архивом

```bash
# НА ВИРТУАЛКЕ: создать архив
cd /data && tar czf /data/v2_all_logs.tar.gz v2_*.log

# ЛОКАЛЬНО: скачать через MLC CLI
mlc job exec graphcast_v2-bjgvt7 -- cat /data/v2_all_logs.tar.gz > results/v2_all_logs.tar.gz
# или через scp, если MLC поддерживает
```

---

## 7. Обсуждение: почему не corr_len > 150 км?

При 10% станций (250 obs):
- 100 km → 83.79%
- 150 km → 83.39% (уже падение)

Skill **не растёт**, а **падает** при 150 km. Причина: при большом L ковариационная матрица B становится слишком гладкой — OI «размазывает» наблюдаемую коррекцию далеко от станции, включая области с другой метеоусловиями. Оптимум ~100 км означает, что средняя корреляция ошибок прогноза в нашем регионе затухает на масштабах ~100 км.

При 1% (25 obs) наблюдается обратная тенденция:
- 100 km → 75.05%
- 150 km → 75.84% (ещё растёт)

При малой плотности станций увеличение L полезно (нужно «дотянуться» до удалённых узлов). Возможно, для 1% оптимум лежит в районе 200–300 км. Можно провести дополнительный эксперимент (см. §8).

---

## 8. Воспроизведение результатов

### Пререквизиты

```bash
# Клонировать и переключиться на ветку
git clone <repo> && cd graphcast-lite
git checkout main-arthur

# Чекпойнт
# MD5: c42ba49fa635c12c65fa3eb2e3813b7c
# experiments/multires_nores_freeze6/best_model.pth

# Данные
# WeatherBench2 ERA5 сетка 64x32, 15f, 4pred, регион Красноярск
```

### Запуск полного батча

```bash
bash scripts/da_experiments_v2.sh 2>&1 | tee /data/v2_batch_master.log
```

### Запуск одного эксперимента (пример — OI 1%, corr=100km)

```bash
.venv/bin/python -u scripts/predict.py experiments/multires_nores_freeze6 \
    --max-samples 200 --ar-steps 4 --per-channel --no-residual \
    --obs-roi-only --obs-seed 42 \
    --obs-sparsity 0.01 --assim-method oi \
    --oi-corr-len 100000 --oi-sigma-o 0.5
```

---

## 9. Ключевые выводы

1. **corr_len — главный гиперпараметр OI.** Оптимум ~100 км при 10% станций, ~150+ км при 1%.
2. **σ_o почти не влияет** (~1% разницы). Фиксируем 0.5.
3. **OI >> Nudging** (до +11% skill при 10% станций, +4% при 1%).
4. **Даже 1% станций (25 точек) полезен:** +9% skill, t2m RMSE 1.05→0.95°C.
5. **Наблюдение только t2m** даёт ~72% skill (vs 73% при всех 17 каналах).
6. **Offline nudging ≡ sequential nudging** при одинаковом α на 1-м шаге.
7. **Чем реже станции, тем больше нужен corr_len** — логично, нужно распространить коррекцию дальше.
