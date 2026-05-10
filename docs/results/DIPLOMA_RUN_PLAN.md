# План прогонов и замеров для диплома

> Дата: 13 апреля 2026  
> Цель: собрать все эксперименты воедино, прогнать финальные замеры, оформить сравнительные таблицы  
> WRF и live-прогнозы — вне скоупа, фокус на сборе всей информации

---

## Часть 0. Инвентаризация: что у нас есть

### Боевые (production) модели

| # | Модель | Эксперимент | Параметры | Статус |
|---|---|---|---|---|
| 1 | **Глобальная GNN v2** | `wb2_512x256_19f_ar_v2` | 5.9M, InteractionNet 12 MP, hidden=256 | ✅ Обучена |
| 2 | **Мультирез. freeze6** | `multires_nores_freeze6` | v2 + finetune, use_residual=false | ✅ **ОСНОВНАЯ** |
| 3 | **U-Net V1 (downscaler)** | `downscaler_krsk` / `unet_region_krsk` | 7.8M, base_filters=64 | ✅ Обучена |
| 4 | **Каскад GNN→U-Net** | `predict_cascade.py` | v2 GNN + U-Net downscaler | ✅ Код + веса есть, метрики частичные |
| 5 | **Learned MOS** | `learned_mos_t2m_19stations.joblib` | HistGBR, 19 станций | ✅ Готова |
| 6 | **DA (Nudging + OI)** | `src/assimilation/` | nudging + optimal_interpolation | ✅ Код есть, параметры подобраны для НСК |

### Базлайны (для сравнения)

| # | Модель | Эксперимент | Skill |
|---|---|---|---|
| B1 | Глобальная GNN v1 (GCNConv) | `wb2_512x256_19f_ar` | 56.42% |
| B2 | Глобальная GNN 64×32 | `wb2_64x32_15f_4pred` | 31.03% |
| B3 | Региональная GNN с нуля (19f) | `region_krsk_cds_19f` | 29.69% |
| B4 | Региональная GNN 23f | `region_krsk_cds_23f` | 30.95% |
| B5 | Мультирез. nofreeze | `multires_nores_nofreeze` | 74.45% |

### Экспериментальные (упомянуть в обзоре, но не в основных результатах)

- `dual_mesh_krsk` — каскад Global GNN + Regional Mesh, не доведён
- `roi_residual_krsk` — residual learning в ROI
- `attention`, `sparse_attention` — GAT-варианты, сравнимы с GCN, но медленнее
- `product_graph` — пространственно-временной граф, хуже базлайна (отрицательный результат)
- `unet_v2_region_krsk` — U-Net V2 с attention, переобучилась (лучше @6h, хуже @24h)

---

## Часть 1. Сюжетная линия диплома

### Глава 1: Эволюция глобальной модели
**Тезис:** разрешение — главный фактор (+25 п.п.), архитектура InteractionNet — ещё +0.84 п.п.

✅ **Всё готово**, метрики собраны.

### Глава 2: Региональное уточнение — 4 подхода

**Тезис:** прямая GNN ~30% Skill (мало данных) → U-Net 55.7% → Каскад GNN→U-Net ~72% → Мультирез freeze6 **75.82%** (winner).

| Подход | Идея | Результат |
|---|---|---|
| GNN с нуля | Обучить маленькую GNN на региональных данных | Потолок ~30%, oversmoothing при увеличении глубины |
| U-Net standalone | Свёрточная сеть на 2D-сетке региона | 55.7% Skill, 7.8M параметров |
| **Каскад GNN→U-Net** | Глобальная GNN даёт грубый прогноз → U-Net уточняет | **~72% Skill**, t2m +11% vs GNN-only |
| **Мультирезолюция** | Файнтюн глобальной GNN на мультимасштабной сетке | **75.82% Skill** (winner) |

**Что уже есть:**
- ✅ GNN с нуля: 5 экспериментов с полными метриками
- ✅ U-Net standalone: метрики, per-channel
- ✅ Мультирез: freeze6 vs nofreeze, полные метрики
- ⚠️ Каскад: есть частичные результаты (50 samples), **нужен полный прогон на 200 samples**

### Глава 3: Постпроцессинг (MOS + Lapse-rate)
**Тезис:** Learned MOS исправляет систематический bias, lapse-rate — высотное смещение.

- ✅ MOS обучена (19 станций), MAE 1.32°C на тесте
- ⚠️ **Нужен замер**: Skill/RMSE multires_freeze6 **до и после** MOS на ERA5-тесте

### Глава 4: Усвоение данных (DA)
**Тезис:** OI + boundary taper снижает RMSE на ~35%, усвоение только температуры бесполезно.

- ✅ DA для **НСК** (64×32, 15f): полные метрики, подбор параметров
- 🔴 **КРИТИЧНО:** DA для **Красноярска** (133K, 19f) — **не делалось вообще!**

**Техническая проблема с OI на multires:**
- Полная матрица B (133K × 133K) = ~70 ГБ — невозможно
- **Решение:** OI только на ROI-узлах (~2795 точек), B = 100 МБ — ОК
- Nudging работает на полной сетке без проблем
- Нужна доработка `predict.py` (~20 строк): region masking в OI init + apply

---

## Часть 2. Конкретный план прогонов

### ПРОГОН 1: Каскад GNN→U-Net (полные метрики) 🟡

**Цель:** получить Skill/RMSE каскада на **200 образцах** (сейчас есть на 50).

**Уже известно (50 samples):**
- Каскад t2m: 1.58°C (+11% vs GNN-only 1.77°C) @+24h
- Каскад Skill vs persistence: ~72% (GNN-only: 65.5%)
- Каскад помогает по t2m, tp, sp; чуть хуже по ветру

**Нужно:**
1. Проверить наличие датасета `downscaler_krsk_19f` (нужна fine-grid ERA5 0.25° как ground truth)
2. Прогнать на 200 samples для согласованности с другими экспериментами

```bash
python scripts/predict_cascade.py \
  experiments/multires_nores_freeze6 \
  experiments/downscaler_krsk \
  --downscaler-data data/datasets/downscaler_krsk_19f \
  --gnn-data data/datasets/multires_krsk_19f \
  --roi 50 60 83 98 \
  --ar-steps 4 \
  --max-samples 200
```

---

### ПРОГОН 2: MOS на ERA5-тесте 🟡

**Цель:** замерить вклад MOS в метрики на ретроспективных данных (не live).

```bash
python scripts/evaluate_full_pipeline.py \
  --experiment-dir experiments/multires_nores_freeze6 \
  --learned-mos live_runtime_bundle/learned_mos_t2m_19stations.joblib \
  --max-samples 200
```

---

### ПРОГОН 3: DA для Красноярска 🔴 КРИТИЧНЫЙ

**Цель:** повторить OSSE-эксперименты НСК на текущей боевой модели.

**Протокол (OSSE = Observing System Simulation Experiment):**
- Модель: `multires_nores_freeze6` (1-step, obs=2, 19 переменных)
- «Наблюдения» = ERA5 на 10% узлов ROI (~250 из ~2500)
- Методы: control (без DA), nudging (α=0.5, 0.9), OI (L=50km, σ_b=0.8, σ_o=0.2), OI+boundary

**Шаги:**
1. Сгенерировать obs-файл (маскированный ERA5 — 10% узлов видимы)
2. Прогнать nudging (sequential rollout, α=0.5 и 0.9)
3. Прогнать OI (ROI-only, ~2795 узлов) — нужна доработка predict.py
4. Вычислить per-channel RMSE/Skill/ACC по ROI

**Доработка predict.py для OI на multires:**
```python
# В блок инициализации OI (~строка 310):
# Если grid > 10K узлов — использовать только ROI-координаты для B-матрицы

# В блок apply OI (~строка 490):
# Применять OI только к region_idxs, остальные узлы не трогать
```

```bash
# 1. Генерация наблюдений
python scripts/generate_assim_dataset.py \
  experiments/multires_nores_freeze6 \
  --vars "t2m,10u,10v,msl" \
  --out-name assimilation_dynamic

# 2. Nudging
python scripts/predict.py experiments/multires_nores_freeze6 \
  --data-dir data/datasets/multires_krsk_19f \
  --region 50 60 83 98 \
  --assim-method nudging --nudging-alpha 0.9 --nudging-mode sequential \
  --obs-path experiments/multires_nores_freeze6/assimilation_dynamic/y_obs.pt \
  --max-samples 100 --per-channel

# 3. OI (после доработки predict.py)
python scripts/predict.py experiments/multires_nores_freeze6 \
  --data-dir data/datasets/multires_krsk_19f \
  --region 50 60 83 98 \
  --assim-method oi --oi-corr-len 50000 --oi-sigma-b 0.8 --oi-sigma-o 0.2 \
  --obs-path experiments/multires_nores_freeze6/assimilation_dynamic/y_obs.pt \
  --max-samples 100 --per-channel
```

---

## Часть 3. Итоговые таблицы для диплома

### Таблица A: Эволюция глобальной модели ✅

| Модель | Сетка | Перем. | Параметры | Skill | ACC +6ч | ACC +24ч |
|---|---|---|---|---|---|---|
| Базовая (GCNConv) | 64×32 | 15 | ~50K | 31.03% | 0.977 | 0.881 |
| v1 (GCNConv) | 512×256 | 19 | ~210K | 56.42% | 0.987 | 0.955 |
| **v2 (InteractionNet)** | **512×256** | **19** | **~5.9M** | **57.26%** | **0.988** | **0.956** |

### Таблица B: Подходы к региональному прогнозу ⚠️ (не хватает каскада @200 samples)

| Подход | t2m +6ч | t2m +24ч | Skill (рег.) |
|---|---|---|---|
| GNN с нуля (19f) | 2.66°C | 4.33°C | 29.69% |
| GNN с нуля (23f) | 2.25°C | — | 30.95% |
| U-Net V1 standalone | 2.19°C | 3.82°C | 55.7% |
| **Каскад GNN→U-Net** | **~1.58°C** *(50s)* | **???** | **~72%** *(50s)* |
| Мультирез. nofreeze | 0.98°C | 1.82°C | 74.45% |
| **Мультирез. freeze6** | **0.96°C** | **1.40°C** | **75.82%** |

### Таблица C: Стратегии файнтюна мультирезолюции ✅

| Стратегия | Skill (рег.) | t2m +24ч |
|---|---|---|
| Без заморозки | 74.45% | 1.82°C |
| **Заморозка 6 эпох** | **75.82%** | **1.40°C** |

### Таблица D: Постпроцессинг (MOS) ⚠️ (ПРОГОН 2)

| Конфигурация | t2m RMSE (сред.) | Δ |
|---|---|---|
| Модель (без постпроцессинга) | **???** | — |
| + Lapse-rate | **???** | **???** |
| + Learned MOS (19 стн.) | **???** | **???** |

### Таблица E: Усвоение данных 🔴 (ПРОГОН 3)

| Конфигурация | RMSE | Skill | Δ vs control |
|---|---|---|---|
| **НСК (64×32, 15f):** | | | |
| Control | 0.629 | 19.7% | — |
| Nudging (α=0.9) | ~0.60 | ~24% | +4 п.п. |
| OI (L=50km, σ_o=0.2) | 0.561 | 28.35% | +8.6 п.п. |
| OI + boundary | **0.414** | **47.2%** | **+27.5 п.п.** |
| **Красноярск (133K, 19f):** | | | |
| Control | **???** | **???** | — |
| Nudging | **???** | **???** | **???** |
| OI + boundary | **???** | **???** | **???** |

---

## Часть 4. Приоритеты

| Приоритет | Задача | Сложность | Блокер |
|---|---|---|---|
| 🔴 P0 | **ПРОГОН 3: DA для Красноярска** | Средняя (доработка ~20 строк predict.py + прогон) | Нужен датасет multires + доработка OI |
| 🟡 P1 | **ПРОГОН 1: Каскад @200 samples** | Лёгкая (код готов) | Нужен датасет downscaler_krsk_19f |
| 🟡 P1 | **ПРОГОН 2: MOS на ERA5-тесте** | Лёгкая | Нужен датасет multires |
| ⚪ | ~~WRF сравнение~~ | — | Вне скоупа |
| ⚪ | ~~Live прогнозы~~ | — | Вне скоупа |

---

## Часть 5. Чеклист

### Уже готово ✅
- [x] Эволюция глобальной модели (64×32 → 512×256, GCNConv → InteractionNet)
- [x] Региональная GNN с нуля (5 экспериментов, потолок ~30%)
- [x] U-Net standalone (V1 и V2)
- [x] Каскад GNN→U-Net (частичные метрики, 50 samples, ~72% Skill)
- [x] Мультирезолюция freeze6 vs nofreeze
- [x] DA эксперименты для НСК (nudging, OI, подбор параметров)
- [x] Learned MOS (19 станций)

### Нужно сделать ⏳
- [ ] DA для Красноярска (ПРОГОН 3) ← **блокирует главу 4**
- [ ] Каскад @200 samples (ПРОГОН 1) ← уточняет главу 2
- [ ] MOS на ERA5-тесте (ПРОГОН 2) ← уточняет главу 3

### Общий блокер: данные
Все 3 прогона требуют датасетов на диске (`multires_krsk_19f`, `downscaler_krsk_19f`). Они не в репе — нужно собрать/скачать (инструкции в temp.md).
