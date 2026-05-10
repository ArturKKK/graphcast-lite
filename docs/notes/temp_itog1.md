Let me собрать все данные из всех источников — результаты v2 (interpolate), merge, MOS, и всё что есть.    

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_MERGE_RESULTS.md#1-1), lines 1 to 50

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_RESULTS.md#1-1), lines 1 to 100

Read memory [](file:///Users/a.s.tabakov/Library/Application%20Support/Code/User/workspaceStorage/2d58b91583f8c02072696770ca298c5f/GitHub.copilot-chat/memory-tool/memories/repo/graphcast_architecture.md)

Read memory [](file:///Users/a.s.tabakov/Library/Application%20Support/Code/User/workspaceStorage/2d58b91583f8c02072696770ca298c5f/GitHub.copilot-chat/memory-tool/memories/repo/mos_and_forecast.md)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_RESULTS.md#100-100), lines 100 to 360

Now let me also check for any experiment folders to see what training experiments were done:  

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_MERGE_RESULTS.md#50-50), lines 50 to 200

Now let me check training experiment results and any other data sources:   

Gather all experiment results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/RESULTS_SUMMARY.md#1-1), lines 1 to 300

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/diploma_experiment_results.md#1-1), lines 1 to 500

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/temp.md)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/EXPERIMENT_RESULTS.md)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/results)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/today.md#1-1), lines 1 to 200

Searched for files matching `**/experiments/**/config.json`, 27 matches

Searched for files matching `**/experiments/**/training_log.txt`, 4 matches

Searched for files matching `**/experiments/**/itog.md`, 4 matches

Searched for files matching `**/data/v2_all_logs/**`, no matches

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/region_krsk_cds_19f/itog.md#1-1), lines 1 to 150

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/unet_region_krsk/itog.md#1-1), lines 1 to 150

Searched for regex `rmse|skill|acc|epoch|loss` (`**/data/v2_all_logs/**`), 20 results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_freeze6/training_log.txt)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/aaaa/2010-2020_15var_4obs_4_pred_50e/itog.md)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/v2_all_logs)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/wb2_512x256_19f_ar/config.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_freeze6/config.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/region_krsk_cds_23f/config.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_RESULTS.md#1-1), lines 1 to 100

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/wb2_512x256_19f_ar_v2/config.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_RESULTS.md#100-100), lines 100 to 300

Searched for regex `config\.json|results\.txt|results\.md|\.log` (`**/results/**`), 7 results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/DA_EXPERIMENTS_RESULTS.md#100-100), lines 100 to 361

Read [](file:///Users/a.s.tabakov/Library/Application%20Support/Code/User/workspaceStorage/2d58b91583f8c02072696770ca298c5f/GitHub.copilot-chat/chat-session-resources/28e6ed13-6300-4199-8cbe-8ba648ef5850/toolu_bdrk_018Zs5YfsTt4x8dsFHk6kXRu__vscode-1776103823793/content.txt#1-1), lines 1 to 200

Read [](file:///Users/a.s.tabakov/Library/Application%20Support/Code/User/workspaceStorage/2d58b91583f8c02072696770ca298c5f/GitHub.copilot-chat/chat-session-resources/28e6ed13-6300-4199-8cbe-8ba648ef5850/toolu_bdrk_018Zs5YfsTt4x8dsFHk6kXRu__vscode-1776103823793/content.txt#200-200), lines 200 to 247

Now let me also check what's in diploma_experiment_results.md and RESULTS_SUMMARY.md for any additional data:    

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/RESULTS_SUMMARY.md#1-1), lines 1 to 80

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/diploma_experiment_results.md#1-1), lines 1 to 80

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/RESULTS_SUMMARY.md#80-80), lines 80 to 250

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/RESULTS_SUMMARY.md#250-250), lines 250 to 351

Now I have a comprehensive picture. Let me also check for live forecast results:   

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/results)

Вот абсолютно все результаты по проекту GraphCast-Lite. Собрал из всех файлов.

---

## ПОЛНЫЙ КАТАЛОГ РЕЗУЛЬТАТОВ

### I. ГЛОБАЛЬНЫЕ МОДЕЛИ (обучение с нуля)

**Датасет:** ERA5 (WeatherBench 2), 2010–2020

| # | Модель | Сетка | Перем. | Параметры | Архитектура | Skill (ср.) | ACC +6h | t2m → |
|---|---|---|---|---|---|---|---|---|
| 1 | 3-var demo | 64×32 (2048) | 3 | ~10K | GCNConv | 29.63% | — | 38.73% skill |
| 2 | 15-var obs=4 pred=1 | 64×32 | 15 | ~50K | GCNConv | 35.62% | — | 56.11% skill |
| 3 | **wb2_64x32_15f_4pred** | 64×32 | 15 | ~50K | conv_gcn [96,96,96] | **31.03%** | 0.977 | — |
| 4 | **wb2_512x256_v1** | 512×256 (131K) | 19 | ~210K | GCNConv 4 слоя | **56.42%** | 0.987 | — |
| 5 | **wb2_512x256_v2** | 512×256 (131K) | 19 | **~5.9M** | InteractionNet 12 MP | **57.26%** | 0.988 | — |

**Вывод:** Разрешение 64×32→512×256 = **+25 п.п.** Архитектура v1→v2 = **+0.84 п.п.**

---

### II. РЕГИОНАЛЬНЫЕ МОДЕЛИ (Красноярск, 61×41, 0.25°)

**Датасет:** ERA5, регион 50–60°N × 83–98°E, 2501 узел, 2010–2020

| # | Модель | Перем. | Mesh | Hidden×MP | Params | Skill | t2m +6h |
|---|---|---|---|---|---|---|---|
| 1 | GNN exp1 (базовая) | 19 | 259 | 128×8 | — | 29.69% | 2.66°C |
| 2 | GNN крупный mesh | 19 | 1029 | 128×8 | — | хуже | — |
| 3 | GNN глубокая (oversmoothing) | 19 | 1029 | 256×16 | — | **−11.52%** | 💀 |
| 4 | GNN + маска границ | 19 | 259 | 256×8 | — | 25.37% | — |
| 5 | GNN + 23 перем. | 23 | 259 | 128×8 | — | 30.95% | 2.25°C |
| 6 | **U-Net V1** | 19 | — | base=64, 3 уровня | 7.8M | **55.7%** | **2.19°C** |
| 7 | U-Net V2 (attention+spectral) | 23 | — | — | 25.5M | ~40.5% | 2.05°C |

**Вывод:** U-Net >> GNN на малом датасете (+26 п.п.). V2 переобучилась.

---

### III. МУЛЬТИРЕЗОЛЮЦИЯ (файнтюн глобальной на региональном)

**Датасет:** multires_krsk_19f = 131K global + 2501 regional = **133,279 узлов**

| Стратегия | Skill глоб. | Skill рег. | t2m +6h | t2m +24h |
|---|---|---|---|---|
| nofreeze | 65.19% | 74.45% | 0.98°C | 1.82°C |
| **freeze6** (BEST) | **66.94%** | **75.82%** | **0.96°C** | **1.40°C** |

**Per-horizon (freeze6, регион, 200 сэмплов, ERA5 тест 2010–2021):**

| Горизонт | Skill | ACC | t2m RMSE | 10u RMSE | 10v RMSE | msl RMSE |
|---|---|---|---|---|---|---|
| +6h | 71.45% | 0.780 | 0.96°C | 0.43 м/с | 0.35 м/с | 0.63 Па |
| +12h | 76.59% | 0.730 | 1.22°C | 0.52 | 0.40 | 0.88 |
| +18h | 76.99% | 0.699 | 1.29°C | 0.58 | 0.45 | 1.23 |
| +24h | 75.76% | 0.676 | 1.40°C | 0.63 | 0.48 | 1.58 |

**OOD-тест (январь 2023, 27 сэмплов):** Skill глоб.=54.17%, регион=73.56%, t2m +6h=1.14–1.36°C

---

### IV. СРАВНЕНИЕ С WRF (20 янв 2023, 1 образец)

WRF4, домен d03, 96×84, ~1 км

| Переменная | Наша RMSE (ср.) | WRF RMSE | Победитель |
|---|---|---|---|
| t2m | 2.34 K | **1.79 K** | WRF |
| 10u | **0.48 м/с** | 0.59 м/с | **Мы** |
| 10v | **0.41 м/с** | 0.91 м/с | **Мы (×2.2)** |
| sp | 5.43 гПа | **0.83 гПа** | WRF |

**Итог: 2:2.** WRF лучше по температуре/давлению (разрешает рельеф, ~1 км). Мы лучше по ветру (глобальный контекст GNN).

---

### V. DA ЭКСПЕРИМЕНТЫ — РАННИЕ (64×32, НСКО)

**Датасет:** 64×32, 15 переменных, регион НСКО
**Модель:** региональная GNN, 250 станций (10%)

| Конфигурация | Skill (рег.) |
|---|---|
| Control (нет DA) | 15.5% |
| Nudging α=0.5 | 20.3% |
| **OI L=300km σ=0.5** | **40.1%** |
| OI + граничная коррекция | **47.2%** |

---

### VI. DA ЭКСПЕРИМЕНТЫ V2 (interpolate-датасет, 200 сэмплов)

**Датасет:** multires interpolate, ~17,527 сэмплов, **133K узлов**
**Модель:** freeze6, 200 тестовых, AR=4
**Baseline:** Global 66.52%, Region +6h ~71.5%

#### OI с 10% станций (σ=0.5)

| corr_len | +6h Skill | +6h ACC |
|---|---|---|
| 10 km | 73.07% | 0.9720 |
| 25 km | 76.75% | 0.9787 |
| 50 km | 82.13% | 0.9873 |
| **100 km** | **83.79%** | **0.9899** |
| 150 km | 83.39% | 0.9891 |

#### OI σ-sweep (10%)

| corr×σ | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---|---|---|
| 10 km | 73.22% | 73.07% | 72.51% |
| 50 km | 82.66% | 82.13% | 79.49% |
| 100 km | 83.57% | 83.79% | 82.54% |

#### OI с 1% станций (≈25 точек, σ=0.5)

| corr_len | +6h Skill | t2m +6h |
|---|---|---|
| 10 km | 71.60% | 1.14°C |
| 50 km | 73.10% | — |
| 100 km | 75.05% | 0.95°C |
| 150 km | 75.84% | — |

#### Nudging (interpolate)

| Конфиг | +6h Skill |
|---|---|
| 10% α=0.3 | 72.18% |
| 10% α=0.7 | 72.76% |
| 1% α=0.5 | 71.55% |

#### Variable groups (c=10km, 10%)

| Группа | +6h Skill |
|---|---|
| t2m only | 71.49% |
| t + ветер | 71.66% |
| surface | 71.70% |
| all dynamic (17) | 73.07% |

---

### VII. DA ЭКСПЕРИМЕНТЫ V3 (merge-датасет, реальный режим)

**Датасет:** multires merge, 16,072 сэмплов, **133,279 узлов**, 2501 regional
**Модель:** freeze6, AR=4
**Baseline (merge):** Global 65.15%, Region +6h **59.42%**, t2m +6h 0.79°C (global) / 1.42°C (region)

#### OI 10% станций (σ=0.5)

| corr_len | +6h Skill | +6h ACC | RMSE (norm) |
|---|---|---|---|
| 10 km | 61.73% | 0.9627 | 0.0848 |
| 25 km | 67.16% | 0.9685 | 0.0728 |
| 50 km | 75.40% | 0.9747 | 0.0545 |
| **100 km** | **77.86%** | **0.9754** | **0.0491** |
| 150 km | 76.83% | 0.9734 | 0.0513 |
| 200 km | 75.62% | 0.9712 | 0.0540 |
| 300 km | 73.84% | 0.9675 | 0.0580 |
| 500 km | 71.62% | 0.9633 | 0.0629 |

#### OI 10% σ-sweep

| corr×σ | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---|---|---|
| 10 km | 61.94% | 61.73% | 60.93% |
| 50 km | 76.39% | 75.40% | 71.32% |
| 100 km | 77.77% | 77.86% | 75.86% |

#### OI 1% станций (σ=0.5, расширенный sweep)

| corr_len | +6h Skill | ACC |
|---|---|---|
| 10 km | 59.66% | 0.9608 |
| 50 km | 62.08% | 0.9625 |
| 100 km | 65.67% | 0.9630 |
| 150 km | 67.52% | 0.9619 |
| **200 km** | **67.88%** | 0.9604 |
| 300 km | 67.38% | 0.9577 |
| 500 km | 66.86% | 0.9569 |

#### Per-horizon лучших конфигов (merge)

**OI 10% c=100km:**

| Горизонт | Skill | ACC | t2m global | t2m region |
|---|---|---|---|---|
| +6h | 77.86% | 0.9754 | 0.77°C | 0.84°C |
| +12h | 82.00% | 0.9672 | 0.95°C | 1.02°C |
| +18h | 83.54% | 0.9618 | 1.09°C | 1.09°C |
| +24h | 84.28% | 0.9584 | 1.19°C | 1.15°C |

**OI 1% c=200km:**

| Горизонт | Skill | t2m global | t2m region |
|---|---|---|---|
| +6h | 67.88% | 0.78°C | 1.22°C |
| +12h | 74.25% | 0.96°C | 1.53°C |
| +18h | 76.59% | 1.10°C | 1.69°C |
| +24h | 77.86% | 1.20°C | 1.79°C |

#### Nudging (merge)

| Конфиг | +6h Skill |
|---|---|
| 10% α=0.3 seq | 60.46% |
| 10% α=0.5 seq | 60.96% |
| 10% α=0.7 seq | 61.30% |
| 10% α=0.3 offline | 60.46% |
| 1% α=0.3 seq | 59.53% |
| 1% α=0.5 seq | 59.58% |

#### Variable groups (merge, c=10km, 10%)

| Группа | +6h Skill |
|---|---|
| t2m only | 59.49% |
| t + wind | 59.70% |
| surface | 59.74% |
| surface + upper | 59.79% |
| all dynamic | 61.72% |

---

### VIII. MOS/IDW ПОСТПРОЦЕССИНГ (merge, 50 сэмплов)

| Конфигурация | t2m RMSE °C | Skill vs persistence |
|---|---|---|
| Persistence | 7.236 | — |
| **GNN raw** | **2.083** | **71.21%** |
| GNN + lapse | 2.868 | 60.36% |
| GNN + lapse + MOS station | 2.938 | 59.39% |
| +IDW p=2.0 r=50 | 3.582 | 50.50% |
| +IDW p=2.0 r=100 | 4.305 | 40.51% |
| +IDW p=2.0 r=150 | 4.942 | 31.70% |
| +IDW p=2.0 r=200 | 5.517 | 23.75% |
| +IDW p=2.0 r=300 | 6.487 | 10.35% |
| +IDW p=3.0 r=100 | 4.305 | 40.50% |
| +IDW p=3.0 r=150 | 4.943 | 31.68% |
| +IDW p=3.0 r=300 | 6.492 | 10.28% |
| +IDW p=1.5 r=150 | 4.942 | 31.71% |
| +IDW p=1.5 r=300 | 6.485 | 10.38% |

**Вывод:** MOS/IDW ухудшает прогноз на merge-датасете. GNN raw = лучший вариант.

---

### IX. LIVE-ПРОГНОЗЫ (апрель 2026, GDAS/GFS)

**19-station MOS:** MAE 1.66°C vs Open-Meteo (было 2.59 с 9 станциями)
- Cold bias ~1.2°C на t2m
- Ветер занижен на ~3.3 м/с
- Давление: msl vs sp mismatch ~20 гПа

---

### X. СВОДНАЯ ТАБЛИЦА — ВСЕ ПОДХОДЫ

| # | Подход | Датасет | t2m +6h | Skill +6h | Замечание |
|---|---|---|---|---|---|
| 1 | GNN региональная | 61×41, 19f | 2.66°C | 29.7% | потолок ~30% |
| 2 | U-Net V1 | 61×41, 19f | 2.19°C | 55.7% | лучший CNN |
| 3 | **Multires freeze6** | **133K, 19f** | **0.96°C** | **75.8%** | **BEST модель** |
| 4 | freeze6 + OI 10% c=100 | 133K interp | — | 83.8% | interpolate DA |
| 5 | freeze6 + OI 1% c=150 | 133K interp | 0.95°C | 75.8% | sparse interp |
| 6 | freeze6 (merge baseline) | 133K merge | 1.42°C | 59.4% | merge режим |
| 7 | **freeze6 + OI 10% c=100** | **133K merge** | **0.84°C** | **77.9%** | **+18.4 п.п.** |
| 8 | freeze6 + OI 1% c=200 | 133K merge | 1.22°C | 67.9% | +8.5 п.п. |
| 9 | freeze6 + Nudging 10% | 133K merge | — | 61.3% | слабее OI |
| 10 | WRF 1 km | 96×84, d03 | 0.54 K (+6h) | — | 1 образец |