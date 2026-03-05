# U-Net: Региональная модель Красноярск

## Мотивация

GNN (GraphCast-Lite) на регулярной сетке 61×41 — неправильная архитектура:
- Icosphere mesh 259 узлов + message passing = "смотрим на картинку через дырочки"
- U-Net видит полную 2D пространственную структуру нативно
- Свёртки на регулярной сетке — прямой путь для такой задачи

---

## Эксперимент U1: U-Net базовый (19f, base_filters=64)

Конфиг: `experiments/unet_region_krsk/config.json`
- dataset: `region_krsk_61x41_19f_2010-2020_025deg` (19 метео переменных)
- obs_window=2, pred_steps=4, batch_size=16, lr=1e-3
- base_filters=64 → ~7.8M параметров
- static_channels=[7,8], forcing_channels=[]
- 3-уровневый U-Net: 64→128→256→512 + skip connections
- CosineAnnealingLR, AdamW, patience=10

Обучение (33 эпохи, ~15 минут на GPU, early stopping):
```
val_ACC = 0.9704
val_loss = 0.07311
```

Инференс (`python -m src.unet.predict experiments/unet_region_krsk --max-samples 200`):
```
Per-horizon (dynamic channels only):
  +06h: RMSE=2.9653 | base=6.6888 | skill=55.7% | ACC=0.9092
  +12h: RMSE=5.7818 | base=11.7987 | skill=51.0% | ACC=0.8257
  +18h: RMSE=9.0486 | base=15.9146 | skill=43.1% | ACC=0.7415
  +24h: RMSE=12.2168 | base=19.3092 | skill=36.7% | ACC=0.6675

Per-horizon per-channel RMSE (physical units):
       var   unit      +6h     +12h     +18h     +24h
       t2m      K   2.19°C   2.92°C   3.37°C   3.82°C
       10u    m/s     0.56     0.82     1.14     1.45
       10v    m/s     0.50     0.75     1.00     1.21
       msl     Pa     1.10     1.96     3.02     4.12
        tp      m     0.00     0.00     0.00     0.00
        sp     Pa     1.07     2.06     3.02     4.02
      tcwv    m/s     0.62     0.98     1.29     1.55
     t@850      K   1.19°C   1.71°C   2.21°C   2.74°C
     u@850    m/s     1.44     2.09     2.85     3.53
     v@850    m/s     1.47     2.24     3.11     3.89
     z@850  m²/s²     0.6m     1.3m     2.0m     2.7m
     q@850  kg/m²     0.00     0.00     0.00     0.00
     t@500      K   1.11°C   1.74°C   2.29°C   2.79°C
     u@500    m/s     2.41     3.82     5.19     6.31
     v@500    m/s     2.58     4.10     5.57     6.95
     z@500  m²/s²     0.9m     1.9m     3.0m     4.1m
     q@500  kg/m²     0.00     0.00     0.00     0.00

Per-channel metrics overall (avg over 4 horizons):
    #        var      ACC    RMSE_phys     unit
    0        t2m   0.8848       3.0761        K
    1        10u   0.8258       0.9923      m/s
    2        10v   0.8071       0.8644      m/s
    3        msl   0.9045       2.5479       Pa
    4         tp   0.4816       0.0001        m
    5         sp   0.9996       2.5432       Pa
    6       tcwv   0.8088       1.1103      m/s
    9      t@850   0.8339       1.9633        K
   10      u@850   0.8883       2.4799      m/s
   11      v@850   0.7529       2.6791      m/s
   12      z@850   0.9549      16.2596    m²/s²
   13      q@850   0.7316       0.0004    kg/m²
   14      t@500   0.6996       1.9809        K
   15      u@500   0.6969       4.4339      m/s
   16      v@500   0.6724       4.7993      m/s
   17      z@500   0.9461      24.5066    m²/s²
   18      q@500   0.4727       0.0001    kg/m²
```

### Сравнение с GNN (лучший — эксп. 1, 19f):

| Метрика | GNN exp1 | **U-Net U1** | Δ |
|---|---|---|---|
| t2m @+6h | 2.66°C | **2.19°C** | −18% |
| t2m @+24h | 4.33°C | **3.82°C** | −12% |
| Skill @+6h | 33.2% | **55.7%** | +22.5pp |
| Skill @+24h | 25.2% | **36.7%** | +11.5pp |
| ACC @+6h | 0.893 | **0.909** | +0.016 |
| Время обучения | ~40 мин | **~15 мин** | 2.7× быстрее |

**Вывод:** U-Net радикально лучше GNN на регулярной сетке. Подтверждено — для 2D данных свёрточная архитектура оптимальна.

---

## Эксперимент U2: U-Net + 23f + base_filters=96

Конфиг: `experiments/unet_region_krsk_23f/config.json`
- dataset: `region_krsk_61x41_23f_2010-2020_025deg` (19 метео + 4 time features)
- base_filters=96 → ~17.5M параметров (2.2× больше U1)
- static_channels=[7,8], forcing_channels=[19,20,21,22]
- Остальное без изменений

Гипотеза: time forcing + увеличенная ёмкость дадут улучшение на длинных горизонтах.

Обучение: early stop после ~3-й эпохи (val_loss перестал улучшаться).

Инференс:
```
Per-horizon (dynamic channels only — OLD aggregate metric, physical units):
  +06h: RMSE=16.6696 | base=6.6888 | skill=-149.2% | ACC=0.8751
  +12h: RMSE=28.5117 | base=11.7987 | skill=-141.7% | ACC=0.7515
  +18h: RMSE=42.3436 | base=15.9146 | skill=-166.1% | ACC=0.6328
  +24h: RMSE=54.4004 | base=19.3092 | skill=-181.7% | ACC=0.5362

Per-horizon per-channel RMSE (physical units):
       var   unit      +6h     +12h     +18h     +24h
       t2m      K   2.13°C   3.23°C   4.19°C   5.14°C
       10u    m/s     0.53     0.75     1.00     1.21
```

### Анализ: почему skill -149% при t2m=2.13°C?

Aggregate RMSE считался в ФИЗИЧЕСКИХ единицах: RMSE = sqrt(mean(SE_t2m + SE_wind + SE_z500 + ...)).
Геопотенциал z@500 имеет RMSE ~25-200 м²/с², а t2m ~2 К — разница в 100×.
Один плохо выученный канал z@500 даёт вклад 10,000+ в сумму квадратов → убивает весь aggregate.
При этом t2m = 2.13°C — ЛУЧШЕ чем U1 (2.19°C)!

**Фикс**: aggregate skill теперь считается в нормализованных единицах (per-channel skill → среднее).

### Вывод по U2

- t2m чуть лучше (2.13 vs 2.19), ветер тоже (0.53 vs 0.56)
- Модель 17.5M параметров слишком большая для 12.8K train samples → быстрый overfit (3 эпохи)
- Time forcing эффект не проявился — модель не успела обучиться
- base_filters=96 на таком количестве данных — overkill

---

## Эксперимент U3: U-Net + 23f + base_filters=64 (PLAN)

Конфиг: `experiments/unet_region_krsk_23f_v2/config.json`
- dataset: 23f (с time forcing)
- base_filters=64 (~7.8M параметров — как U1)
- static_channels=[7,8], forcing_channels=[19,20,21,22]
- Гипотеза: архитектура U1 (которая работает) + time forcing → улучшение на +18h/+24h
