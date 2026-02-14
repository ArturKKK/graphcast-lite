# Changelog: v1 → v2 архитектуры

> Дата: 14 февраля 2025
> Эксперимент v1: `wb2_512x256_19f_ar`
> Эксперимент v2: `wb2_512x256_19f_ar_v2`
> Датасет: один и тот же (512×256, 19 переменных, 2010-2021, ERA5)

---

## Что изменилось

### 1. Processor: GCNConv → InteractionNetwork

| | v1 | v2 |
|---|---|---|
| GNN тип | `GCNConv` (только узлы) | `InteractionNetwork` (узлы + рёбра) |
| Message passing steps | 4 (+ 1 выходной слой) | **12** (unshared weights) |
| Edge features | ❌ Нет | ✅ 4D: distance + 3D relative position в локальных координатах получателя |
| Residual connections | ❌ Нет | ✅ Node + Edge residuals на каждом шаге |
| LayerNorm | 1 раз в конце | На каждом шаге (node + edge) |

**Почему это важно:**
- InteractionNetwork обновляет **и рёбра, и узлы** — модель "знает" расстояния и направления
- 12 шагов × residuals = информация распространяется по всему mesh (глобальный рецептивный фон)
- Без residuals невозможно обучить 12 слоёв (vanishing gradients, oversmoothing)

### 2. Latent dimension: 128 → 256

| | v1 | v2 |
|---|---|---|
| Encoder MLP | [128, 128] → 128 | [256, 256] → 256 |
| Encoder GCN | [128, 128] → 128 | [256, 256] → 256 |
| Processor | 128 | 256 |
| Decoder MLP | [128, 64] → 64 | [256, 128] → 128 |
| Decoder GCN | [64, 64] → 19 | [128, 128] → 19 |

**Почему:** 2× шире латентное пространство = больше capacity для сложных паттернов.

### 3. Activation: PReLU → Swish (SiLU)

| | v1 | v2 |
|---|---|---|
| Encoder/Decoder GCN | PReLU | **Swish** (SiLU) |
| Processor | PReLU (внутри GCN) | **Swish** (InteractionNet MLP) |

**Почему:** Swish — то, что использует оригинальный GraphCast. Гладкая производная, лучше для глубоких сетей.

### 4. Training hyperparameters

| | v1 | v2 |
|---|---|---|
| Learning rate | 5e-4 | **3e-4** |
| Epochs | 60 | **80** |
| Early stopping patience | 12 | **15** |
| Max AR steps | 4 | 4 |

**Почему:** Более глубокая модель требует чуть меньший LR и больше эпох для сходимости.

### 5. Параметры модели

| | v1 | v2 |
|---|---|---|
| Примерное число параметров | ~210K | **~5.9M** |
| Соотношение | 1× | **28×** |

Для сравнения: Google GraphCast = 36.7M (ещё 6× больше v2).

---

## Что НЕ изменилось

- **Датасет**: тот же (512×256, 19 vars, 12 лет, float16/float32)
- **Mesh topology**: [4, 6] уровни (10,242 mesh-узлов)
- **Encoder/Decoder**: GCNConv (но с увеличенным latent и swish)
- **AR curriculum**: 4 стадии (1→2→3→4 шага)
- **Latitude weighting**: включено
- **Grid resolution**: 512×256 (~0.7°)
- **obs_window**: 2 (два входных кадра)

---

## Как сравнивать результаты

### Ключевые метрики для сравнения:
1. **val_ACC** (spatial correlation) — основная метрика качества
2. **val_loss** (weighted MSE) — основная метрика лосса
3. **Время одной эпохи** — для оценки compute cost
4. **Эпоха best model** — когда модель сошлась

### Что ожидаем от v2:
- **val_ACC**: значимо выше чем v1 (v1 = 0.971 на epoch 22, AR=2)
- **train_loss**: может быть ниже за счёт большей capacity
- **Время эпохи**: ~3-5× дольше чем v1 (больше параметров, 12 msg steps вместо 4)
- **Общее время**: ~10-14 дней на A100

### Остаточные различия с Google GraphCast (после v2):
- Latent 256 vs 512 (2×)
- 12 msg steps vs 16 (1.3×)
- 2 pressure levels vs 37 (18×)
- 19 vars vs 227 (12×)
- 12 лет vs 39 лет (3×)
- ~5.9M vs 36.7M params (6×)

---

## Как запустить v2

```bash
python -m src.main --experiment wb2_512x256_19f_ar_v2
```

Лог обучения будет писаться в:
```
experiments/wb2_512x256_19f_ar_v2/training_log.txt
```
