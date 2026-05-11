# wb2_512x256_19f_ar_v2 — апгрейд архитектуры: InteractionNetwork + Swish + hidden 256

Те же 19 переменных на той же сетке 512×256, что и в `wb2_512x256_19f_ar`, но **полностью переделан processor**: вместо стека GCN-свёрток теперь GraphCast-style **InteractionNetwork с 12 шагами message passing**, edge-features, Swish-активации, удвоенная скрытая размерность.

## Что было

| Параметр | Значение |
|---|---|
| Датасет | `wb2_512x256_19f_ar_v2` (тот же набор 19 переменных, что и v1) |
| Разрешение сетки | 512 × 256 |
| Число переменных | 19 |
| `obs_window_used` | 2 (12 ч истории) |
| `pred_window_used` | 1 (AR разворачивает loop) |

Состав 19 переменных — идентичен `wb2_512x256_19f_ar` (см. его README).

## Как обучалось

| Параметр | Значение |
|---|---|
| `batch_size` | 1 |
| `learning_rate` | **3e-4** (ниже, чем 5e-4 в v1 — модель крупнее) |
| `num_epochs` | 80 (vs 60 в v1) |
| `early_stopping_patience` | **15** (vs 12) |
| `use_latitude_weighting` | true |
| `max_ar_steps` | 4 |

## Архитектура

- **Граф**: тот же — `mesh_levels=[4, 6]`, radius 0.6 / contained
- **Encoder**:
  - MLP `[256, 256]→256` (vs `[128,128]→128`)
  - GCN `[256, 256]→256` с **Swish**
- **Processor**: **`interaction_net`** — GraphCast-style InteractionNetwork:
  - `output_dim=256`
  - **`num_message_passing_steps=12`** (vs 4 слоя GCN в v1)
  - **`edge_feature_dim=4`** (рёбра несут признаки, в v1 рёбра были «голые»)
  - Swish + layer-norm
- **Decoder**:
  - MLP `[256, 128]→128`
  - GCN `[128, 128]→**19**` с Swish
- Скрытая размерность — **256** (vs 128 в v1).

## Что нового по сравнению с `wb2_512x256_19f_ar`

| | v1 (19f_ar) | **v2 (19f_ar_v2)** |
|---|---|---|
| Processor type | 4× `conv_gcn` | **InteractionNetwork, 12 MP-шагов** |
| Edge features | ❌ | **✅ (`edge_feature_dim=4`)** |
| Hidden dim | 128 | **256** |
| Активация | (по умолчанию, PReLU/ReLU) | **Swish** во всех графовых блоках |
| Encoder MLP | [128,128]→128 | [256,256]→256 |
| Decoder MLP | [128,64]→64 | [256,128]→128 |
| Learning rate | 5e-4 | 3e-4 |
| Эпох | 60 | 80 |
| Early-stop patience | 12 | 15 |

### Почему важен переход на InteractionNetwork
- В `conv_gcn` рёбра — просто «кто с кем связан». В **InteractionNetwork** каждое ребро хранит свой `edge_feature_dim=4` вектор (расстояние, направление, угол), а обновление узлов идёт через явный message-функция `f(node_i, node_j, edge_ij)`. Это даёт модели понятие геометрии меша.
- 12 шагов MP против 4 слоёв GCN → информация может пройти по графу гораздо дальше за один forward.

## Что всё ещё отсутствует

- Нет `static_channels` / `forcing_channels` — `z_surf` и `lsm` учатся как динамика.
- Нет input-noise injection.
- Нет per-channel loss weights.
- Только 19 каналов (нет верхних уровней, нет приземного 1000 гПа, нет time-forcing).

Все эти пункты добавлены в `wb2_512x256_33f_ar_v3`.

## Роль в проекте

`wb2_512x256_19f_ar_v2` — это **pretrained global** для региональных multires-моделей. Например, эксперимент `multires_merge_freeze6_v2` использует его веса как стартовую точку и дообучается на merge-датасете Russia (см. memory `861bd6ab...` про `roi_only_loss=false`).
