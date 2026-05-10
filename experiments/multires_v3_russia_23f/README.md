# multires_v3_russia_23f

**Тип:** базовая модель каскада (short-range, 0–48 ч, шаг 6 ч).
**Цель:** научить более устойчивую к длинному rollout модель Russia, готовую под каскадную надстройку для 9-дневного прогноза.

## Что нового по сравнению с v2 (`multires_russia_19f`)

### 1. Расширенный набор переменных (19 → 23)
Изменения каналов (см. `scripts/build_multires_dataset_v3.py` — TODO):

| # | имя | тип | что нового |
|---|---|---|---|
| 0 | `t2m` | dyn | weight ×2 |
| 1 | `10u` | dyn | |
| 2 | `10v` | dyn | |
| 3 | `msl` | dyn | weight ×2 |
| 4 | `tp` | dyn | weight ×1.5 |
| 5 | `sp` | dyn | weight ×0.1 (дубликат msl) — **под удаление в v4** |
| 6 | `tcwv` | dyn | |
| 7 | `z_surf` | **static** | вынесено из dynamic loss (carry-forward) |
| 8 | `lsm` | **static** | вынесено из dynamic loss |
| 9–13 | t/u/v/z/q @ 850 | dyn | без изменений |
| 14–16 | t/u/v @ 500 | dyn | |
| 17 | `z@500` | dyn | weight ×1.5 (золотая метрика синоптики) |
| 18 | `q@500` | dyn | |
| **19** | `z@250` | **NEW dyn** | jet stream — обязателен для синоптики 5+ дней |
| **20** | `t@250` | **NEW dyn** | (заменено в v3.1 на time forcing — см. ниже) |
| **21** | `u@250` | **NEW dyn** | |
| **22** | `v@250` | **NEW dyn** | |

**ВНИМАНИЕ — текущая версия конфига:**
В первом запуске v3 (этот config.json) каналы 19–22 — это **time forcing** `[sin_hour, cos_hour, sin_doy, cos_doy]`, чтобы успеть начать обучение на уже доступном датасете без долгой докачки CDS. Уровень 250 hPa добавим в **v3.1** (`multires_v3_russia_27f`) после докачки.

### 2. Time forcing (sin/cos hour + doy) → каналы 19–22
- Решает проблему суточного цикла t2m: сеть знает абсолютное время, не угадывает по двум предыдущим кадрам.
- Помечены как `forcing_channels` → carry-forward'ятся из target (известны заранее, не предсказываются).
- Реализация: `scripts/add_time_features.py` → добавляет 4 канала к `data.npy`.

### 3. Static-каналы (`z_surf`, `lsm`) переведены в `static_channels`
- `loss_weight = 0`, carry-forward из входа в AR-цикле.
- Освобождает capacity сети — раньше она учила «предсказывать константы».

### 4. AR curriculum 1 → 8 шагов (max_ar_steps: 4 → 8)
- Стадии по ~8 эпох: AR=1 (8) → 2 (8) → 3 (8) → 4 (8) → 6 (8) → 8 (24, с patience).
- Учит модель работать с собственными ошибками на горизонте до +48 ч.
- **Главный фактор для подавления взрыва после +24 ч.**
- Требует пересборки датасета с `pred_steps=8` (см. `scripts/build_multires_dataset.py --pred-steps 8`).

### 5. Noise injection (σ=0.05)
- Гауссов шум σ=0.05·σ_data добавляется к `curr_state` на AR-шагах ≥ 1.
- Не зашумляет статику и forcing (channel_mask).
- Регуляризует compound error. Источник: Keisler 2022, GraphCast §3.2.
- Конфиг: `noise_sigma`, `noise_apply_from_ar_step`.
- **TODO:** реализовать в `src/train.py` (4 строки, см. ниже).

### 6. Per-channel loss weights
- `t2m, msl, z@500` ×2 — то, что важно пользователю и MOS-цепочке.
- `tp` ×1.5 — шумный, но критичный.
- `sp` ×0.1 — дубликат msl.
- `z_surf, lsm, time-forcing` ×0 — не учатся.
- В сумме: 1–2 пп скилла бесплатно, без roughness.

## План каскада (для 9-дневного прогноза)

Этот эксперимент — **base** в каскаде:

```
v3_russia_23f (base)        : 0–48 ч, шаг 6 ч, AR=8
        ↓ output как input
v3_russia_long (followup)   : 48–216 ч (9 дн), шаг 12 ч, дообучен на ошибках base
        ↓
MOS на станциях             : per-station × per-horizon
        ↓
weather.arturt.com / API
```

После того как обучится `v3_russia_23f`, делаем:
1. Inference на train-периоде → строим cascade-датасет (input = выход base на t+48ч, target = ERA5 на t+48..+216ч).
2. Обучаем `v3_russia_long` с шагом 12 ч (вдвое меньше итераций → квадратично меньше compound error на 9 днях).

## Что **не** включено (followups)

- **GenCast diffusion на residual** — даёт стохастический ансамбль и калиброванную неопределённость. Слишком тяжёлый для дипломного срока (3–4 недели). Записан как followup в память.
- **Уровень 250 hPa (z, t, u, v)** — критичен для синоптики 5+ дней, но требует докачки CDS (~50 ГБ × 11 лет × Russia ROI). Появится в v3.1.
- **2t-2d (dewpoint depression), sst, sd, stl1, swvl1** — отложены до v4.
- **Spectrum regularization** — спектральный штраф в loss, отложен.

## Команды (для запуска на VM)

### 0. Подготовка датасета
```bash
# На VM (graphcast-5fzkx1)
cd /workdir/graphcast-lite
git pull origin main-arthur

# Пересобрать с pred_steps=8 и time-forcing каналами:
/data/venv/bin/python scripts/build_multires_dataset.py \
  --global-dir /data/datasets/wb2_512x256_19f_ar \
  --region-dir /data/datasets/region_russia_0.25deg_19f \
  --out-dir /data/datasets/multires_russia_23f_tmp \
  --mode interpolate \
  --pred-steps 8

/data/venv/bin/python scripts/add_time_features.py \
  --in-dir /data/datasets/multires_russia_23f_tmp \
  --out-dir /data/datasets/multires_russia_23f
```

### 1. Запуск с pretrained = v2 чекпоинт
```bash
# Скопировать v2 best как starting point для v3 (encoder/processor weights)
cp experiments/multires_russia_19f/best_model.pth \
   experiments/multires_v3_russia_23f/pretrained_v2.pth

setsid nohup /data/venv/bin/python -u -m src.main \
  experiments/multires_v3_russia_23f \
  --pretrained experiments/multires_v3_russia_23f/pretrained_v2.pth \
  > /data/v3_train.log 2>&1 < /dev/null & disown
```

### 2. Мониторинг
```bash
mlc job exec graphcast-5fzkx1 -- bash -lc 'tail -f /data/v3_train.log'
```

## TODO в коде (ещё не реализовано)

1. **`src/train.py`** — добавить noise injection в AR-loop:
```python
# в train_epoch, в for step in range(steps_to_run):
if step > 0 and getattr(config, 'noise_sigma', 0) > 0:
    noise = torch.randn_like(curr_state) * config.noise_sigma
    if channel_mask is not None:
        noise = noise * channel_mask.view(1, 1, 1, -1)
    curr_state = curr_state + noise
```

2. **`src/config.py`** — добавить поля:
```python
noise_sigma: float = 0.0
noise_apply_from_ar_step: int = 1
channel_loss_weights: Optional[Dict[str, float]] = None
```

3. **`src/train.py`** в `weighted_mse_loss` — поддержка `channel_loss_weights` (если задан, домножает на per-channel вектор весов).

4. **`scripts/build_multires_dataset.py`** — добавить `--pred-steps N` (сейчас pred_steps жёстко вшит, см. `multires_krsk_19f_merge` 4 шага). Возможно, уже параметризован — проверить.

5. **`scripts/add_time_features.py`** — добавить опцию `--out-dir` (сейчас, возможно, in-place).

## Журнал

| Дата | Версия | Статус |
|---|---|---|
| 2026-04-25 | v3 (config + README) | создан |
| TBD | v3 train start | — |
| TBD | v3.1 (+250 hPa) | — |
| TBD | v3_long (cascade) | — |
