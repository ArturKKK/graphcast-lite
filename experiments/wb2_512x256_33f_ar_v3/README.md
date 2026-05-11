# wb2_512x256_33f_ar_v3 — global v3

**Тип:** глобальная базовая модель, обучается **с нуля** (no pretrained).

## Что входит (33 канала = 19 base + 10 plev + 4 time)

| idx | имя | source |
|-----|-----|--------|
| 0-18 | t2m, 10u, 10v, msl, tp, sp, tcwv, z_surf, lsm, t/u/v/z/q@850, t/u/v/z/q@500 | base 19f (data.npy) |
| 19-23 | z/t/u/v/q @ 250 hPa | data_extra.npy (10ch, индексы 0-4) |
| 24-28 | z/t/u/v/q @ 1000 hPa | data_extra.npy (индексы 5-9) |
| 29-32 | sin_hour, cos_hour, sin_doy, cos_doy | data_extra.npy (индексы 10-13, computed) |

**NB**: `2d` (dewpoint) НЕ доступен в WB2 zarr 512x256_equiangular_conservative → 29 dyn вместо 30 (см. комментарий в [extend_dataset_512x256_to_30f.py](../../scripts/extend_dataset_512x256_to_30f.py)).

## Что нового по сравнению с v2 (`wb2_512x256_19f_ar_v2`)

| | v2 | v3 |
|---|---|---|
| Каналов | 19 | **33** |
| Уровни давления | 500, 850 | + **250** (струйные течения) + **1000** (приземный) |
| Time-forcing | — | **sin/cos hour/doy** (модель явно "знает" солнце+сезон) |
| `max_ar_steps` | 4 | **8** (rollout 0–48 ч) |
| `forcing_channels` | — | `[29-32]` carry-forward, zero loss |
| `static_channels` | — | `[7, 8]` (z_surf, lsm) carry-forward, zero loss |
| Архитектура | hidden=256, mesh [4,6], 12 MP | то же |

**Зачем новые каналы (по-человечески):**
- **250 hPa** — высота струйных течений (~10 км). Без них модель не "видит" быстрых атмосферных рек, которые управляют циклонами на средних широтах.
- **1000 hPa** — самый низ атмосферы. Помогает в граничном слое: суточный нагрев/выхолаживание, бризы, температурные инверсии.
- **sin/cos hour + sin/cos doy** — модель явно знает где солнце и какой сезон, не угадывает суточный цикл из шумных данных.

## Что хотел добавить, но в коде нет

- `noise_sigma` (input noise injection с AR≥1) — в `src/train.py` нет реализации.
- `channel_loss_weights` per-channel (t2m ×2, tp ×1.5) — в `weighted_mse_loss` поддерживается только 0/1 mask через static/forcing.

Эти улучшения требуют изменения `src/train.py` (пользователь просил код не трогать). Сейчас веса каналов = 1.0 для всех **динамических** и 0.0 для статических/forcing (что уже даёт корректный train signal — модель не учится "предсказывать" lsm/z_surf/time).

## Связывание данных (на VM)

Loader (`src/data/dataloader_chunked.py:_ConcatMemmap`) умеет склеивать `data.npy` + `data_extra.npy` на лету по last-axis. Поэтому:

```
/data/datasets/wb2_512x256_33f_v3/
  data.npy           → SYMLINK на base 19f data.npy (~80 GB на диске)
  data_extra.npy     → НОВЫЙ файл 14ch (10 plev + 4 time), ~60 GB
  scalers.npz        → mean/std (33,)
  variables.json     → 33 имени
  coords.npz         → копия из base
  dataset_info.json  → n_feat=33, n_feat_base=19, n_feat_extra=14
```

Готовится через [scripts/build_v3_extra_with_time.py](../../scripts/build_v3_extra_with_time.py):
- читает base `data.npy` (19ch), создаёт symlink в out_dir
- читает extra `data_extra.npy` (10 plev)
- вычисляет 4 time-forcing канала из `time_start` (2010-01-01, 6h step)
- chunk-копирует 10+4 в новый `data_extra.npy` (т.е. меняем "extra с 10ch" на "extra с 14ch")
- собирает scalers = base(19) + extra(10) + time(4)

## Запуск на VM

```bash
nohup bash /data/run_v3_global.sh > /dev/null 2>&1 &
mlc job exec <VM> -- bash -lc 'tail -f /data/logs/v3_global_pipeline.log'
```

## Журнал

| Дата | Статус |
|---|---|
| 2026-05-11 | config + pipeline готовы, 33f setup. Pre-launch sanity check. |
