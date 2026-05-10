# wb2_512x256_34f_ar_v3 — global v3

**Тип:** глобальная базовая модель, обучается **с нуля** (no pretrained).
**Цель:** улучшенная версия `wb2_512x256_19f_ar_v2`: 30 динамических каналов (вместо 19) + 4 time-forcing = **34 features**, AR=8, noise injection, per-channel loss weights.

## Что нового по сравнению с v2 (по-человечески)

v2 знал погоду на двух высотных уровнях (500 и 850 hPa) и 7 поверхностных полей. v3 знает **больше физики**:

| Категория | v2 (19f) | v3 (34f) |
|---|---|---|
| Поверхностные | t2m, 10u, 10v, msl, tp, sp, tcwv | + **2d (точка росы)** — влажность у земли |
| Уровни | 500, 850 hPa | + **250 hPa** (струйные течения, вверх) + **1000 hPa** (приземный слой) |
| Каналы на уровень | t/u/v/z/q (5) | те же 5 |
| Time forcing | — | **sin/cos часа + sin/cos дня года** (4 канала) |
| Static | z_surf, lsm | те же 2 |
| **Всего** | 19 | **34** = 8 surf + 2 static + 5×4 plev + 4 time |

**Зачем новые каналы (объяснение для не-гига-умных):**
- **2d (dewpoint)** — точка росы. Разница `t2m − 2d` даёт относительную влажность → модель лучше предсказывает туманы, осадки, конденсацию.
- **250 hPa** — это высота струйных течений (≈10 км). Без них модель не "видит" быстрых атмосферных рек, которые управляют циклонами на средних широтах.
- **1000 hPa** — это самый низ атмосферы (≈100 м над уровнем моря). Помогает в граничном слое — суточный нагрев/выхолаживание, бризы, температурные инверсии.
- **time-forcing (sin/cos hour/doy)** — модель "знает" где сейчас солнце и какой сезон. Без этого она вынуждена угадывать суточный цикл из самих данных, а это шумно и плохо обобщается на новые годы.

В сумме: модель должна стать сильно точнее по осадкам, ветру в свободной атмосфере и суточному ходу температуры.

## Что ещё добавил из training tricks

| | v2 | v3 |
|---|---|---|
| `max_ar_steps` | 4 | **8** (учим 0–48 ч rollout — больше горизонт) |
| `noise_sigma` | 0 | **0.05** (даём модели "грязный" вход с AR≥1, чтоб не разъезжалась на autoregressive) |
| `channel_loss_weights` | uniform | **per-channel**: t2m/msl/z@500 ×2, tp ×1.5, sp ×0.1, статика/forcing ×0 |
| `forcing_channels` | — | `[30,31,32,33]` (carry-forward, не учит их предсказывать — они известны точно) |
| `static_channels` | — | `[7, 8]` (z_surf, lsm — заморожены в AR-rollout) |
| Pretrained | — | — (от нуля) |
| Архитектура | hidden=256, mesh [4,6], 12 MP-steps | **то же** (тот же бэкбон) |

## Связывание data.npy (вопрос пользователя)

На VM нужно получить **ОДИН** итоговый `/data/datasets/wb2_512x256_34f_v3/data.npy`. Возможны три сценария от того что лежит в S3:

### Сценарий A: один архив `dataset_512x256_34f.tar.zst`
Уже всё собрано через `scripts/build_dataset_512x256_30f.py` + `add_time_features.py` локально или на VM ранее. Просто распаковываем.
```
/data/datasets/wb2_512x256_34f_v3/data.npy   (~140-150 GB)
```
Использовать `--scenario one_34f` (см. ниже, нужно будет добавить в `_mlc_run_v3_global.sh` если архив прямо такой).

### Сценарий B: два архива `dataset_512x256_30f_part1.tar.zst` + `_part2.tar.zst`
Например part1 = 2010–2015 (6 лет), part2 = 2016–2021 (6 лет). Нужна конкатенация по time-axis, потом +time_features.
```
data_30f_part1/data.npy (T1, 512, 256, 30)
data_30f_part2/data.npy (T2, 512, 256, 30)
       ↓ scripts/concat_time_chunks.py
data_30f/data.npy       ((T1+T2), 512, 256, 30)
       ↓ scripts/add_time_features.py
data_34f/data.npy       ((T1+T2), 512, 256, 34)
```
Использовать `--scenario two_time_chunks_30f`.

### Сценарий C: `dataset_512x256_19f.tar.zst` + `dataset_512x256_11f_extra.tar.zst` (delta-каналы отдельно)
Как делает `build_dataset_512x256_30f.py` с `--base-dir`: добирает только 11 новых.
```
/data/datasets/wb2_512x256_19f_ar/data.npy   (T, 512, 256, 19)
/data/datasets/wb2_512x256_11f_extra/data.npy (T, 512, 256, 11)
       ↓ concat feat-axis (scripts/concat_feat_chunks.py)
data_30f/data.npy                            (T, 512, 256, 30)
       ↓ add_time_features.py
data_34f/data.npy                            (T, 512, 256, 34)
```
Использовать `--scenario two_feat_chunks_30f`.

**TL;DR**: дай мне завтра имена архивов в S3 — отредактирую `scripts/_mlc_run_v3_global.sh` под нужный сценарий. Если хочешь — могу собрать всё в один шаг ещё на VM с нуля через `build_dataset_512x256_30f.py` (но это качать 12 лет ERA5 заново).

## Запуск на VM

```bash
# 1. Залить архивы в /data/datasets/
# 2. Запустить:
SCENARIO=<выбранный> nohup bash /data/run_v3_global.sh > /dev/null 2>&1 &

# Мониторинг:
mlc job exec <VM> -- bash -lc 'tail -f /data/logs/v3_global_pipeline.log'
```

## TODO (зависит от реализованности в коде)

- [ ] Убедиться что `src/main.py` поддерживает `noise_sigma`, `noise_apply_from_ar_step`, `channel_loss_weights`, `forcing_channels`, `static_channels` (для regional v3 уже было сделано; для global wb2 проверить).
- [ ] Если в коде ещё нет — пробросить из конфига перед запуском.

## Журнал

| Дата | Статус |
|---|---|
| 2026-05-11 | config + README созданы (34f setup) |
