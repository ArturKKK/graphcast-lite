# Промпт для агента: сайт прогноза погоды

Создай веб-сайт для прогноза погоды Красноярска на базе нейросетевой модели GraphCast-lite. Сайт будет работать на VPS (Ubuntu, 10 ГБ диска, без GPU).

## Что делает система

Каждые 6 часов cron запускает скрипт `scripts/live_gdas_forecast.py`, который:
1. Скачивает 2 свежих цикла GDAS (~700 МБ grib2, удаляются после парсинга)
2. Прогоняет нейросеть (27 МБ модель, CPU inference ~60 сек)
3. Применяет MOS-постобработку (температура + ветер)
4. Сохраняет `forecast.pt` с результатами

Сайт показывает результаты последнего прогноза.

---

## Модель и файлы для деплоя

### Модель GNN (обязательно скопировать на VPS)
- **Конфиг:** `experiments/multires_nores_freeze6/config.json` (~1.6 КБ)
- **Веса:** `experiments/multires_nores_freeze6/best_model.pth` (**27 МБ**)
- Это основная модель-победитель (5.9M параметров, 133K узлов, 12 message passing steps)

### Runtime bundle (обязательно скопировать на VPS)
Директория `live_runtime_bundle/` целиком (~30 МБ), содержит:
- `scalers.npz` — нормализация (mean/std для 19 переменных)
- `coords.npz` — координаты 133K узлов multires-сетки
- `static_fields.npz` — статические поля (рельеф z_surf, маска суша/море lsm)
- `variables.json` — порядок переменных
- `bundle_meta.json` — метаданные
- `learned_mos_t2m_wind_19st.joblib` — **MOS-модель постобработки** (температура + ветер, 19 станций, t2m MAE=1.32°C, wind MAE=0.79 м/с)

### MOS-постобработка: как работает
Скрипт `live_gdas_forecast.py` принимает аргумент `--learned-mos`, путь к joblib-файлу.
MOS — это HistGradientBoostingRegressor (sklearn), обученный на ISD-Lite станциях Красноярского края.
Он корректирует систематические ошибки t2m и ветра (10u/10v).

**Полная команда запуска прогноза:**
```bash
python scripts/live_gdas_forecast.py \
  --experiment-dir experiments/multires_nores_freeze6 \
  --runtime-bundle live_runtime_bundle \
  --learned-mos live_runtime_bundle/learned_mos_t2m_wind_19st.joblib \
  --spatial-idw \
  --lapse-target-elevation 287 \
  --ar-steps 12 \
  --out-dir results/live_latest \
  --cache-dir /tmp/gdas_cache
```

Аргументы:
- `--ar-steps 12` — 12 шагов по 6ч = 72ч прогноз (3 суток)
- `--spatial-idw` — размазывает MOS-коррекцию на все grid-точки через IDW-интерполяцию
- `--lapse-target-elevation 287` — поправка на рельеф (287м = Емельяново)
- `--cache-dir /tmp/gdas_cache` — временные grib2 файлы (удалятся)

---

## Управление дисковым пространством (КРИТИЧНО — только 10 ГБ!)

### Размеры
| Компонент | Размер |
|---|---|
| PyTorch CPU + зависимости | ~1.5 ГБ |
| Модель + bundle + код | ~100 МБ |
| Один forecast.pt | ~370 МБ |
| Временные GDAS grib2 (2 цикла) | ~700 МБ (удаляются!) |
| **Итого постоянно** | **~2.1 ГБ + прогнозы** |

### Ротация прогнозов
**Хранить максимум 2 последних прогноза.** Каждый forecast.pt = ~370 МБ!
При запуске нового прогноза:
1. Переименовать `results/live_latest/` → `results/live_previous/`
2. Удалить `results/live_old/` если есть
3. Смотреть что нового прогноза стало → удалить самый старый

Cron-скрипт ОБЯЗАН:
- Удалять grib2 файлы из cache после парсинга (скрипт делает это сам, если не передать `--keep-grib`)
- Удалять прогнозы старше 2 циклов
- Логировать в `/var/log/graphcast-forecast.log`

### Cron-скрипт: пример логики
```bash
#!/bin/bash
set -e
BASEDIR=/opt/graphcast-lite
RESULTS=$BASEDIR/results

# Ротация: оставляем только 2 последних
rm -rf $RESULTS/live_old
[ -d $RESULTS/live_previous ] && mv $RESULTS/live_previous $RESULTS/live_old
[ -d $RESULTS/live_latest ] && mv $RESULTS/live_latest $RESULTS/live_previous

# Запуск прогноза
cd $BASEDIR
/opt/graphcast-venv/bin/python scripts/live_gdas_forecast.py \
  --experiment-dir experiments/multires_nores_freeze6 \
  --runtime-bundle live_runtime_bundle \
  --learned-mos live_runtime_bundle/learned_mos_t2m_wind_19st.joblib \
  --spatial-idw \
  --lapse-target-elevation 287 \
  --ar-steps 12 \
  --out-dir $RESULTS/live_latest \
  --cache-dir /tmp/gdas_cache

# Удаляем кеш grib
rm -rf /tmp/gdas_cache

# Удаляем самый старый
rm -rf $RESULTS/live_old

# Генерируем JSON для frontend
/opt/graphcast-venv/bin/python $BASEDIR/website/forecast_parser.py \
  --input $RESULTS/live_latest/forecast.pt \
  --output $BASEDIR/website/static/forecast.json
```

---

## Данные из forecast.pt

```python
import torch
data = torch.load("forecast.pt", map_location="cpu", weights_only=False)
# data["prediction_physical"] — shape (133279, 12, 19) — все узлы × 12 AR-шагов × 19 переменных
# data["var_names"] — ["t2m", "10u", "10v", "msl", "tp", "sp", "tcwv", "z_surf", "lsm",
#                       "t@850", "u@850", "v@850", "z@850", "q@850",
#                       "t@500", "u@500", "v@500", "z@500", "q@500"]
# data["latitudes"], data["longitudes"] — координаты 133279 узлов (numpy arrays)
# data["cycles"] — список datetime объектов (2 цикла инициализации, UTC)
# data["learned_mos_applied"] — True/False
# data["warnings"] — список предупреждений
#
# Как извлечь данные для Красноярска:
CITY_BBOX = (55.5, 56.5, 92.0, 94.0)  # lat_min, lat_max, lon_min, lon_max
lat = data["latitudes"]
lon = data["longitudes"]
mask = (lat >= 55.5) & (lat <= 56.5) & (lon >= 92.0) & (lon <= 94.0)
city_pred = data["prediction_physical"][mask]  # shape (N_city, 12, 19)
city_mean = city_pred.mean(axis=0)  # (12, 19) — среднее по городу

# Переменные (индексы):
# 0: t2m (Кельвины → вычесть 273.15 для °C)
# 1: 10u (м/с, восточная компонента ветра)
# 2: 10v (м/с, северная компонента ветра)
# 3: msl (давление на уровне моря, уже в гПа!)
# 4: tp (осадки, м → мм * 1000)
#
# Скорость ветра: sqrt(10u² + 10v²)
# Направление ветра (откуда дует): (270 - atan2(10v, 10u) * 180/π) % 360
#
# Время прогноза:
# last_cycle = data["cycles"][-1]  # последний цикл (datetime UTC)
# +6ч, +12ч, ..., +72ч — горизонты: last_cycle + timedelta(hours=6*(step+1))
# Красноярск = UTC+7
```

## forecast_parser.py — парсит forecast.pt в JSON для фронтенда

Напиши скрипт `website/forecast_parser.py`, который:
1. Загружает `forecast.pt`
2. Извлекает данные для Красноярска (CITY_BBOX)
3. Усредняет по городскому bbox
4. Формирует JSON:
```json
{
  "generated_at": "2026-04-16T18:30:00Z",
  "last_cycle": "2026-04-16T12:00:00Z",
  "forecast": [
    {
      "valid_time_utc": "2026-04-16T18:00:00Z",
      "valid_time_krsk": "2026-04-17T01:00:00+07:00",
      "horizon_hours": 6,
      "t2m_celsius": 3.2,
      "wind_speed_ms": 4.1,
      "wind_direction_deg": 225,
      "wind_direction_text": "ЮЗ",
      "pressure_hpa": 1012.5,
      "precip_mm": 0.1
    },
    ...
  ]
}
```
5. Сохраняет в `website/static/forecast.json`

---

## Архитектура сайта

**Backend:** Python FastAPI, минимальный.
- `GET /` — отдаёт `static/index.html`
- `GET /api/forecast` — отдаёт `static/forecast.json` (сгенерированный парсером)
- `GET /api/status` — когда последний прогноз, какие циклы, здоровье системы
- Статические файлы из `static/`

**Frontend:** Одностраничное приложение (vanilla JS, без фреймворков — минимум зависимостей).
- Красивая карточка с текущим прогнозом для Красноярска
- Таблица погоды на 3 суток (72ч, каждые 6 часов)
- Переменные: t2m (°C), ветер (м/с + направление стрелкой), давление (гПа), осадки (мм)
- График температуры (Chart.js CDN)
- Время отображать в красноярском времени (UTC+7)
- Информация "последнее обновление", "следующее обновление через..."
- Иконки погоды (день/ночь, можно простые Unicode: ☀️🌤️🌧️❄️)

## Структура проекта

Создай всё в директории `website/`:
```
website/
  app.py              # FastAPI backend
  forecast_parser.py  # парсинг forecast.pt → forecast.json
  cron_forecast.sh    # cron-скрипт (полный pipeline: прогноз → парсинг → ротация)
  static/
    index.html
    style.css
    app.js
    forecast.json     # генерируется парсером, НЕ коммитить
  requirements.txt    # fastapi, uvicorn (torch не нужен для сервера!)
  deploy.md           # инструкция по деплою на VPS
```

**Важно:** torch нужен только для `forecast_parser.py` и `live_gdas_forecast.py`, НЕ для веб-сервера. FastAPI просто отдаёт `forecast.json`.

## Дизайн

- Тёмная тема, современный минималистичный дизайн
- Адаптивный (mobile-friendly)
- Город: Красноярск
- Показывать логотип/название "GraphCast-lite Weather"
- Отображать disclaimer: "Экспериментальный прогноз на базе нейросетевой модели. Не является официальным метеопрогнозом."
- Подвал: "Модель: GraphCast-lite (GNN, 5.9M параметров) + MOS-коррекция по 19 станциям"

## Требования к VPS

- Ubuntu 22.04+
- Один venv с CPU-only PyTorch для прогноза
- Отдельный venv (или тот же) с FastAPI + uvicorn для сервера
- CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Также: `pip install xarray cfgrib eccodes scipy scikit-learn joblib requests matplotlib`
- Cron: каждые 6 часов (01:00, 07:00, 13:00, 19:00 UTC — с запасом на публикацию GDAS)
- Systemd unit для uvicorn (порт 8000, за nginx если нужно)
- **Ротация: хранить только 2 последних прогноза** (каждый ~370 МБ!)

## Безопасность

- Никаких exec(), eval()
- CORS: allow only same origin (или конкретный домен)
- Rate limit на API (slowapi или nginx)
- forecast.json — статический файл, никакого динамического парсинга на каждый запрос
- weights_only=False при torch.load — это OK, т.к. мы загружаем свои собственные файлы

## Deploy: что скопировать на VPS

```
# На VPS в /opt/graphcast-lite/:
experiments/multires_nores_freeze6/config.json
experiments/multires_nores_freeze6/best_model.pth     # 27 МБ
live_runtime_bundle/                                   # ~30 МБ (все файлы)
scripts/live_gdas_forecast.py                          # скрипт прогноза
src/                                                   # весь src (модель, постобработка)
website/                                               # сам сайт
requirements.txt                                       # зависимости
```
