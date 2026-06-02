# GraphCast-lite — графовая нейросетевая 2.5D-модель краткосрочного прогноза погоды

Гибридная схема краткосрочного прогноза погоды над территорией России, сочетающая графовую нейросетевую 2.5D-модель с усвоением данных метеорологических наблюдений и нейросетевой постобработкой по сети станций.

Репозиторий — рабочее окружение выпускной квалификационной работы бакалавра «**Разработка алгоритма усвоения данных наблюдений для графовой нейросетевой 2.5D-модели краткосрочного прогноза погоды**» (НГУ, ММФ, кафедра вычислительной математики, 2026). Автор: Табаков А. С. Научный руководитель: д.ф.-м.н. Пененко А. В. (ИВМиМГ СО РАН).

---

## Идея проекта

При авторегрессионном применении графовых нейросетевых моделей прогноза погоды возникают две характерные трудности:

1. **Накопление ошибки** с ростом горизонта прогноза.
2. **Систематический сдвиг** модельных полей относительно станционных наблюдений.

Проект реализует гибридную схему, в которой эти трудности адресуются на разных уровнях:

- **Графовая 2.5D-модель** даёт основу — авторегрессионный прогноз состояния атмосферы на регулярной сетке.
- **Усвоение данных на этапе прогноза** (Nudging и Optimal Interpolation) корректирует прогнозное состояние в сторону наблюдений на каждом шаге.
- **Нейросетевая постобработка** с обучаемой станционной поправкой снимает остаточный сдвиг прогноза в точках конкретных метеостанций.

Итоговая схема работает в оперативном режиме на свежих анализах NOAA GDAS и обновляет прогноз четыре раза в сутки.

---

## Основные компоненты

### Базовая модель: GraphCast-lite

Графовая нейросеть формата «кодировщик — процессор — декодер» по образцу Google GraphCast, обученная на реанализе ERA5:

- **Кодировщик** — проекция полей с регулярной сетки на узлы икосаэдрального меша.
- **Процессор** — 12 шагов обмена сообщениями (Interaction Network) с обновлением рёбер и узлов через MLP.
- **Декодер** — обратная проекция прогноза на регулярную сетку.

Скрытая размерность 256, ~5.9 М параметров (на порядок меньше оригинального Google GraphCast, отсюда «lite»).

### 2.5D-архитектура

Узлы графа размещены только на 2D-сетке (широта × долгота), но в каждом узле — каналы для приземного уровня и для уровней давления 850 и 500 гПа. Это даёт вертикальную информацию без полноценного 3D-меша.

### Мультимасштабная сетка над Россией

Глобальная подсетка ~131 тыс. узлов (0.7°) объединяется с детализированной региональной подсеткой ~106 тыс. узлов (0.25°) над Россией. Узлы двух масштабов связываются общими рёбрами в зоне перекрытия. Кодировщик, процессор и декодер работают с единым графом — без искусственных границ региона.

### Усвоение данных

На каждом шаге авторегрессионного прогноза модель корректирует своё состояние по разрежённой сети наблюдений. Реализованы два известных алгоритма, адаптированных под графовую модель:

- **Nudging** — итеративная релаксация модельного поля к наблюдениям в узлах со станциями.
- **Optimal Interpolation** — статистически оптимальное взвешивание прогноза и наблюдений с учётом пространственных корреляций ошибок. По результатам подбора гиперпараметров оптимум — радиус корреляции $L = 100$ км, дисперсия ошибки наблюдений $\sigma_o = 0.5$.

### Нейросетевая постобработка (v3)

На выходе модели — добавочный MLP с обучаемой станционной поправкой:

$$\hat y = y_{\text{GNN}} + \Delta_{\text{MLP}}(\text{признаки}, \text{станция}, \text{горизонт}) + b_{\text{станция}}$$

- 27 признаков: приземные и верхнеуровневые поля модели, лапс-рейт, точка росы, час, день года, лид, рельеф;
- `Embedding(689 × 3)` — аддитивная поправка на каждую из 689 станций для T2m, U10, V10;
- обучение на ретроспективе ERA5 ↔ ISD-Lite за 2018 и 2020 годы;
- веса: [experiments/neural_postproc_v3/best_model.pth](experiments/neural_postproc_v3/best_model.pth).

---

## Эксперименты

Основные обученные модели:

| Эксперимент | Описание | Лучшая модель |
|---|---|---|
| `wb2_512x256_19f_ar_v2` | Глобальная модель v2 (InteractionNet, 256d, 12 шагов) | `best_model.pth` |
| `wb2_512x256_33f_ar_v3` | Глобальная модель v3 (33 канала: + 250/1000 гПа, time-forcing, channel weights) | `best_model.pth` |
| `multires_russia_33f_v3_noroi` | Мультимасштабная модель над Россией (33 канала, файнтюн от v3 GLOBAL, freeze6, без roi_only_loss) | `best_model.pth` |
| `neural_postproc_v3` | Нейросетевая постобработка с per-station bias на 689 станций | `best_model.pth` |

Каждый эксперимент содержит `config.json`, `README.md`, `training_log.txt`, `results.json`.

### Обучение глобальной модели v3

```bash
source .venv/bin/activate
python -m src.main experiments/wb2_512x256_33f_ar_v3
```

Конфиг в `experiments/wb2_512x256_33f_ar_v3/config.json`: 33 канала, curriculum AR=3→4→5, 50 эпох, channel weights, time-forcing.

### Файнтюн на Россию

```bash
python -m src.main experiments/multires_russia_33f_v3_noroi \
    --pretrained experiments/wb2_512x256_33f_ar_v3/best_model.pth
```

Стратегия `freeze6`: первые 6 эпох процессор заморожен, обучаются только кодировщик и декодер; затем процессор размораживается с уменьшенной скоростью обучения (×0.1) — это удерживает выученную глобальную физику.

### Обучение постобработки

```bash
# Сборка корпуса (ERA5 ↔ ISD-Lite)
python scripts/postproc/build_corpus.py \
    --experiment-dir experiments/multires_russia_33f_v3_noroi \
    --stations-json data/russia_mos_stations.json \
    --top-stations 689 \
    --years 2018 2020

# Обучение
python -m src.postprocessing.neural.train experiments/neural_postproc_v3
```

---

## Оперативный режим

Цепочка работает на сервере ИВМиМГ СО РАН (`city.sscc.ru`) под cron каждые 6 часов:

```cron
0 1,7,13,19 * * * /home/tabakov_2026/graphcast-lite/cron_forecast_russia_v3.sh
```

Скрипт `cron_forecast_russia_v3.sh` (хранится отдельно на сервере, шаблон ниже) делает четыре шага: скачивает свежие анализы NOAA GDAS, прогоняет графовую модель на 5 суток, применяет нейросетевую постобработку по 689 станциям, формирует JSON и тайлы для сайта прогноза.

### Что нужно для развёртывания на новой VM

1. Установить зависимости (см. «Установка»).
2. Убедиться, что в репозитории есть веса:
   - `experiments/multires_russia_33f_v3_noroi/best_model.pth` — графовая модель;
   - `experiments/neural_postproc_v3/best_model.pth` — постобработка;
   - `live_runtime_bundle_russia_33f/` — статика модели (`coords.npz`, `scalers.npz`, `static_fields.npz`, `variables.json`, `dataset_info.json`).
3. Создать скрипт `cron_forecast_russia_v3.sh` (см. шаблон ниже), сделать его исполняемым (`chmod +x`).
4. Добавить в `crontab -e`:

   ```cron
   0 1,7,13,19 * * * /path/to/graphcast-lite/cron_forecast_russia_v3.sh >> /var/log/graphcast-russia.log 2>&1
   ```

5. Поднять веб-приложение (FastAPI + uvicorn) на каком-то локальном порту, прокинуть nginx-ом наружу.

### Шаблон `cron_forecast_russia_v3.sh`

```bash
#!/bin/bash
set -euo pipefail
BASEDIR=/path/to/graphcast-lite
VENV=$BASEDIR/.venv/bin/python
OUT=$BASEDIR/results/live_russia_33f_5d
CACHE=/tmp/gdas_cache

cd "$BASEDIR"

# 1. Прогноз графовой моделью
$VENV scripts/live_gdas_forecast.py \
    --experiment-dir experiments/multires_russia_33f_v3_noroi \
    --runtime-bundle live_runtime_bundle_russia_33f \
    --ar-steps 20 \
    --out-dir "$OUT" \
    --cache-dir "$CACHE"

# 2. Нейросетевая постобработка
$VENV scripts/postproc_apply_v3_russia.py \
    --forecast "$OUT/forecast.pt" \
    --stations data/russia_mos_stations.json \
    --postproc-dir experiments/neural_postproc_v3 \
    --out "$OUT/station_postproc_v3.json"

# 3. JSON для сайта
$VENV scripts/build_russia_forecast_json.py \
    --forecast "$OUT/forecast.pt" \
    --postproc-json "$OUT/station_postproc_v3.json" \
    --out website/static/russia_forecast.json

# 4. Чистка кеша GDAS и старого forecast.pt
rm -rf "$CACHE"
rm -f "$OUT/forecast.pt"  # тяжёлый (~1.8 ГБ), JSON уже собран
```

### Запуск веб-приложения на сервере

```bash
# В сессии (для отладки)
uvicorn website.app:app --host 127.0.0.1 --port 18080

# Постоянно — через systemd / supervisor / @reboot crontab.
# Пример @reboot:
@reboot cd /path/to/graphcast-lite && .venv/bin/uvicorn website.app:app --host 127.0.0.1 --port 18080 >> /var/log/graphcast-web.log 2>&1
```

Снаружи прокидывается через nginx (reverse proxy `127.0.0.1:18080` → публичный домен).

---

## Данные

### ERA5 (тренировочный)

Используем подвыборку реанализа ERA5 на сетке 512×256 (~0.7°), 33 переменные:

- Приземные: `t2m`, `10u`, `10v`, `msl`, `tp`, `sp`, `tcwv`, `z_surf`, `lsm`.
- Уровни давления: `t/u/v/z/q` на 250, 500, 850, 1000 гПа.
- Период: 2010–2021 гг., шаг 6 ч.
- Хранение: `float16` memory-mapped (`data.npy`, ~81 ГБ).

Сборка:

```bash
python scripts/build_dataset_512x256_33f.py
# С резюмом:
python scripts/build_dataset_512x256_33f.py --resume
```

### Региональный датасет (Россия, 0.25°)

Используем CDS API для скачивания ERA5 на сетке 0.25° над bbox России:

```bash
python scripts/build_region_russia_19f.py
python scripts/build_region_cds_extra.py  # дополнительные каналы (250/1000 гПа)
python scripts/build_multires_russia_33f.py  # сборка мультимасштабной структуры
```

### ISD-Lite (станционные наблюдения)

Для обучения постобработки используются наблюдения 689 станций ВМО на территории России из архива NOAA ISD-Lite. Реестр станций: `data/russia_mos_stations.json`.

```bash
python scripts/download_russia_isd_lite.py
```

---

## Структура репозитория

```text
.
├── src/
│   ├── mesh/                       # Построение икосаэдрального меша и связей
│   │   ├── create_mesh.py
│   │   └── grid_mesh_connectivity.py
│   ├── data/                       # Загрузчики данных
│   │   ├── data_configs.py
│   │   ├── dataloaders.py
│   │   └── dataloader_chunked.py   # memmap + sliding window для 512×256+
│   ├── assimilation/               # Усвоение данных на этапе прогноза
│   │   ├── nudging.py
│   │   └── optimal_interpolation.py
│   ├── postprocessing/             # Нейросетевая постобработка
│   │   ├── corpus.py               # сборка корпуса ERA5 ↔ ISD
│   │   ├── dataset.py
│   │   └── neural/
│   │       ├── models.py           # StationLeadBiasResidualMLP (v3)
│   │       └── train.py
│   ├── models.py                   # Графовая 2.5D-модель
│   ├── train.py
│   └── main.py                     # Точка входа для обучения
│
├── scripts/
│   ├── build_dataset_512x256_33f.py
│   ├── build_region_russia_19f.py
│   ├── build_region_cds_extra.py
│   ├── build_multires_russia_33f.py
│   ├── build_russia_mos.py
│   ├── build_russia_forecast_json.py
│   ├── download_russia_isd_lite.py
│   ├── live_gdas_forecast.py       # Один шаг оперативного прогона
│   ├── postproc_apply_v3_russia.py # Применение постобработки v3
│   └── predict.py
│
├── experiments/
│   ├── wb2_512x256_19f_ar/         # Глобальная v1 (GCN)
│   ├── wb2_512x256_19f_ar_v2/      # Глобальная v2 (InteractionNet)
│   ├── wb2_512x256_33f_ar_v3/      # Глобальная v3 (33 канала)
│   ├── multires_russia_33f_v3_noroi/  # Россия (33 канала, файнтюн от v3)
│   ├── neural_postproc_v2/         # Постобработка v2 (50 станций)
│   └── neural_postproc_v3/         # Постобработка v3 (689 станций + bias)
│
├── live_runtime_bundle/                # Артефакты live-режима (Красноярск)
├── live_runtime_bundle_russia_33f/     # Артефакты live-режима (Россия)
│
├── website/                        # Сайт прогноза (FastAPI + статика)
│   ├── app.py
│   ├── forecast_parser.py
│   └── static/
│       ├── russia.html
│       ├── russia.js
│       ├── russia_forecast.json
│       ├── russia_stations.json
│       └── overlays_russia/        # Тайлы прогноза для карты
│
├── data/
│   ├── russia_mos_stations.json    # Реестр 689 станций
│   └── datasets/                   # Локальные датасеты (gitignored)
│
├── docs/                           # Документация и текст диплома
│   ├── diploma/diploma.tex
│   ├── postprocessing_rfc.md       # RFC-001 нейросетевая постобработка
│   └── era5_urban_problems_and_postprocessing.md
│
├── requirements.txt
└── README_RU.MD                    # Этот файл
```

### Точки входа в код

| Что хотите делать | Где смотреть |
|---|---|
| Понять архитектуру графовой модели | [`src/models.py`](src/models.py) — класс `GraphCastLite`, encoder/processor/decoder |
| Понять цикл обучения | [`src/train.py`](src/train.py) — `train()`, curriculum AR, freeze_processor |
| Понять, как строятся графы | [`src/create_graphs.py`](src/create_graphs.py), [`src/mesh/`](src/mesh/) |
| Понять, как читаются данные | [`src/data/dataloader_chunked.py`](src/data/dataloader_chunked.py) — memmap + sliding window |
| Понять усвоение наблюдений | [`src/assimilation/`](src/assimilation/) — `nudging.py`, `optimal_interpolation.py` |
| Понять постобработку | [`src/postprocessing/neural/models.py`](src/postprocessing/neural/models.py) — `StationLeadBiasResidualMLP` |
| Понять, как запускается живой прогноз | [`scripts/live_gdas_forecast.py`](scripts/live_gdas_forecast.py) |
| Понять, как работает сайт | [`website/app.py`](website/app.py), статика — [`website/static/`](website/static/) |

---

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Основные зависимости: `torch`, `torch_geometric`, `numpy`, `scipy`, `xarray`, `cfgrib`, `matplotlib`, `cartopy`, `trimesh`, `wandb` (опционально).

Для скачивания свежих анализов GDAS нужен интернет без блокировок NOAA NOMADS. Для скачивания ERA5 (CDS API) нужен бесплатный аккаунт `cdsapi` и файл `~/.cdsapirc` с ключом.

---

## Быстрый старт (для нового разработчика)

После `git clone` и `pip install -r requirements.txt` — четыре сценария, чтобы быстро убедиться, что всё работает:

### 1. Сгенерировать живой прогноз России на 5 суток

Использует обученные веса из репозитория (`experiments/multires_russia_33f_v3_noroi/best_model.pth`). Скачивает свежие анализы NOAA GDAS, прогоняет графовую модель, применяет нейросетевую постобработку, формирует JSON для сайта.

```bash
# (а) Шаг 1: прогноз графовой моделью
python scripts/live_gdas_forecast.py \
    --experiment-dir experiments/multires_russia_33f_v3_noroi \
    --runtime-bundle live_runtime_bundle_russia_33f \
    --ar-steps 20 \
    --out-dir results/live_russia_33f_5d \
    --cache-dir /tmp/gdas_cache

# (б) Шаг 2: нейросетевая постобработка по 689 станциям
python scripts/postproc_apply_v3_russia.py \
    --forecast results/live_russia_33f_5d/forecast.pt \
    --stations data/russia_mos_stations.json \
    --postproc-dir experiments/neural_postproc_v3 \
    --out results/live_russia_33f_5d/station_postproc_v3.json

# (в) Шаг 3: сборка финального JSON для сайта
python scripts/build_russia_forecast_json.py \
    --forecast results/live_russia_33f_5d/forecast.pt \
    --postproc-json results/live_russia_33f_5d/station_postproc_v3.json \
    --out website/static/russia_forecast.json
```

После этого `website/static/russia_forecast.json` содержит свежий прогноз.

### 2. Поднять сайт локально

```bash
uvicorn website.app:app --host 0.0.0.0 --port 8000 --reload
```

Откроется в браузере: <http://localhost:8000/static/russia.html> — карта России с прогнозом по 689 станциям, переключение между лидами, оверлеи температуры/ветра/осадков.

### 3. Запустить инференс старой (Красноярской) цепочки

Если по какой-то причине нужен исходный пайплайн с MOS-постобработкой по 19 станциям Красноярска:

```bash
python scripts/live_gdas_forecast.py \
    --experiment-dir experiments/multires_nores_freeze6 \
    --runtime-bundle live_runtime_bundle \
    --learned-mos live_runtime_bundle/learned_mos_t2m_wind_19st.joblib \
    --wind-scale live_runtime_bundle/wind_monthly_scale.json \
    --spatial-idw --selective \
    --lapse-target-elevation 287 --ar-steps 12 \
    --out-dir results/live_krsk_3d --cache-dir /tmp/gdas_cache

python website/forecast_parser.py \
    --input results/live_krsk_3d/forecast.pt \
    --output website/static/forecast.json
```

### 4. Поставить на обучение свою модель

См. раздел «[Эксперименты](#эксперименты)» ниже. Простой случай — повторить файнтюн на Россию от имеющегося чекпойнта v3 GLOBAL:

```bash
python -m src.main experiments/multires_russia_33f_v3_noroi \
    --pretrained experiments/wb2_512x256_33f_ar_v3/best_model.pth
```

Обучение идёт до 50 эпох с curriculum AR=3→4→5; чекпойнт сохраняется в `experiments/multires_russia_33f_v3_noroi/best_model.pth` каждый раз, когда улучшается валидационный лосс. Лог пишется в `training_log.txt`, его можно смотреть в реальном времени через `tail -f`.

---

## Связанные документы

- Текст ВКР: [`docs/diploma/diploma.tex`](docs/diploma/diploma.tex), [`docs/diploma/diploma.pdf`](docs/diploma/diploma.pdf).
- RFC по нейросетевой постобработке: [`docs/postprocessing_rfc.md`](docs/postprocessing_rfc.md).
- Тезисы доклада на МНСК 2026 и Марчуковских чтениях 2026.

---

## Возможные направления развития

Гипотезы, которые в рамках ВКР проверить не успели — могут пригодиться следующему студенту:

- **Накопительная статистика по оперативному режиму.** Сейчас итоговая верификация сделана на одном-двух прогонах. Имеет смысл собрать журнал предсказаний за несколько недель / месяцев и сравнивать с независимым ground truth (например, по архиву ISD-Lite после факта). Базовая идея — построить простую таблицу «дата → MAE/RMSE по T2m, U10, V10 на каждой станции» и смотреть на тренд.
- **Перенос гиперпараметров усвоения между регионами и сезонами.** Гиперпараметры $L = 100$ км и $\sigma_o = 0.5$ подбирались на Сибири, ERA5 2018–2020. Проверить, что они работают на других регионах (Европейская часть РФ, Дальний Восток) и в других сезонах.
- **Усвоение по другим переменным.** Сейчас в OI/Nudging «усваиваются» все 19/33 канала. Возможно, для разрежённой реальной сети станций имеет смысл усваивать только T2m и MSL, а остальное достраивать через корреляции — это упростит конфигурацию для оперативного режима.
- **Сравнение с независимыми системами.** В качестве референсной серии в работе использовались ретроспективные данные Open-Meteo (агрегация ERA5 + станций). Полноценная независимая верификация — отдельная задача: например, прямое сравнение с прогнозами Гидрометцентра или с сетями автоматических станций.
- **Расширение нейросетевой постобработки.** Сейчас постобработка предсказывает поправку только в точках станций. Можно попробовать применять её как пространственное поле (через интерполяцию поправок) или ввести обучаемый IDW.
- **Дополнительные горизонты прогноза.** Модель устойчиво работает до 3–4 суток. Имеет смысл систематически измерить деградацию навыка после 5 суток и понять, что ограничивает горизонт — накопление ошибки в графовой модели или ошибка постобработки.

---

## Лицензия

Исходный форк (механизмы внимания / SparseGAT / product graphs из академического прототипа GraphCast-lite) — Apache 2.0. Все доработки в рамках ВКР — на тех же условиях.
