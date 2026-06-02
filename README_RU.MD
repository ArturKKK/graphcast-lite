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

Развёрнут на сервере ИВМиМГ СО РАН (city.sscc.ru) под cron:

```cron
0 1,7,13,19 * * * /home/tabakov_2026/graphcast-lite/cron_forecast_russia_v3.sh
```

Цепочка работы:

1. Скачивание свежих анализов NOAA GDAS (6-часовое окно).
2. Авторегрессионный прогноз графовой моделью на 5 суток.
3. Нейросетевая постобработка по 689 станциям.
4. Генерация JSON-прогнозов и тайлов для сайта прогноза.

Все артефакты, нужные для оперативного режима, хранятся в `live_runtime_bundle_russia_33f/` (статика модели: `coords.npz`, `scalers.npz`, `static_fields.npz`, `variables.json`).

Текущий прогноз публикуется на сайте: `website/static/russia.html` + `russia_forecast.json` + тайлы `overlays_russia/`.

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

---

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Основные зависимости: `torch`, `torch_geometric`, `numpy`, `scipy`, `xarray`, `cfgrib`, `matplotlib`, `cartopy`, `trimesh`, `wandb` (опционально).

---

## Связанные документы

- Текст ВКР: [`docs/diploma/diploma.tex`](docs/diploma/diploma.tex), [`docs/diploma/diploma.pdf`](docs/diploma/diploma.pdf).
- RFC по нейросетевой постобработке: [`docs/postprocessing_rfc.md`](docs/postprocessing_rfc.md).
- Тезисы доклада на МНСК 2026 и Марчуковских чтениях 2026.

---

## Лицензия

Исходный форк (механизмы внимания / SparseGAT / product graphs из академического прототипа GraphCast-lite) — Apache 2.0. Все доработки в рамках ВКР — на тех же условиях.
