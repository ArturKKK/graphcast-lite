# Проблемы ERA5 и постобработка: контекст для слайдов

## 1. Проблема ERA5 в городах (urban representativeness gap)

### Суть проблемы

ERA5 — глобальный реанализ ECMWF с разрешением 0.25° (~31 км). Одна ячейка усредняет застройку, реку, промышленную зону, пригород и рельеф в одно значение.

На тесте по ERA5 модель выглядит хорошо: она учится воспроизводить тот же сглаженный реанализ, на котором её оценивают (in-distribution evaluation). При переходе в live-прогноз для конкретного города качество падает, потому что:

1. **Сглаживание grid-box mean.** ERA5 не разрешает неоднородности внутри ячейки. Для городов это критично: температура и ветер меняются на расстояниях в сотни метров. Coarse-resolution продукты систематически недопредставляют локальные экстремумы и urban heat island (Nogueira et al., 2022).

2. **Ошибка представительности (representativeness error).** Станция измеряет точечные условия, а ERA5 описывает среднее состояние ячейки ~31×31 км. Это фундаментальная проблема сравнения point observations с gridded fields (Janjić et al., 2018).

3. **Неразрешённые urban processes.** Теплоёмкость материалов, антропогенное тепло, уличные каньоны, затенение, аэродинамическая шероховатость, ночное выхолаживание, суточный ход UHI — всё это в ERA5 либо сильно ослаблено, либо вообще не разрешается (Huang et al., 2025).

4. **Для холодных городов проблема сильнее.** Зимние эффекты, отопление, снежный покров, инверсии и особенности турбулентного обмена делают городской климат ещё менее похожим на «среднюю ячейку» реанализа (Brozovsky et al., 2021).

### Красноярск: особо тяжёлый случай

- Локальный рельеф + Енисей (не замерзает зимой → тепловой контраст)
- Частые зимние инверсии, слабый ветер → очень локальная структура температуры
- Связь загрязнения и метеоусловий с температурными инверсиями (Belolipetskii et al., 2023)
- UHI Красноярска подтвержден данными Landsat (Matuzko & Yakubailik, 2018)
- Влияние UHI, орографии и Енисея на локальную циркуляцию (Hrebtov et al., 2019)

### Измеренный bias ERA5 vs станция WMO 29570 (Красноярск)

```
         0h    3h    6h    9h   12h   15h   18h   21h  (UTC)
Январь  +7.1  +7.2  +6.2  +6.3  +7.0  +6.7  +6.2  +6.8   ← станция теплее ERA5 на 5-7°C (UHI + инверсии)
Апрель  +0.5  -3.8  -4.1  -2.5  -0.4  +2.6  +2.5  +1.9   ← весна: днём ERA5 теплее
Июль    -2.1  -6.2  -5.2  -3.9  -2.3  +1.4  +2.4  +1.0   ← лето: днём ERA5 сильно теплее
Октябрь +3.9  +2.0  -0.0  +1.8  +4.9  +5.0  +4.6  +4.1   ← осень: станция теплее
```

### Формулировка для работы

> Несмотря на высокое качество на тестовой выборке ERA5, при переходе к оперативному прогнозированию для конкретного города качество модели может заметно ухудшаться. ERA5 представляет собой сглаженное grid-box описание атмосферы с разрешением ~0.25°, тогда как городской микроклимат определяется процессами существенно более мелкого масштаба: urban heat island, неоднородностью застройки, антропогенным тепловыделением и локальной циркуляцией. При сравнении gridded-полей с данными городской станции возникает representativeness error. Для Красноярска ситуация дополнительно осложняется частыми зимними инверсиями, влиянием рельефа и реки.

---

## 2. Постобработка (Postprocessing Pipeline)

Для компенсации систематических ошибок модели и ERA5 реализован 3-ступенчатый пайплайн постобработки.

### Этап 1: Lapse-rate коррекция высоты

**Проблема:** высота узла грида ≠ высота станции. На грубой сетке ячейка усредняет рельеф, и модельная температура соответствует средней высоте ячейки, а не высоте города.

**Решение:** коррекция по стандартному температурному градиенту тропосферы:

$$T_{\text{corrected}} = T_{\text{model}} + \Gamma \cdot (z_{\text{station}} - z_{\text{grid}})$$

- $\Gamma = 6.5$ °C/км — стандартный lapse rate
- $z_{\text{grid}}$ — высота узла по z_surf из ERA5 (геопотенциал / g)
- $z_{\text{station}}$ — высота целевой станции

Для Красноярска: средняя высота 19 станций ≈ 230 м, что часто отличается от высоты ячейки на десятки метров → коррекция ±0.3–1.0°C.

### Этап 2: Learned MOS (Model Output Statistics)

**Проблема:** систематический bias ERA5 vs станция зависит от сезона, времени суток, синоптической ситуации.

**Решение:** обучаем HistGradientBoostingRegressor предсказывать bias = T_station − T_ERA5.

**Признаки (20):**
- Метео: t2m, dewpoint, wind_speed, wind_dir (sin/cos), sp, cloudcover, shortwave, tp
- Временные: hour (sin/cos), day-of-year (sin/cos), solar elevation
- Лаговые: t2m_lag6h, delta_t2m_6h
- Пространственные: station_lat, station_lon, station_elev

**Данные:** ERA5 (Open-Meteo Archive API) + 19 станций NOAA ISD-Lite в радиусе ~270 км от Красноярска, 2016–2024.

**Результат на тесте (2024):** MAE = **1.32°C**.

При инференсе: для каждой станции предсказывается bias → прибавляется к ближайшему узлу сетки.

### Этап 3: Spatial IDW (пространственная интерполяция)

**Проблема:** MOS корректирует только 19 узлов из 2501 регионального грида (0.7%).

**Решение:** Inverse Distance Weighting — пространственная интерполяция bias от станционных точек на все узлы:

$$\text{bias}(v) = \frac{\sum_{k} w_k \cdot \text{bias}(s_k)}{\sum_{k} w_k}, \quad w_k = \frac{1}{d(v, s_k)^p}$$

- $s_k$ — станции в пределах радиуса
- $d(v, s_k)$ — расстояние между узлом и станцией (по Хаверсину)
- $p = 2$ — степень обратного расстояния
- Макс. радиус: 300 км

Итог: каждый узел региональной сетки получает скорректированную температуру, а не только ближайшие к станциям.

---

## 3. Порядок применения в live pipeline

```
GDAS input → [input bias correction] → нормализация → GNN inference → денормализация
  → Lapse-rate коррекция
  → Learned MOS (19 станций, HistGBR)
  → Spatial IDW (интерполяция bias на всю сетку)
  → финальный прогноз
```

Флаги запуска:
```bash
python scripts/live_gdas_forecast.py \
  --runtime-bundle live_runtime_bundle \
  --learned-mos live_runtime_bundle/learned_mos_t2m_19stations.joblib \
  --spatial-idw \
  --lapse-target-elevation 230
```

---

## Литература

1. Nogueira M. et al. (2022). Assessment of the Paris urban heat island in ERA5 and offline SURFEX-TEB simulations. *Geosci. Model Dev.*, 15, 5949–5981.
2. Janjić T. et al. (2018). On the representation error in data assimilation. *Q.J.R. Meteorol. Soc.*, 144(713), 1257–1278.
3. Park J. et al. (2026). A super-resolution framework for downscaling ML weather prediction toward 1-km air temperature. *npj Clim. Atmos. Sci.*
4. Brozovsky J. et al. (2021). A systematic review of urban climate research in cold and polar climate regions. *Renew. Sustain. Energy Rev.*, 138, 110551.
5. Huang J. et al. (2025). Urban heat forecasting in small cities. *Geosci. Model Dev.*, 18, 9237–9261.
6. Belolipetskii V. et al. (2023). Parametrization of temperature inversion over Krasnoyarsk. *E3S Web Conf.*
7. Matuzko A., Yakubailik O. (2018). Monitoring of Land Surface Temperature in Krasnoyarsk with Landsat data.
8. Hrebtov M. et al. (2019). River-Induced Anomalies in Seasonal Variation of Traffic-Related Air Pollution. *Atmosphere*, 10(7), 407.
