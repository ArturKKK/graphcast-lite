"""Описатели рельефа вокруг станции по матрице высот.

Зачем. Выигрыш поправки живёт в сложном рельефе: связь с высотой станции +0,72,
наибольшие поправки у станций на 420-1850 м с холодным смещением до -4,7 °C. При
этом о рельефе модель знает только высоту станции и орографию ПРОГНОСТИЧЕСКОЙ
СЕТКИ, сглаженную до 0,25° — около 28 км. На таком масштабе долина шириной в
километр попросту не существует, а именно в ней и застаивается холодный воздух.

Признаки окрестности узлов сетки дали 0,71 % ровно потому, что добавили сведения
о рельефе. Настоящая матрица высот подробнее на два порядка (90 м против 28 км),
и те же по смыслу величины на ней осмысленны физически, а не статистически.

Что считается и почему именно это:

* положение относительно окрестности (TPI) на нескольких радиусах — главный
  описатель застоя холодного воздуха: отрицательное значение означает дно
  долины, где ночью выхолаживание сильнее всего, положительное — вершину или
  склон, продуваемый и остающийся тёплым при инверсии;
* изрезанность (стандартное отклонение высот) — насколько площадка вообще
  представительна для своей ячейки;
* глубина долины и превышение над окрестностью — крайние значения того же;
* уклон и экспозиция склона — приход солнечного тепла и сток холодного воздуха;
* закрытость горизонта — доля неба, определяющая ночное выхолаживание.

Все величины СТАТИЧНЫ для станции: их считают один раз и приклеивают к корпусу
по номеру станции, а не пересобирают развёрткой.

Модуль на одном numpy и покрыт тестами на искусственном рельефе с известным
ответом: конусе, долине, наклонной плоскости.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6371000.0


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Сколько метров в градусе широты и долготы на данной широте."""
    lat = np.radians(lat_deg)
    return float(EARTH_R * np.pi / 180.0), float(EARTH_R * np.pi / 180.0 * np.cos(lat))


def _disc(shape: tuple[int, int], center: tuple[int, int],
          radius_px: tuple[float, float]) -> np.ndarray:
    """Маска эллипса: по широте и долготе разный шаг в метрах, круг на местности
    в пикселях становится эллипсом."""
    ry, rx = radius_px
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - center[0]) / max(ry, 1e-9)) ** 2 + \
           ((xx - center[1]) / max(rx, 1e-9)) ** 2 <= 1.0


def topographic_position(dem: np.ndarray, center: tuple[int, int],
                         radius_px: tuple[float, float]) -> float:
    """Превышение точки над средней высотой круга: TPI.

    Отрицательное — точка ниже окрестности (дно долины, котловина), нулевое —
    ровное место или середина склона, положительное — вершина или гребень.
    """
    m = _disc(dem.shape, center, radius_px)
    vals = dem[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(dem[center] - vals.mean())


def roughness(dem: np.ndarray, center: tuple[int, int],
              radius_px: tuple[float, float]) -> float:
    """Изрезанность: стандартное отклонение высот в круге, метры."""
    vals = dem[_disc(dem.shape, center, radius_px)]
    vals = vals[np.isfinite(vals)]
    return float(vals.std()) if vals.size else float("nan")


def relief_extremes(dem: np.ndarray, center: tuple[int, int],
                    radius_px: tuple[float, float]) -> tuple[float, float]:
    """Глубина долины и превышение: точка минус минимум и максимум минус точка."""
    vals = dem[_disc(dem.shape, center, radius_px)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(dem[center] - vals.min()), float(vals.max() - dem[center])


def slope_aspect(dem: np.ndarray, center: tuple[int, int],
                 dy_m: float, dx_m: float) -> tuple[float, float]:
    """Уклон в градусах и экспозиция склона в градусах от севера по часовой.

    Экспозиция — направление, КУДА склон смотрит: 0 — на север, 90 — на восток.
    На ровном месте не определена и возвращается NaN, а не произвольное число:
    иначе признак нёс бы шум там, где склона нет.
    """
    i, j = center
    if not (0 < i < dem.shape[0] - 1 and 0 < j < dem.shape[1] - 1):
        return float("nan"), float("nan")
    # ось строк идёт с юга на север, поэтому производная по северу — вперёд
    dzdy = (dem[i + 1, j] - dem[i - 1, j]) / (2.0 * dy_m)
    dzdx = (dem[i, j + 1] - dem[i, j - 1]) / (2.0 * dx_m)
    slope = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))
    if np.hypot(dzdx, dzdy) < 1e-12:
        return slope, float("nan")
    # смотрит склон вниз по уклону, то есть против градиента
    aspect = float(np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0)
    return slope, aspect


def horizon_closure(dem: np.ndarray, center: tuple[int, int],
                    dy_m: float, dx_m: float, radius_m: float,
                    n_dirs: int = 8) -> float:
    """Закрытость горизонта: средний угол подъёма горизонта, градусы.

    Ноль — открытое место, большие значения — узкая долина или подножие склона.
    Определяет ночное выхолаживание: чем закрытее горизонт, тем меньше уходит
    длинноволнового излучения и тем... наоборот, тем сильнее застой холодного
    воздуха. Величина сама по себе информативна, знак связи выучит модель.
    """
    i, j = center
    h0 = dem[i, j]
    if not np.isfinite(h0):
        return float("nan")
    angles = []
    for k in range(n_dirs):
        a = 2 * np.pi * k / n_dirs
        best = 0.0
        step = 1
        while True:
            di = int(round(np.cos(a) * step))
            dj = int(round(np.sin(a) * step))
            ii, jj = i + di, j + dj
            dist = np.hypot(di * dy_m, dj * dx_m)
            if dist > radius_m:
                break
            if not (0 <= ii < dem.shape[0] and 0 <= jj < dem.shape[1]):
                break
            h = dem[ii, jj]
            if np.isfinite(h) and dist > 0:
                best = max(best, float(np.degrees(np.arctan((h - h0) / dist))))
            step += 1
            if step > 10000:
                break
        angles.append(best)
    return float(np.mean(angles))


def station_terrain(dem: np.ndarray, center: tuple[int, int], lat_deg: float,
                    cell_deg: float, radii_m=(1000.0, 5000.0, 20000.0)) -> dict:
    """Все описатели для одной станции. Ключи начинаются с ``terr_``.

    ``cell_deg`` — шаг матрицы высот в градусах: 1/3600 для листов в 1 угловую
    секунду (их и качаем), 1/1200 для трёхсекундных. Радиусы задаются в метрах и
    переводятся в точки через этот шаг, поэтому от разрешения описатели не
    зависят — проверено сравнением на 1\u2033 и 3\u2033. Шаг по долготе в метрах
    короче, чем по широте: на 55° градус долготы вдвое короче градуса широты.
    """
    m_lat, m_lon = meters_per_degree(lat_deg)
    dy_m, dx_m = cell_deg * m_lat, cell_deg * m_lon
    out: dict[str, float] = {}
    for r in radii_m:
        px = (r / dy_m, r / dx_m)
        key = f"{int(r / 1000)}km"
        out[f"terr_tpi_{key}"] = topographic_position(dem, center, px)
        out[f"terr_rough_{key}"] = roughness(dem, center, px)
        low, high = relief_extremes(dem, center, px)
        out[f"terr_above_min_{key}"] = low
        out[f"terr_below_max_{key}"] = high
    slope, aspect = slope_aspect(dem, center, dy_m, dx_m)
    out["terr_slope"] = slope
    # Экспозиция — угол на круге, и подавать её числом нельзя: 359° и 1° рядом,
    # а по величине далеки. Раскладываем на синус и косинус.
    out["terr_aspect_sin"] = float(np.sin(np.radians(aspect))) if np.isfinite(aspect) else 0.0
    out["terr_aspect_cos"] = float(np.cos(np.radians(aspect))) if np.isfinite(aspect) else 0.0
    out["terr_horizon"] = horizon_closure(dem, center, dy_m, dx_m, max(radii_m))
    out["terr_dem_elev"] = float(dem[center])
    return out


# ─── Чтение матрицы высот в формате .hgt ─────────────────────────────────────
#
# Почему именно он. Формат до предела прост: сырые двухбайтовые целые с обратным
# порядком байт, квадратная сетка на градусный лист, никаких заголовков. Читается
# одним numpy, тогда как GeoTIFF потребовал бы rasterio или GDAL — тяжёлых
# зависимостей, которых на виртуалке нет и ставить их ради статичной таблицы
# признаков незачем.
#
# Листы называются по юго-западному углу: N55E093.hgt — от 55° до 56° с.ш. и от
# 93° до 94° в.д. Первая строка файла — САМАЯ СЕВЕРНАЯ, поэтому при чтении массив
# переворачивается: дальше по всему модулю ось строк идёт с юга на север.


def hgt_tile_name(lat: float, lon: float) -> str:
    """Имя листа, содержащего точку. Лист называется по юго-западному углу."""
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def tiles_for_points(lats, lons, margin_deg: float = 0.35) -> list[str]:
    """Какие листы нужны, чтобы вокруг каждой точки был запас в margin_deg.

    Запас обязателен: описатели считаются в круге радиусом до 20 км, а это около
    0,18° по широте. Без запаса станция у края листа получила бы обрезанную
    окрестность, и признак молча посчитался бы по половине круга.
    """
    need = set()
    for la, lo in zip(np.atleast_1d(lats), np.atleast_1d(lons)):
        for dla in (-margin_deg, 0.0, margin_deg):
            for dlo in (-margin_deg, 0.0, margin_deg):
                need.add(hgt_tile_name(la + dla, lo + dlo))
    return sorted(need)


def read_hgt(path) -> np.ndarray:
    """Прочитать лист .hgt (можно .gz). Возвращает (N, N) в метрах, юг внизу.

    Пустые значения (-32768) заменяются на NaN: считать по ним среднюю высоту
    нельзя, а молча принять их за -32 км — верный способ испортить все описатели.
    """
    import gzip
    import os

    path = str(path)
    raw = (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")).read()
    n = int(round(np.sqrt(len(raw) / 2)))
    if n * n * 2 != len(raw):
        raise ValueError(
            f"{os.path.basename(path)}: {len(raw)} байт — не квадратная сетка "
            f"из двухбайтовых целых. Формат .hgt не распознан.")
    a = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
    a[a == -32768] = np.nan
    return a[::-1]                       # первая строка была самой северной


def load_mosaic(tile_dir, lat: float, lon: float, margin_deg: float = 0.35):
    """Склеить листы вокруг точки в одну матрицу.

    Возвращает (матрица, номер строки и столбца точки, шаг в градусах). Склейка
    нужна потому, что станция часто стоит близко к краю листа, а круг в 20 км
    заходит на соседний.
    """
    from pathlib import Path as _P

    tile_dir = _P(tile_dir)
    la0, lo0 = int(np.floor(lat - margin_deg)), int(np.floor(lon - margin_deg))
    la1, lo1 = int(np.floor(lat + margin_deg)), int(np.floor(lon + margin_deg))

    rows = []
    cell = None
    size = None
    for la in range(la0, la1 + 1):
        cols = []
        for lo in range(lo0, lo1 + 1):
            name = hgt_tile_name(la, lo)
            f = next((p for p in (tile_dir / f"{name}.hgt", tile_dir / f"{name}.hgt.gz")
                      if p.exists()), None)
            if f is None:
                raise FileNotFoundError(
                    f"нет листа {name} в {tile_dir} — он нужен для точки "
                    f"{lat:.3f}, {lon:.3f}. Список нужных листов даёт "
                    f"scripts/postproc/list_dem_tiles.py")
            t = read_hgt(f)
            # Листы разного разрешения склеивать нельзя: шаг сетки берётся от
            # последнего прочитанного, и часть мозаики оказалась бы растянута.
            # numpy упал бы и сам, но невнятно — на несовпадении размеров.
            if size is not None and t.shape[0] != size:
                raise ValueError(
                    f"лист {name} имеет размер {t.shape[0]}, а предыдущие "
                    f"{size}. Листы разного разрешения (1\u2033 и 3\u2033) "
                    f"в одном каталоге смешивать нельзя.")
            size = t.shape[0]
            # У листов перекрывается последняя строка и столбец с соседним:
            # отбрасываем их, иначе стык задваивается и сетка перестаёт быть
            # равномерной, а по ней считаются расстояния.
            cols.append(t[:-1, :-1])
            cell = 1.0 / (t.shape[0] - 1)
        rows.append(np.hstack(cols))
    mosaic = np.vstack(rows)

    i = int(round((lat - la0) / cell))
    j = int(round((lon - lo0) / cell))
    i = min(max(i, 0), mosaic.shape[0] - 1)
    j = min(max(j, 0), mosaic.shape[1] - 1)
    return mosaic, (i, j), cell
