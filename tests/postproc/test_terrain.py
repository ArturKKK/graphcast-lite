"""Описатели рельефа вокруг станции.

Проверяются на искусственном рельефе с известным ответом: наклонной плоскости,
конусе, долине, ровном поле. Так видно не «код не падает», а что величины
означают именно то, что заявлено, — иначе модель выучит по ним что угодно, а
объяснить результат будет нечем.

Матрица высот для этих тестов не нужна: они на чистой геометрии.
"""
from pathlib import Path

import numpy as np
import pytest

from src.postprocessing.terrain import (
    horizon_closure,
    meters_per_degree,
    relief_extremes,
    roughness,
    slope_aspect,
    station_terrain,
    topographic_position,
)

# Шаг сетки: 3 угловые секунды, как у Copernicus GLO-90.
CELL = 1.0 / 1200.0
LAT = 55.0


def flat(n=81, h=100.0):
    return np.full((n, n), h, dtype=np.float64)


def plane(n=81, slope_north=0.0, slope_east=0.0, cell_m=90.0):
    """Наклонная плоскость: высота растёт на slope_* метров на метр."""
    i = (np.arange(n) - n // 2) * cell_m
    yy, xx = np.meshgrid(i, i, indexing="ij")
    return 100.0 + slope_north * yy + slope_east * xx


def cone(n=81, peak=500.0, cell_m=90.0):
    """Конус: вершина в середине."""
    i = (np.arange(n) - n // 2) * cell_m
    yy, xx = np.meshgrid(i, i, indexing="ij")
    r = np.hypot(yy, xx)
    return np.maximum(0.0, peak - r * 0.05) + 100.0


def valley(n=81, depth=200.0, cell_m=90.0):
    """Долина: жёлоб вдоль оси север-юг, дно посередине."""
    i = (np.arange(n) - n // 2) * cell_m
    yy, xx = np.meshgrid(i, i, indexing="ij")
    return 100.0 + np.minimum(depth, np.abs(xx) * 0.05)


CENTER = (40, 40)


# --- метры в градусе ---------------------------------------------------------

def test_degree_of_longitude_shrinks_towards_the_pole():
    m_lat0, m_lon0 = meters_per_degree(0.0)
    m_lat55, m_lon55 = meters_per_degree(55.0)
    assert m_lat0 == pytest.approx(m_lat55)          # широта не зависит от места
    assert m_lon55 == pytest.approx(m_lon0 * np.cos(np.radians(55.0)))
    assert m_lat0 == pytest.approx(111195.0, rel=1e-3)


# --- положение относительно окрестности --------------------------------------

def test_flat_ground_has_zero_position():
    assert topographic_position(flat(), CENTER, (10, 10)) == pytest.approx(0.0)


def test_peak_is_above_its_surroundings():
    """На вершине конуса превышение положительное и заметное."""
    assert topographic_position(cone(), CENTER, (10, 10)) > 20.0


def test_valley_bottom_is_below_its_surroundings():
    """На дне долины превышение отрицательное — там и застаивается холод.

    Это главный описатель для нашей задачи: наибольшие ошибки прогноза как раз у
    станций, где холодный воздух ведёт себя не так, как в среднем по ячейке.
    """
    assert topographic_position(valley(), CENTER, (10, 10)) < -5.0


def test_position_grows_with_the_radius_on_a_cone():
    """Чем шире круг, тем сильнее вершина возвышается над ним."""
    c = cone()
    small = topographic_position(c, CENTER, (5, 5))
    large = topographic_position(c, CENTER, (20, 20))
    assert large > small > 0


# --- изрезанность и крайние значения -----------------------------------------

def test_flat_ground_is_not_rough():
    assert roughness(flat(), CENTER, (10, 10)) == pytest.approx(0.0)


def test_rough_terrain_is_rough():
    assert roughness(cone(), CENTER, (20, 20)) > 10.0


def test_extremes_are_measured_from_the_point():
    """Превышение над минимумом и недобор до максимума считаются от самой точки."""
    above, below = relief_extremes(cone(), CENTER, (20, 20))
    assert above > 0 and below == pytest.approx(0.0, abs=1e-9)  # мы на вершине
    above_v, below_v = relief_extremes(valley(), CENTER, (20, 20))
    assert above_v == pytest.approx(0.0, abs=1e-9) and below_v > 0  # мы на дне


# --- уклон и экспозиция ------------------------------------------------------

def test_flat_ground_has_zero_slope_and_undefined_aspect():
    """На ровном месте экспозиция не определена и возвращается NaN.

    Подставить туда любое число значило бы кормить модель шумом там, где склона
    нет вовсе.
    """
    s, a = slope_aspect(flat(), CENTER, 90.0, 90.0)
    assert s == pytest.approx(0.0)
    assert np.isnan(a)


@pytest.mark.parametrize("sn, se, want_aspect", [
    (0.1, 0.0, 180.0),    # растёт на север -> склон смотрит на юг
    (-0.1, 0.0, 0.0),     # растёт на юг    -> смотрит на север
    (0.0, 0.1, 270.0),    # растёт на восток -> смотрит на запад
    (0.0, -0.1, 90.0),    # растёт на запад  -> смотрит на восток
])
def test_aspect_points_downhill(sn, se, want_aspect):
    """Экспозиция — направление, КУДА склон смотрит, то есть вниз по уклону."""
    _, a = slope_aspect(plane(slope_north=sn, slope_east=se), CENTER, 90.0, 90.0)
    assert (a - want_aspect + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)


def test_slope_matches_the_gradient():
    s, _ = slope_aspect(plane(slope_north=0.1), CENTER, 90.0, 90.0)
    assert s == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-6)


def test_slope_at_the_edge_is_not_invented():
    """У самого края окна производную не посчитать — возвращается NaN."""
    s, a = slope_aspect(flat(), (0, 0), 90.0, 90.0)
    assert np.isnan(s) and np.isnan(a)


# --- закрытость горизонта ----------------------------------------------------

def test_open_ground_has_no_horizon():
    assert horizon_closure(flat(), CENTER, 90.0, 90.0, 2000.0) == pytest.approx(0.0)


def test_valley_bottom_sees_a_closed_horizon():
    """Со дна долины горизонт поднят — это и определяет ночное выхолаживание."""
    assert horizon_closure(valley(), CENTER, 90.0, 90.0, 2000.0) > 1.0


def test_peak_sees_an_open_horizon():
    assert horizon_closure(cone(), CENTER, 90.0, 90.0, 2000.0) == pytest.approx(0.0)


# --- всё вместе --------------------------------------------------------------

def test_station_terrain_returns_every_descriptor():
    got = station_terrain(valley(), CENTER, LAT, CELL, radii_m=(1000.0, 5000.0))
    for key in ("terr_tpi_1km", "terr_rough_1km", "terr_above_min_1km",
                "terr_below_max_1km", "terr_tpi_5km", "terr_slope",
                "terr_aspect_sin", "terr_aspect_cos", "terr_horizon",
                "terr_dem_elev"):
        assert key in got, key
    assert all(np.isfinite(v) for v in got.values()), got


def test_aspect_is_split_into_sine_and_cosine():
    """Экспозиция подаётся синусом и косинусом, а не углом.

    Иначе 359° и 1° — соседние направления — оказались бы на разных концах
    шкалы, и модель считала бы их противоположными.
    """
    got = station_terrain(plane(slope_east=0.1), CENTER, LAT, CELL,
                          radii_m=(1000.0,))
    assert got["terr_aspect_sin"] ** 2 + got["terr_aspect_cos"] ** 2 == pytest.approx(1.0)


def test_longitude_step_is_shorter_than_latitude_step():
    """Круг на местности — эллипс в пикселях: шаг по долготе короче.

    Без поправки радиус в 20 км по долготе оказался бы вдвое больше нужного, и
    описатель считался бы по вытянутой области.
    """
    got_n = station_terrain(valley(), CENTER, 0.0, CELL, radii_m=(2000.0,))
    got_s = station_terrain(valley(), CENTER, 70.0, CELL, radii_m=(2000.0,))
    # долина вытянута по северу-югу, поэтому расширение круга по долготе
    # захватывает склоны и меняет изрезанность
    assert got_n["terr_rough_2km"] != pytest.approx(got_s["terr_rough_2km"])


# --- чтение и склейка листов матрицы высот -----------------------------------

def write_hgt(path, arr):
    """Записать лист в формате .hgt: юг внизу в памяти, север первым в файле."""
    import numpy as _np
    _np.asarray(arr[::-1], dtype=">i2").tofile(str(path))


def synth_tile(n=121, base=100.0):
    """Лист, где высота кодирует номер строки — так видно, куда что легло."""
    return (base + np.arange(n)[:, None] + np.zeros((1, n))).astype(np.int16)


def test_tile_name_follows_the_south_west_corner():
    from src.postprocessing.terrain import hgt_tile_name
    assert hgt_tile_name(55.7, 93.2) == "N55E093"
    assert hgt_tile_name(55.0, 93.0) == "N55E093"
    assert hgt_tile_name(-1.2, -0.5) == "S02W001"


def test_needed_tiles_include_the_margin():
    """Вокруг точки у края листа нужен и соседний.

    Описатели считаются в круге до 20 км — это около 0,18°. Без запаса станция у
    края получила бы обрезанную окрестность, и признак посчитался бы по половине
    круга, ничем себя не выдав.
    """
    from src.postprocessing.terrain import tiles_for_points
    got = tiles_for_points([55.98], [93.02], margin_deg=0.35)
    assert "N55E093" in got and "N56E093" in got and "N55E092" in got


def test_hgt_round_trip_puts_south_at_the_bottom(tmp_path):
    """Первая строка файла — самая северная, в памяти она должна стать последней."""
    from src.postprocessing.terrain import read_hgt
    p = tmp_path / "N55E093.hgt"
    write_hgt(p, synth_tile())
    got = read_hgt(p)
    assert got.shape == (121, 121)
    assert got[0, 0] == pytest.approx(100.0)     # юг — минимальное значение
    assert got[-1, 0] == pytest.approx(220.0)    # север — максимальное


def test_hgt_reads_gzip(tmp_path):
    import gzip

    from src.postprocessing.terrain import read_hgt
    raw = tmp_path / "N55E093.hgt"
    write_hgt(raw, synth_tile())
    gz = tmp_path / "N55E093.hgt.gz"
    gz.write_bytes(gzip.compress(raw.read_bytes()))
    assert np.allclose(read_hgt(gz), read_hgt(raw), equal_nan=True)


def test_void_values_become_missing(tmp_path):
    """Пустые значения не должны молча стать высотой минус 32 километра."""
    from src.postprocessing.terrain import read_hgt
    t = synth_tile()
    t[10, 10] = -32768
    p = tmp_path / "N55E093.hgt"
    write_hgt(p, t)
    got = read_hgt(p)
    assert np.isnan(got).sum() == 1


def test_broken_file_is_refused(tmp_path):
    from src.postprocessing.terrain import read_hgt
    p = tmp_path / "N55E093.hgt"
    p.write_bytes(b"\x00" * 100)              # не квадрат
    with pytest.raises(ValueError, match="не распознан"):
        read_hgt(p)


def write_tiles(d, lats, lons):
    for la in lats:
        for lo in lons:
            write_hgt(d / f"N{la:02d}E{lo:03d}.hgt", synth_tile())


def test_mosaic_joins_tiles_without_a_seam(tmp_path):
    """Склейка соседних листов не задваивает общий край.

    У листов последняя строка совпадает с первой строкой соседнего. Оставь обе —
    и сетка перестанет быть равномерной, а по ней считаются расстояния в метрах.

    Точка взята у самого угла листа, чтобы с запасом понадобились четыре листа:
    ровно тот случай, ради которого склейка и нужна.
    """
    from src.postprocessing.terrain import load_mosaic
    write_tiles(tmp_path, (55, 56), (93, 94))
    mosaic, (i, j), cell = load_mosaic(tmp_path, 55.9, 93.9, margin_deg=0.35)
    n = 120                                   # 121 минус перекрытие
    assert mosaic.shape == (2 * n, 2 * n)
    assert cell == pytest.approx(1.0 / 120)


def test_mosaic_points_at_the_right_cell(tmp_path):
    """Номер строки и столбца соответствуют заданным координатам."""
    from src.postprocessing.terrain import load_mosaic
    write_tiles(tmp_path, (54, 55), (92, 93))
    _, (i, j), cell = load_mosaic(tmp_path, 55.25, 93.05, margin_deg=0.35)
    # начало склейки — юго-западный угол самого южного и западного листа
    assert i == pytest.approx(round((55.25 - 54) / cell), abs=1)
    assert j == pytest.approx(round((93.05 - 92) / cell), abs=1)


def test_single_tile_is_enough_in_the_middle(tmp_path):
    """В середине листа склейка не нужна — берётся один лист."""
    from src.postprocessing.terrain import load_mosaic
    write_tiles(tmp_path, (55,), (93,))
    mosaic, _, _ = load_mosaic(tmp_path, 55.5, 93.5, margin_deg=0.35)
    assert mosaic.shape == (120, 120)


def test_missing_tile_says_which_one(tmp_path):
    """Не нашёлся лист — сообщение называет его и говорит, чем получить список."""
    from src.postprocessing.terrain import load_mosaic
    with pytest.raises(FileNotFoundError, match="N55E093"):
        load_mosaic(tmp_path, 55.5, 93.5, margin_deg=0.1)


# --- сборка таблицы по станциям и склейка с корпусом -------------------------

def test_terrain_table_is_built_for_every_station(tmp_path):
    """Таблица считается по станциям и содержит все описатели."""
    import importlib.util as u
    from pathlib import Path as _P
    spec = u.spec_from_file_location(
        "add_terrain", _P(__file__).resolve().parents[2] / "scripts/postproc/add_terrain.py")
    at = u.module_from_spec(spec)
    spec.loader.exec_module(at)

    dem = tmp_path / "dem"
    dem.mkdir()
    write_tiles(dem, (54, 55, 56), (92, 93, 94))
    stations = [{"usaf": "20001", "lat": 55.3, "lon": 93.3, "elev": 150.0},
                {"usaf": "20002", "lat": 55.6, "lon": 93.6, "elev": 180.0}]
    t = at.build_table(stations, dem, [1000.0, 5000.0])
    assert len(t) == 2
    assert set(t["station_usaf"]) == {"20001", "20002"}
    for c in ("terr_tpi_1km", "terr_rough_5km", "terr_slope", "terr_horizon"):
        assert c in t.columns


def test_station_without_tiles_is_skipped_not_fatal(tmp_path, capsys):
    """Станция без листов пропускается, а не роняет счёт остальных.

    Листов над морем не существует вовсе, и падать из-за одной приморской
    станции, теряя семьдесят посчитанных, было бы неверно.
    """
    import importlib.util as u
    from pathlib import Path as _P
    spec = u.spec_from_file_location(
        "add_terrain2", _P(__file__).resolve().parents[2] / "scripts/postproc/add_terrain.py")
    at = u.module_from_spec(spec)
    spec.loader.exec_module(at)

    dem = tmp_path / "dem"
    dem.mkdir()
    write_tiles(dem, (54, 55, 56), (92, 93, 94))
    stations = [{"usaf": "20001", "lat": 55.3, "lon": 93.3, "elev": 150.0},
                {"usaf": "30001", "lat": 10.0, "lon": 10.0, "elev": 5.0}]
    t = at.build_table(stations, dem, [1000.0])
    assert len(t) == 1 and t.iloc[0]["station_usaf"] == "20001"
    assert "без листов" in capsys.readouterr().out


def test_elevation_mismatch_is_reported(tmp_path):
    """Расхождение высоты из матрицы с заявленной в описании станции.

    Сотня метров расхождения означает, что координаты станции неточны — а тогда
    и все описатели вокруг неё считаны не в том месте. Признак сам по себе
    полезен, но главное — по нему видно испорченные записи.
    """
    import importlib.util as u
    from pathlib import Path as _P
    spec = u.spec_from_file_location(
        "add_terrain3", _P(__file__).resolve().parents[2] / "scripts/postproc/add_terrain.py")
    at = u.module_from_spec(spec)
    spec.loader.exec_module(at)

    dem = tmp_path / "dem"
    dem.mkdir()
    write_tiles(dem, (54, 55), (92, 93))
    t = at.build_table([{"usaf": "20001", "lat": 55.3, "lon": 93.3, "elev": 0.0}],
                       dem, [1000.0])
    got = t.iloc[0]
    assert got["terr_elev_mismatch"] == pytest.approx(got["terr_dem_elev"])


def test_terrain_features_are_picked_up_by_the_dataset(tmp_path):
    """Датасет подхватывает terr_* сам, как подхватывает nb_* и наблюдения."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conftest import load_module
    ds = load_module("src/postprocessing/neural/dataset.py", "ds_terr")

    import pandas as pd
    n = 2000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n).astype("float32")
                       for c in ds.DEFAULT_FEATURES})
    df["station_lat"] = df.pop("lat")
    df["station_lon"] = df.pop("lon")
    df["station_elev"] = df.pop("elev")
    df["station_usaf"] = "20001"
    df["init_time_utc"] = pd.Timestamp("2020-01-01")
    df["valid_time_utc"] = pd.Timestamp("2020-01-02")
    df["lead_h"] = 24
    df["obs_t2m_K"] = 273.0 + rng.normal(size=n)
    df["obs_u10"] = rng.normal(size=n)
    df["obs_v10"] = rng.normal(size=n)
    df["terr_tpi_1km"] = rng.normal(size=n)
    df["terr_slope"] = rng.uniform(0, 30, n)
    p = tmp_path / "c.parquet"
    df.to_parquet(p, index=False)

    got = ds.StationCorpusDataset(p, station_to_idx={"20001": 0})
    assert "terr_tpi_1km" in got.feature_cols and "terr_slope" in got.feature_cols


def cone_metric(n, cell_deg, lat=LAT, peak=500.0, drop=0.05):
    """Конус на сетке с настоящим шагом по каждой оси.

    В градусах ячейка квадратная, в метрах — нет: градус долготы короче. Синтетика
    обязана это повторять, иначе сравнение разрешений сравнивало бы разный рельеф.
    """
    from src.postprocessing.terrain import meters_per_degree
    m_lat, m_lon = meters_per_degree(lat)
    dy, dx = cell_deg * m_lat, cell_deg * m_lon
    i = np.arange(n) - n // 2
    yy, xx = np.meshgrid(i * dy, i * dx, indexing="ij")
    return np.maximum(0.0, peak - np.hypot(yy, xx) * drop) + 100.0


def test_descriptors_do_not_depend_on_grid_resolution():
    """Один рельеф на сетке 3\u2033 и 1\u2033 даёт те же признаки.

    Не отвлечённая аккуратность: скачанные листы оказались в 1 угловую секунду,
    а не в 3, как закладывалось при разработке. Радиусы задаются в метрах и
    переводятся в точки через шаг сетки — если бы где-то остался счёт в точках,
    все описатели поехали бы втрое и молча, ничем себя не выдав.

    Станция стоит НЕ на вершине, иначе уклон, экспозиция и горизонт вышли бы
    нулями и проверка их не касалась бы вовсе.
    """
    from src.postprocessing.terrain import meters_per_degree, station_terrain
    m_lat, m_lon = meters_per_degree(LAT)
    got = {}
    for cell_deg, n in ((1.0 / 1200.0, 161), (1.0 / 3600.0, 481)):
        dy, dx = cell_deg * m_lat, cell_deg * m_lon
        c = (n // 2 + int(round(1800 / dy)), n // 2 + int(round(900 / dx)))
        got[n] = station_terrain(cone_metric(n, cell_deg), c, LAT, cell_deg,
                                 radii_m=(1000.0, 3000.0))
    coarse, fine = got[161], got[481]
    assert set(coarse) == set(fine)
    for k in coarse:
        scale = max(abs(coarse[k]), 1.0)
        assert abs(fine[k] - coarse[k]) / scale < 0.03, (
            f"{k}: {coarse[k]:.4f} против {fine[k]:.4f} — зависит от разрешения")
    # проверка не пустая: величины, которые могли бы выйти нулями, ненулевые
    assert coarse["terr_slope"] > 1.0 and coarse["terr_horizon"] > 0.5


def test_tiles_of_different_resolution_are_refused(tmp_path):
    """Смесь листов 1\u2033 и 3\u2033 в одном каталоге — внятный отказ.

    Шаг сетки берётся от последнего прочитанного листа, поэтому часть мозаики
    оказалась бы растянута втрое. numpy упал бы и сам, но на несовпадении
    размеров массивов — по такому сообщению причину не найти.
    """
    import pytest as _pytest

    from src.postprocessing.terrain import load_mosaic
    write_hgt(tmp_path / "N55E093.hgt", synth_tile(n=121))
    write_hgt(tmp_path / "N55E094.hgt", synth_tile(n=61))
    with _pytest.raises(ValueError, match="разного разрешения"):
        load_mosaic(tmp_path, 55.5, 93.9, margin_deg=0.3)


def test_mosaic_step_follows_the_tile_size(tmp_path):
    """Шаг сетки выводится из размера листа, а не задан числом."""
    from src.postprocessing.terrain import load_mosaic
    for n in (121, 361):
        d = tmp_path / f"t{n}"
        d.mkdir()
        write_hgt(d / "N55E093.hgt", synth_tile(n=n))
        _, _, cell = load_mosaic(d, 55.5, 93.5, margin_deg=0.1)
        assert cell == pytest.approx(1.0 / (n - 1))
