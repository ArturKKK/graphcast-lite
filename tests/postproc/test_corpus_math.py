"""Расчёты сборщика корпуса.

Все они молчаливые: ошибка здесь не роняет счёт, а тихо портит корпус, на
котором потом два часа обучается постпроцессор и строятся числа для статьи.
"""
import math
from datetime import datetime

import numpy as np
import pytest

from src.postprocessing.corpus_math import (bilinear_sample, compute_forcing,
                                            dewpoint_depression_K,
                                            solar_elevation, wind_components)


# --- ветер -------------------------------------------------------------------

@pytest.mark.parametrize("direction, u, v", [
    (0.0,   0.0, -5.0),   # ветер с севера дует на юг
    (90.0, -5.0,  0.0),   # с востока — на запад
    (180.0, 0.0,  5.0),   # с юга — на север
    (270.0, 5.0,  0.0),   # с запада — на восток
])
def test_wind_direction_convention(direction, u, v):
    """Направление метеорологическое: откуда дует, от севера по часовой."""
    gu, gv = wind_components([5.0], [direction])
    assert gu[0] == pytest.approx(u, abs=1e-9)
    assert gv[0] == pytest.approx(v, abs=1e-9)


def test_calm_with_missing_direction_is_zero_not_nan():
    """Штиль даёт нулевой вектор, а не NaN.

    При штиле направления не существует, и оно может быть помечено пропуском.
    Прямая формула u = -V·sin(dd) дала бы тогда NaN, а вектор очевидно нулевой.

    Насколько это важно — измерено, а не предположено. На красноярском корпусе
    (3 083 614 строк) штилей 19 801, и у ВСЕХ направление проставлено, поэтому
    заплатка вернула ноль строк: дефект настоящий, но здесь ни разу не сработал.
    Пропусков в ветре 695 159, это строки без самой скорости — там ветер
    неизвестен, и выбрасывать их правильно. Тест остаётся защитой на случай
    другого набора наблюдений, где штиль кодируют иначе.
    """
    u, v = wind_components([0.0], [np.nan])
    assert u[0] == 0.0 and v[0] == 0.0


def test_calm_with_any_direction_is_zero():
    u, v = wind_components([0.0, 0.0], [0.0, 123.0])
    assert np.all(u == 0.0) and np.all(v == 0.0)


def test_missing_direction_with_real_wind_stays_missing():
    """А вот настоящий ветер без направления — по-прежнему пропуск.

    Скорость есть, направления нет: составляющие неизвестны, и выдумывать их
    нельзя. Отбрасывать такую строку — правильно.
    """
    u, v = wind_components([4.0], [np.nan])
    assert np.isnan(u[0]) and np.isnan(v[0])


def test_speed_is_preserved():
    rng = np.random.default_rng(0)
    sp = rng.uniform(0.5, 20, 200)
    dd = rng.uniform(0, 360, 200)
    u, v = wind_components(sp, dd)
    assert np.allclose(np.hypot(u, v), sp)


# --- билинейная выборка ------------------------------------------------------

def test_sampling_at_a_node_returns_that_node():
    lons = np.arange(0.0, 10.1, 1.0)
    lats = np.arange(50.0, 55.1, 1.0)
    g = np.arange(len(lons) * len(lats), dtype=np.float32).reshape(len(lons), len(lats))
    got = bilinear_sample(g, lons, lats, np.array([3.0]), np.array([52.0]))
    assert got[0] == pytest.approx(g[3, 2])


def test_sampling_in_the_middle_is_the_average():
    lons = np.array([0.0, 1.0])
    lats = np.array([0.0, 1.0])
    g = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    got = bilinear_sample(g, lons, lats, np.array([0.5]), np.array([0.5]))
    assert got[0] == pytest.approx(15.0)


def test_linear_field_is_reproduced_exactly():
    """На линейном поле билинейная выборка обязана давать точное значение."""
    lons = np.arange(0.0, 20.1, 0.5)
    lats = np.arange(40.0, 60.1, 0.5)
    LO, LA = np.meshgrid(lons, lats, indexing="ij")
    g = (2.0 * LO + 3.0 * LA).astype(np.float32)
    rng = np.random.default_rng(1)
    q_lon = rng.uniform(0.5, 19.5, 300)
    q_lat = rng.uniform(40.5, 59.5, 300)
    got = bilinear_sample(g, lons, lats, q_lon, q_lat)
    assert np.allclose(got, 2 * q_lon + 3 * q_lat, atol=2e-3)


def test_global_grid_wraps_around_the_meridian():
    """У глобальной сетки долгота замыкается: 359,8° лежит между 359,3° и 0°."""
    lons = np.arange(0.0, 360.0, 0.703125)
    lats = np.array([50.0, 51.0])
    g = np.zeros((len(lons), 2), dtype=np.float32)
    g[-1, :] = 10.0     # последняя ячейка перед нулём
    g[0, :] = 20.0      # сама нулевая
    got = bilinear_sample(g, lons, lats, np.array([lons[-1] + 0.3515625]),
                          np.array([50.0]))
    assert got[0] == pytest.approx(15.0, abs=0.2), "замыкание по кругу не сработало"


def test_regional_grid_clamps_instead_of_wrapping():
    """У вставки замыкания нет: запрос за краем прижимается к краю."""
    lons = np.arange(83.0, 98.01, 0.25)
    lats = np.arange(50.0, 60.01, 0.25)
    g = np.zeros((len(lons), len(lats)), dtype=np.float32)
    g[0, :] = 1.0
    g[-1, :] = 99.0
    left = bilinear_sample(g, lons, lats, np.array([70.0]), np.array([55.0]))
    assert left[0] == pytest.approx(1.0), "запрос слева должен прижаться к краю"


# --- временные признаки ------------------------------------------------------

def test_forcing_is_periodic_over_the_day():
    a = compute_forcing(datetime(2020, 3, 15, 0))
    b = compute_forcing(datetime(2020, 3, 16, 0))
    assert a[0] == pytest.approx(b[0], abs=1e-6)
    assert a[1] == pytest.approx(b[1], abs=1e-6)


def test_forcing_hour_matches_the_circle():
    f = compute_forcing(datetime(2020, 1, 1, 6))
    assert f[0] == pytest.approx(1.0, abs=1e-6)   # 06 UTC — четверть круга
    assert f[1] == pytest.approx(0.0, abs=1e-6)


def test_forcing_is_bounded():
    for month in range(1, 13):
        f = compute_forcing(datetime(2020, month, 15, 12))
        assert np.all(np.abs(f) <= 1.0 + 1e-6)


# --- высота солнца -----------------------------------------------------------

def test_sun_is_higher_at_local_noon_than_at_local_midnight():
    lat, lon = 56.0, 93.0                 # Красноярск, UTC+6 примерно
    noon = solar_elevation(lat, lon, datetime(2020, 6, 21, 6))
    midnight = solar_elevation(lat, lon, datetime(2020, 6, 21, 18))
    assert noon > midnight + 30


def test_polar_night_sun_stays_below_horizon():
    """За полярным кругом в декабре солнце не восходит."""
    for hour in range(0, 24, 3):
        assert solar_elevation(78.0, 15.0, datetime(2020, 12, 21, hour)) < 0


def test_summer_noon_is_higher_than_winter_noon():
    lat, lon = 56.0, 93.0
    summer = solar_elevation(lat, lon, datetime(2020, 6, 21, 6))
    winter = solar_elevation(lat, lon, datetime(2020, 12, 21, 6))
    assert summer > winter + 40


# --- дефицит точки росы ------------------------------------------------------

def test_saturated_air_has_no_dewpoint_depression():
    """При насыщении температура и точка росы совпадают."""
    t_K, sp = 283.15, 101325.0
    e_sat_hPa = 6.112 * math.exp(17.67 * 10.0 / (10.0 + 243.5))   # при 10 °C
    e = e_sat_hPa * 100.0
    q = 0.622 * e / (sp - 0.378 * e)
    assert dewpoint_depression_K(t_K, q, sp) == pytest.approx(0.0, abs=0.05)


def test_drier_air_gives_a_larger_depression():
    t_K, sp = 283.15, 101325.0
    wet = dewpoint_depression_K(t_K, 0.006, sp)
    dry = dewpoint_depression_K(t_K, 0.002, sp)
    assert dry > wet > 0


@pytest.mark.parametrize("q, sp", [(0.0, 101325.0), (-0.001, 101325.0),
                                   (0.005, 0.0), (np.nan, 101325.0),
                                   (0.005, np.nan)])
def test_impossible_input_gives_nan_not_a_crash(q, sp):
    """Мусор на входе даёт NaN, а не исключение посреди двухчасового счёта."""
    assert np.isnan(dewpoint_depression_K(283.15, q, sp))


# --- заплатка на готовый корпус ---------------------------------------------

def test_fix_calm_wind_recovers_rows(tmp_path):
    """Заплатка возвращает штили, не трогая всё остальное."""
    import importlib.util
    import pandas as pd
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "fix_calm", _P(__file__).resolve().parents[2]
        / "scripts" / "postproc" / "fix_calm_wind.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)

    df = pd.DataFrame({
        "obs_ws": [0.0, 3.0, 5.0, 0.0],
        "obs_wd": [np.nan, 90.0, np.nan, 270.0],
    })
    df["obs_u10"] = -df["obs_ws"] * np.sin(np.deg2rad(df["obs_wd"]))
    df["obs_v10"] = -df["obs_ws"] * np.cos(np.deg2rad(df["obs_wd"]))
    assert df["obs_u10"].isna().sum() == 2         # оба штиля потеряны

    out, stat = fx.fix_frame(df)
    assert stat["возвращено строк"] == 1           # штиль с пропущенным направлением
    assert stat["штилей"] == 2
    assert out.loc[0, "obs_u10"] == 0.0 and out.loc[0, "obs_v10"] == 0.0
    assert out.loc[1, "obs_u10"] == pytest.approx(-3.0)   # не тронуто
    assert np.isnan(out.loc[2, "obs_u10"])         # настоящий ветер без направления
    assert out.loc[3, "obs_u10"] == 0.0            # штиль с направлением тоже нуль


def test_fix_calm_wind_refuses_without_raw_columns(tmp_path):
    import importlib.util
    import pandas as pd
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "fix_calm2", _P(__file__).resolve().parents[2]
        / "scripts" / "postproc" / "fix_calm_wind.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    with pytest.raises(SystemExit, match="пересобирать корпус"):
        fx.fix_frame(pd.DataFrame({"obs_u10": [1.0]}))
