"""Расчёты сборщика корпуса: выборка с сетки, форсинги, влажность, ветер.

Вынесено из scripts/postproc/build_corpus.py. Там эти функции сидели рядом с
torch, моделью и памятью на сотни гигабайт, и проверить их можно было только
двухчасовым прогоном. Здесь numpy и math, поэтому они покрыты тестами.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np


def bilinear_sample(grid_data_lonlat, g_lons, g_lats, q_lon, q_lat):
    """Билинейная выборка с равномерной сетки (n_lon, n_lat) в произвольных точках.

    ``g_lons`` и ``g_lats`` должны возрастать. Долгота замыкается по кругу только
    если сетка глобальная, то есть её охват близок к 360°; у региональной вставки
    замыкания нет, и запросы за краем прижимаются к краю.
    """
    n_lon = len(g_lons)
    n_lat = len(g_lats)
    lon_min = g_lons[0]
    lon_max = g_lons[-1]
    dlon = g_lons[1] - g_lons[0]
    if (lon_max - lon_min + dlon) > 359.0:      # глобальная сетка — замыкаем
        q_lon_n = np.mod(q_lon - lon_min, 360.0) + lon_min
        di = (q_lon_n - lon_min) / dlon
        i0 = np.floor(di).astype(np.int64) % n_lon
        i1 = (i0 + 1) % n_lon
        fx = di - np.floor(di)
    else:
        q_lon_c = np.clip(q_lon, lon_min, lon_max)
        di = (q_lon_c - lon_min) / dlon
        i0 = np.clip(np.floor(di).astype(np.int64), 0, n_lon - 2)
        i1 = i0 + 1
        fx = np.clip(di - i0, 0.0, 1.0)

    dlat = g_lats[1] - g_lats[0]
    dj = (q_lat - g_lats[0]) / dlat
    j0 = np.clip(np.floor(dj).astype(np.int64), 0, n_lat - 2)
    j1 = j0 + 1
    fy = np.clip(dj - j0, 0.0, 1.0)

    g = np.asarray(grid_data_lonlat, dtype=np.float32)
    v = ((1 - fx) * (1 - fy) * g[i0, j0] + fx * (1 - fy) * g[i1, j0]
         + (1 - fx) * fy * g[i0, j1] + fx * fy * g[i1, j1])
    return np.asarray(v, dtype=np.float32)


def compute_forcing(dt: datetime) -> np.ndarray:
    """Четыре временных признака: sin/cos часа и sin/cos дня года."""
    h = dt.hour + dt.minute / 60.0
    doy = dt.timetuple().tm_yday
    return np.array([
        math.sin(2 * math.pi * h / 24.0),
        math.cos(2 * math.pi * h / 24.0),
        math.sin(2 * math.pi * doy / 365.25),
        math.cos(2 * math.pi * doy / 365.25),
    ], dtype=np.float32)


def solar_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Высота солнца над горизонтом, градусы. Формула Спенсера, 1971. Время UTC."""
    doy = dt.timetuple().tm_yday
    gamma = 2.0 * math.pi * (doy - 1 + (dt.hour - 12) / 24.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + eq_time + 4.0 * lon_deg
    ha = math.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    return math.degrees(math.asin(
        math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)))


def dewpoint_depression_K(t2m_K: float, q_kg_kg: float, sp_Pa: float) -> float:
    """Дефицит точки росы T - Td, К. Приближение Магнуса по удельной влажности."""
    if not np.isfinite(q_kg_kg) or q_kg_kg <= 0 or not np.isfinite(sp_Pa) or sp_Pa <= 0:
        return float("nan")
    e_hPa = q_kg_kg * sp_Pa / (0.622 + 0.378 * q_kg_kg) / 100.0
    if e_hPa <= 0:
        return float("nan")
    ln_term = math.log(max(e_hPa / 6.112, 1e-6))
    td_K = 243.5 * ln_term / (17.67 - ln_term) + 273.15
    return float(t2m_K - td_K)


def wind_components(speed, direction):
    """Составляющие ветра из скорости и метеорологического направления.

    Направление — откуда дует, в градусах от севера по часовой стрелке, поэтому
    u = -V·sin(dd), v = -V·cos(dd).

    Штиль. При нулевой скорости направления не существует, и в ISD-Lite оно
    помечено пропуском. Прямая формула давала бы NaN, и штили целиком выпадали
    бы из оценки ветра — а это как раз случаи, где относительная ошибка модели
    наибольшая, и терять их нельзя. При нулевой скорости вектор нулевой,
    независимо от направления.
    """
    speed = np.asarray(speed, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    rad = np.deg2rad(direction)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    calm = speed == 0.0
    u = np.where(calm, 0.0, u)
    v = np.where(calm, 0.0, v)
    return u, v
