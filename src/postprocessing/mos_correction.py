"""MOS (Model Output Statistics) bias correction for t2m.

Loads a pre-computed bias table (JSON) and applies additive corrections
to physical-unit predictions based on forecast valid time (month, hour_utc).

Also supports learned MOS (HistGradientBoostingRegressor) for ML-based correction.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


def load_mos_table(path: str | Path) -> dict:
    """Load MOS bias table from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_t2m_bias(mos_table: dict, valid_time: datetime) -> float:
    """Get t2m bias correction (°C) for a given valid time (UTC).

    Returns the additive correction: corrected = model + bias.
    """
    month = str(valid_time.month)
    hour = str(valid_time.hour)
    return mos_table["bias_table"].get(month, {}).get(hour, 0.0)


def apply_mos_t2m(
    prediction_phys: np.ndarray,
    var_order: list[str],
    mos_table: dict,
    forecast_valid_times: list[datetime],
) -> np.ndarray:
    """Apply MOS bias correction to physical-unit predictions.

    Parameters
    ----------
    prediction_phys : ndarray, shape (G, steps, C)
        Physical-unit predictions (t2m expected in Kelvin).
    var_order : list[str]
        Variable names.
    mos_table : dict
        Loaded MOS table (from load_mos_table).
    forecast_valid_times : list[datetime]
        UTC valid times for each forecast step.

    Returns
    -------
    corrected : ndarray, same shape as prediction_phys
        Corrected predictions (only t2m channel is modified).
    """
    if "t2m" not in var_order:
        return prediction_phys

    corrected = prediction_phys.copy()
    t2m_idx = var_order.index("t2m")

    for step_idx, valid_time in enumerate(forecast_valid_times):
        bias_c = get_t2m_bias(mos_table, valid_time)
        # Bias is in °C; for t2m in Kelvin the additive offset is identical
        corrected[:, step_idx, t2m_idx] += bias_c

    return corrected


# ── Learned MOS (ML-based) ──────────────────────────────────────────

def _solar_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Approximate solar elevation angle (degrees). Spencer 1971."""
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    gamma = 2 * math.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                     - 0.032077 * math.sin(gamma)
                     - 0.014615 * math.cos(2 * gamma)
                     - 0.04089 * math.sin(2 * gamma))
    solar_time = hour * 60 + eqt + 4 * lon_deg
    ha = math.radians(solar_time / 4.0 - 180.0)
    lat_rad = math.radians(lat_deg)
    sin_elev = (math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.cos(ha))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def load_learned_mos(path: str | Path) -> dict:
    """Load learned MOS model bundle from joblib file."""
    return joblib.load(path)


def _build_features_from_forecast(
    prediction_phys: np.ndarray,
    var_order: list[str],
    grid_idx: int,
    step_idx: int,
    valid_time: datetime,
    station_lat: float,
    station_lon: float,
    station_elev: float,
    prev_t2m_c: float | None,
) -> np.ndarray:
    """Build feature vector from GNN forecast for a single grid point and step.

    Returns array of shape (20,) matching FEATURE_COLUMNS from build_learned_mos.py.
    Uses NaN for unavailable features (dewpoint, cloudcover, radiation) —
    HistGBR handles missing values natively.
    """
    vals = prediction_phys[grid_idx, step_idx, :]

    def get_var(name: str) -> float:
        if name in var_order:
            return float(vals[var_order.index(name)])
        return float("nan")

    # t2m: model outputs Kelvin, convert to °C
    t2m_k = get_var("t2m")
    t2m_c = t2m_k - 273.15

    # Wind: u10, v10 → speed, direction
    u10 = get_var("u10")
    v10 = get_var("v10")
    ws = math.sqrt(u10**2 + v10**2) if not (math.isnan(u10) or math.isnan(v10)) else float("nan")
    wd_rad = math.atan2(-u10, -v10) if not (math.isnan(u10) or math.isnan(v10)) else float("nan")
    wind_dir_sin = math.sin(wd_rad) if not math.isnan(wd_rad) else float("nan")
    wind_dir_cos = math.cos(wd_rad) if not math.isnan(wd_rad) else float("nan")

    # Surface pressure: Pa → hPa
    sp_pa = get_var("sp")
    sp_hpa = sp_pa / 100.0 if not math.isnan(sp_pa) else float("nan")

    # Precipitation
    precip = get_var("tp")

    # Unavailable from GNN: dewpoint, cloudcover, shortwave radiation
    dewpoint_c = float("nan")
    cloudcover = float("nan")
    shortwave = float("nan")
    dewpoint_depression = float("nan")

    # Time features
    hour = valid_time.hour
    doy = valid_time.timetuple().tm_yday
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    doy_sin = math.sin(2 * math.pi * doy / 365.25)
    doy_cos = math.cos(2 * math.pi * doy / 365.25)

    sol_elev = _solar_elevation(station_lat, station_lon, valid_time)

    # Lag features
    t2m_lag6h = prev_t2m_c if prev_t2m_c is not None else float("nan")
    delta_t2m_6h = (t2m_c - prev_t2m_c) if prev_t2m_c is not None else float("nan")

    # Feature order must match FEATURE_COLUMNS in build_learned_mos.py
    return np.array([
        t2m_c, dewpoint_c, ws, wind_dir_sin, wind_dir_cos,
        sp_hpa, cloudcover, shortwave, precip,
        hour_sin, hour_cos, doy_sin, doy_cos, sol_elev,
        dewpoint_depression, t2m_lag6h, delta_t2m_6h,
        station_lat, station_lon, station_elev,
    ])


def apply_learned_mos_t2m(
    prediction_phys: np.ndarray,
    var_order: list[str],
    model_bundle: dict,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    forecast_valid_times: list[datetime],
    station_lat: float = 56.017,
    station_lon: float = 92.75,
    station_elev: float = 277.0,
) -> np.ndarray:
    """Apply learned MOS correction to t2m at the closest grid point to station.

    Parameters
    ----------
    prediction_phys : ndarray, shape (G, steps, C)
    var_order : list[str]
    model_bundle : dict from load_learned_mos()
    latitudes, longitudes : 1-d arrays of grid coordinates
    forecast_valid_times : UTC valid times for each step
    station_lat, station_lon, station_elev : station location

    Returns
    -------
    corrected : ndarray, same shape
    """
    if "t2m" not in var_order:
        return prediction_phys

    model = model_bundle["model"]
    corrected = prediction_phys.copy()
    t2m_idx = var_order.index("t2m")

    # Find closest grid point
    dist = (latitudes - station_lat)**2 + (longitudes - station_lon)**2
    grid_idx = int(np.argmin(dist))

    prev_t2m_c = None
    for step_idx, valid_time in enumerate(forecast_valid_times):
        features = _build_features_from_forecast(
            corrected, var_order, grid_idx, step_idx, valid_time,
            station_lat, station_lon, station_elev, prev_t2m_c,
        )
        bias = model.predict(features.reshape(1, -1))[0]
        corrected[grid_idx, step_idx, t2m_idx] += bias
        prev_t2m_c = float(corrected[grid_idx, step_idx, t2m_idx]) - 273.15

    return corrected
