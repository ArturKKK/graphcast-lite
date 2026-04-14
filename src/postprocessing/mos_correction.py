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
        # Try alternate naming conventions (e.g. "u10" ↔ "10u")
        alt_names = {"u10": "10u", "10u": "u10", "v10": "10v", "10v": "v10"}
        alt = alt_names.get(name)
        if alt and alt in var_order:
            return float(vals[var_order.index(alt)])
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _idw_interpolate_bias(
    station_biases: dict[int, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    n_steps: int,
    power: float = 2.0,
    max_radius_km: float = 300.0,
) -> np.ndarray:
    """Interpolate station biases to all grid nodes via IDW.

    Parameters
    ----------
    station_biases : {grid_idx: array(n_steps,)} — bias per station grid point
    latitudes, longitudes : 1-d grid coords (G,)
    n_steps : forecast steps
    power : IDW exponent (2 = inverse-square)
    max_radius_km : only stations within this radius contribute

    Returns
    -------
    bias_field : ndarray (G, n_steps) — interpolated bias for every node
    """
    G = len(latitudes)
    bias_field = np.zeros((G, n_steps), dtype=np.float64)

    if not station_biases:
        return bias_field

    st_indices = list(station_biases.keys())
    st_lats = np.array([latitudes[i] for i in st_indices])
    st_lons = np.array([longitudes[i] for i in st_indices])
    st_bias_arr = np.array([station_biases[i] for i in st_indices])  # (K, steps)

    for g in range(G):
        if g in station_biases:
            # station grid point — use its own exact bias
            bias_field[g, :] = station_biases[g]
            continue

        lat_g, lon_g = float(latitudes[g]), float(longitudes[g])
        dists = np.array([_haversine_km(lat_g, lon_g, float(st_lats[k]), float(st_lons[k]))
                          for k in range(len(st_indices))])

        mask = dists < max_radius_km
        if not mask.any():
            continue  # too far from all stations — no correction

        d = dists[mask]
        d = np.maximum(d, 0.1)  # avoid division by zero
        w = 1.0 / d ** power
        w /= w.sum()

        bias_field[g, :] = (w[:, None] * st_bias_arr[mask]).sum(axis=0)

    return bias_field


def apply_learned_mos_t2m(
    prediction_phys: np.ndarray,
    var_order: list[str],
    model_bundle: dict,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    forecast_valid_times: list[datetime],
    station_lat: float = 56.173,
    station_lon: float = 92.493,
    station_elev: float = 287.0,
    stations: list[dict] | None = None,
    spatial_idw: bool = False,
    idw_power: float = 2.0,
    idw_max_radius_km: float = 300.0,
) -> np.ndarray:
    """Apply learned MOS correction to t2m at station grid points.

    If *stations* is provided (list of dicts with lat/lon/elev/name),
    corrections are applied at every unique grid point.  Otherwise
    falls back to the single station_lat/lon/elev.

    When *spatial_idw* is True, station biases are first computed at
    each station's nearest grid point, then interpolated to ALL grid
    nodes via inverse-distance weighting (IDW).

    Parameters
    ----------
    prediction_phys : ndarray, shape (G, steps, C)
    var_order : list[str]
    model_bundle : dict from load_learned_mos()
    latitudes, longitudes : 1-d arrays of grid coordinates
    forecast_valid_times : UTC valid times for each step
    station_lat, station_lon, station_elev : fallback single station
    stations : optional list of {"lat", "lon", "elev", "name"} dicts
    spatial_idw : if True, interpolate bias to all grid nodes
    idw_power : IDW exponent (default 2.0)
    idw_max_radius_km : max radius for IDW interpolation (default 300 km)

    Returns
    -------
    corrected : ndarray, same shape
    n_corrected : int, number of corrected grid points
    """
    if "t2m" not in var_order:
        return prediction_phys

    model = model_bundle["model"]
    corrected = prediction_phys.copy()
    t2m_idx = var_order.index("t2m")
    n_steps = len(forecast_valid_times)

    # Build station list
    if stations is None:
        stations = [{"lat": station_lat, "lon": station_lon,
                      "elev": station_elev, "name": "default"}]

    # Map each station to nearest grid index; deduplicate by grid_idx
    grid_stations: dict[int, list[dict]] = {}
    for st in stations:
        dist = (latitudes - st["lat"])**2 + (longitudes - st["lon"])**2
        gidx = int(np.argmin(dist))
        grid_stations.setdefault(gidx, []).append(st)

    # Step 1: compute bias at each station grid point
    station_biases: dict[int, np.ndarray] = {}
    for grid_idx, st_group in grid_stations.items():
        biases_per_step = np.zeros(n_steps, dtype=np.float64)
        prev_t2m_c = None
        for step_idx, valid_time in enumerate(forecast_valid_times):
            biases = []
            for st in st_group:
                features = _build_features_from_forecast(
                    corrected, var_order, grid_idx, step_idx, valid_time,
                    st["lat"], st["lon"], st["elev"], prev_t2m_c,
                )
                biases.append(model.predict(features.reshape(1, -1))[0])
            biases_per_step[step_idx] = float(np.mean(biases))
            prev_t2m_c = float(corrected[grid_idx, step_idx, t2m_idx]
                               + biases_per_step[step_idx]) - 273.15
        station_biases[grid_idx] = biases_per_step

    # Step 2: apply corrections
    if spatial_idw and len(station_biases) >= 2:
        # IDW interpolation → apply bias to ALL grid nodes
        bias_field = _idw_interpolate_bias(
            station_biases, latitudes, longitudes, n_steps,
            power=idw_power, max_radius_km=idw_max_radius_km,
        )
        corrected[:, :, t2m_idx] += bias_field
        n_corrected = int((np.abs(bias_field).max(axis=1) > 1e-6).sum())
    else:
        # Original: correct only station grid points
        for grid_idx, biases_per_step in station_biases.items():
            corrected[grid_idx, :, t2m_idx] += biases_per_step
        n_corrected = len(station_biases)

    return corrected, n_corrected
