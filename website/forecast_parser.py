#!/usr/bin/env python3
"""Parse forecast.pt -> forecast.json for the web frontend.

Extracts:
  - Core-city forecast (CORE_CITY_BBOX, 3 pts at lat=56.0) for city center
  - City-average forecast (TIGHT_CITY_BBOX, ~9 pts) for city summary
  - Region-average forecast (REGION_BBOX, ~45 pts) for region summary
  - Per-grid-point data (MAP_BBOX) for the interactive Leaflet map

Usage:
    python website/forecast_parser.py \
        --input results/live_latest/forecast.pt \
        --output website/static/forecast.json
"""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

# Core city strip: lat=56.0, 3 points (Ветлужанка — центр — Берёзовка)
CORE_CITY_BBOX = (55.875, 56.125, 92.625, 93.375)

# Tight Krasnoyarsk city bbox (~9 points at 0.25 deg)
TIGHT_CITY_BBOX = (55.75, 56.25, 92.75, 93.25)

# Wider region around Krasnoyarsk (~45 points)
REGION_BBOX = (55.5, 56.5, 92.0, 94.0)

# Extended map bbox for per-point display
MAP_BBOX = (54.0, 57.0, 90.0, 96.0)

KRSK_TZ = timezone(timedelta(hours=7))

HPA_TO_MMHG = 0.750062

WIND_DIRECTIONS = [
    ("С", 0), ("СВ", 45), ("В", 90), ("ЮВ", 135),
    ("Ю", 180), ("ЮЗ", 225), ("З", 270), ("СЗ", 315),
]


def wind_dir_text(deg: float) -> str:
    idx = round(deg / 45) % 8
    return WIND_DIRECTIONS[idx][0]


def compute_wind(u: float, v: float) -> tuple[float, float]:
    speed = math.sqrt(u ** 2 + v ** 2)
    direction = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    return round(speed, 2), round(direction, 1)


def precip_info(tp_mm: float, t2m_c: float) -> dict:
    """Precipitation type and intensity from 6h accumulation."""
    if tp_mm < 0.1:
        return {"type": "none", "icon": "", "intensity": "",
                "intensity_text": ""}

    # Type by temperature
    if t2m_c <= 1.0:
        ptype = "snow"
    elif t2m_c <= 3.0:
        ptype = "sleet"
    else:
        ptype = "rain"

    # Intensity by accumulation (mm / 6h)
    if tp_mm < 1.0:
        intensity = "light"
    elif tp_mm < 4.0:
        intensity = "moderate"
    else:
        intensity = "heavy"

    type_names = {"snow": "снег", "sleet": "мокр. снег", "rain": "дождь"}
    intensity_names = {
        "light": "слабый",
        "moderate": "умеренный",
        "heavy": "сильный",
    }

    # Icons
    if ptype == "snow":
        icon = "🌨️" if intensity != "light" else "❄️"
    elif ptype == "sleet":
        icon = "🌨️"
    else:
        icon = "🌧️" if intensity != "light" else "🌦️"

    intensity_text = f"{intensity_names[intensity]} {type_names[ptype]}"
    return {"type": ptype, "icon": icon, "intensity": intensity,
            "intensity_text": intensity_text}


def build_mask(lats, lons, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    return (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max)
    )


def compute_summary(pred_subset, n_steps, last_cycle, vi):
    """Build forecast summary list from per-point prediction.

    pred_subset: (N_points, n_steps, n_vars) — raw per-point data
    vi: tuple of variable indices (t2m, u10, v10, msl, tp, u850, v850)

    Returns list of dicts with mean + min/max for key variables.
    """
    idx_t2m, idx_u10, idx_v10, idx_msl, idx_tp, idx_u850, idx_v850 = vi
    mean_data = pred_subset.mean(axis=0)  # (n_steps, n_vars)
    summary = []

    for step in range(n_steps):
        valid_utc = last_cycle + timedelta(hours=6 * (step + 1))
        valid_krsk = valid_utc.astimezone(KRSK_TZ)

        t2m_c = float(mean_data[step, idx_t2m]) - 273.15
        u10 = float(mean_data[step, idx_u10])
        v10 = float(mean_data[step, idx_v10])
        msl = float(mean_data[step, idx_msl])
        tp = float(mean_data[step, idx_tp])

        if msl > 10000:
            msl /= 100.0

        tp_mm = max(tp * 1000.0, 0.0)
        ws, wd = compute_wind(u10, v10)

        # Wind gust estimate from 850 hPa
        u850 = float(mean_data[step, idx_u850])
        v850 = float(mean_data[step, idx_v850])
        wind_850 = math.sqrt(u850 ** 2 + v850 ** 2)
        wind_gust = ws + 0.6 * max(wind_850 - ws, 0.0)
        wind_gust = max(wind_gust, ws)

        pi = precip_info(tp_mm, t2m_c)
        pressure_mmhg = msl * HPA_TO_MMHG

        # Per-point min/max for temperature and wind
        t2m_all = pred_subset[:, step, idx_t2m].astype(np.float64) - 273.15
        t_min = round(float(t2m_all.min()), 1)
        t_max = round(float(t2m_all.max()), 1)

        # Per-point wind speeds
        u_all = pred_subset[:, step, idx_u10].astype(np.float64)
        v_all = pred_subset[:, step, idx_v10].astype(np.float64)
        ws_all = np.sqrt(u_all ** 2 + v_all ** 2)
        ws_min = round(float(ws_all.min()), 1)
        ws_max = round(float(ws_all.max()), 1)

        # Per-point precip
        tp_all = np.maximum(pred_subset[:, step, idx_tp].astype(np.float64) * 1000.0, 0.0)
        precip_max = round(float(tp_all.max()), 2)

        summary.append({
            "valid_time_utc": valid_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_time_krsk": valid_krsk.isoformat(),
            "horizon_hours": 6 * (step + 1),
            "t2m_celsius": round(t2m_c, 1),
            "t2m_min": t_min,
            "t2m_max": t_max,
            "wind_speed_ms": ws,
            "wind_speed_min": ws_min,
            "wind_speed_max": ws_max,
            "wind_gust_ms": round(wind_gust, 1),
            "wind_direction_deg": wd,
            "wind_direction_text": wind_dir_text(wd),
            "pressure_hpa": round(msl, 1),
            "pressure_mmhg": round(pressure_mmhg, 1),
            "precip_mm": round(tp_mm, 2),
            "precip_max_mm": precip_max,
            "precip_type": pi["type"],
            "precip_type_icon": pi["icon"],
            "precip_intensity": pi["intensity"],
            "precip_intensity_text": pi["intensity_text"],
        })

    return summary


def parse_forecast(input_path: Path) -> dict:
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    prediction = data["prediction_physical"]
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    prediction = prediction.astype(np.float64)

    latitudes = np.asarray(data["latitudes"], dtype=np.float64)
    longitudes = np.asarray(data["longitudes"], dtype=np.float64)
    var_names = data["var_names"]

    idx_t2m = var_names.index("t2m")
    idx_u10 = var_names.index("10u")
    idx_v10 = var_names.index("10v")
    idx_msl = var_names.index("msl")
    idx_tp = var_names.index("tp")
    idx_u850 = var_names.index("u@850")
    idx_v850 = var_names.index("v@850")
    vi = (idx_t2m, idx_u10, idx_v10, idx_msl, idx_tp, idx_u850, idx_v850)

    n_steps = prediction.shape[1]

    # Parse cycles
    cycles_raw = data["cycles"]
    cycles = []
    for c in cycles_raw:
        if isinstance(c, str):
            c_clean = c.replace("+00:00", "Z").rstrip("Z") + "+00:00"
            cycles.append(datetime.fromisoformat(c_clean))
        elif isinstance(c, datetime):
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            cycles.append(c)
        else:
            cycles.append(c)

    last_cycle = cycles[-1]
    if last_cycle.tzinfo is None:
        last_cycle = last_cycle.replace(tzinfo=timezone.utc)

    # --- Core city (3 pts at lat≈56.0, city center strip) ---
    core_mask = build_mask(latitudes, longitudes, CORE_CITY_BBOX)
    core_pred = prediction[core_mask]  # (N_core, n_steps, n_vars)
    summary_core = compute_summary(core_pred, n_steps, last_cycle, vi)

    # --- City summary (tight bbox, ~9 pts) ---
    city_mask = build_mask(latitudes, longitudes, TIGHT_CITY_BBOX)
    city_pred = prediction[city_mask]  # (N_city, n_steps, n_vars)
    summary_city = compute_summary(city_pred, n_steps, last_cycle, vi)

    # --- Region summary (wider bbox, ~45 pts) ---
    region_mask = build_mask(latitudes, longitudes, REGION_BBOX)
    region_pred = prediction[region_mask]  # (N_region, n_steps, n_vars)
    summary_region = compute_summary(region_pred, n_steps, last_cycle, vi)

    # --- Map grid points ---
    map_mask = build_mask(latitudes, longitudes, MAP_BBOX)
    map_indices = np.where(map_mask)[0]
    map_pred = prediction[map_indices]
    map_lats = latitudes[map_indices]
    map_lons = longitudes[map_indices]

    grid_points = []
    for i in range(len(map_indices)):
        steps_data = []
        for step in range(n_steps):
            t2m_c = float(map_pred[i, step, idx_t2m]) - 273.15
            u10 = float(map_pred[i, step, idx_u10])
            v10 = float(map_pred[i, step, idx_v10])
            tp = float(map_pred[i, step, idx_tp])
            msl_v = float(map_pred[i, step, idx_msl])
            tp_mm = max(tp * 1000.0, 0.0)
            ws, wd = compute_wind(u10, v10)

            if msl_v > 10000:
                msl_v /= 100.0
            pr = round(msl_v * HPA_TO_MMHG, 1)

            # Wind gust
            u850 = float(map_pred[i, step, idx_u850])
            v850 = float(map_pred[i, step, idx_v850])
            w850 = math.sqrt(u850 ** 2 + v850 ** 2)
            wg = ws + 0.6 * max(w850 - ws, 0.0)
            wg = max(wg, ws)

            pi = precip_info(tp_mm, t2m_c)

            steps_data.append({
                "t": round(t2m_c, 1),
                "ws": ws,
                "wd": round(wd),
                "wg": round(wg, 1),
                "p": round(tp_mm, 1),
                "pi": pi["icon"],
                "pt": pi["intensity_text"],
                "pr": pr,
            })

        grid_points.append({
            "lat": round(float(map_lats[i]), 4),
            "lon": round(float(map_lons[i]), 4),
            "steps": steps_data,
        })

    now_utc = datetime.now(timezone.utc)

    return {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_cycle": last_cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mos_applied": bool(data.get("learned_mos_applied", False)
                            or data.get("mos_applied", False)),
        "warnings": data.get("warnings", []),
        "n_core_points": int(core_mask.sum()),
        "n_city_points": int(city_mask.sum()),
        "n_region_points": int(region_mask.sum()),
        "n_map_points": len(grid_points),
        "summary_core": summary_core,
        "summary_city": summary_city,
        "summary_region": summary_region,
        "grid_points": grid_points,
    }


def main():
    ap = argparse.ArgumentParser(description="Parse forecast.pt to JSON")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        raise SystemExit(1)

    print(f"Loading {args.input} ...")
    result = parse_forecast(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    size_kb = args.output.stat().st_size / 1024
    print(f"Saved {args.output} ({size_kb:.0f} KB)")
    print(f"  Core: {len(result['summary_core'])} steps, {result['n_core_points']} pts")
    print(f"  City: {len(result['summary_city'])} steps, {result['n_city_points']} pts")
    print(f"  Region: {result['n_region_points']} pts")
    print(f"  Map grid: {result['n_map_points']} pts")
    print(f"  Last cycle: {result['last_cycle']}")
    print(f"  MOS: {result['mos_applied']}")


if __name__ == "__main__":
    main()
