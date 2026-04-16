#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import torch

CITY_NAME = "Krasnoyarsk"
CITY_LAT = 56.01
CITY_LON = 92.87
CITY_UTC_OFFSET_HOURS = 7
ROI_BBOX = (55.5, 56.5, 92.0, 94.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live forecast bundle with Open-Meteo hourly weather")
    parser.add_argument("--forecast", required=True, help="Path to forecast.pt produced by live_gdas_forecast.py")
    parser.add_argument(
        "--mode",
        choices=["roi-mean", "nearest-point"],
        default="roi-mean",
        help="Compare either ROI mean or the nearest forecast grid point to the target coordinates",
    )
    parser.add_argument("--lat", type=float, default=CITY_LAT, help="Target latitude for nearest-point mode / Open-Meteo request")
    parser.add_argument("--lon", type=float, default=CITY_LON, help="Target longitude for nearest-point mode / Open-Meteo request")
    parser.add_argument("--name", default=CITY_NAME, help="Display name for the target point")
    parser.add_argument("--out-md", default=None, help="Optional markdown output path")
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path")
    return parser.parse_args()


def build_city_mask(latitudes, longitudes):
    lat_min, lat_max, lon_min, lon_max = ROI_BBOX
    return (
        (latitudes >= lat_min)
        & (latitudes <= lat_max)
        & (longitudes >= lon_min)
        & (longitudes <= lon_max)
    )


def fetch_openmeteo_hourly(target_lat: float, target_lon: float) -> dict[str, dict[str, float]]:
    params = {
        "latitude": target_lat,
        "longitude": target_lon,
        "hourly": "temperature_2m,wind_speed_10m,pressure_msl",
        "past_days": 3,
        "forecast_days": 7,
        "timezone": "GMT",
    }
    url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        payload = json.load(response)

    hourly = payload["hourly"]
    out: dict[str, dict[str, float]] = {}
    for idx, time_str in enumerate(hourly["time"]):
        out[f"{time_str}:00+00:00"] = {
            "temperature_2m_c": float(hourly["temperature_2m"][idx]),
            "wind_speed_10m_ms": float(hourly["wind_speed_10m"][idx]) / 3.6,
            "pressure_msl_hpa": float(hourly["pressure_msl"][idx]),
        }
    return out


def normalize_lon(lon: float) -> float:
    return lon % 360.0


def nearest_point_index(latitudes, longitudes, target_lat: float, target_lon: float) -> int:
    target_lon_360 = normalize_lon(target_lon)
    lon_diff = ((longitudes - target_lon_360 + 180.0) % 360.0) - 180.0
    dist2 = (latitudes - target_lat) ** 2 + lon_diff ** 2
    return int(dist2.argmin())


def build_rows(
    forecast_path: Path,
    mode: str,
    target_lat: float,
    target_lon: float,
) -> tuple[list[dict[str, str | float]], dict[str, str | float]]:
    payload = torch.load(forecast_path, map_location="cpu")
    cycles = [datetime.fromisoformat(cycle) for cycle in payload["cycles"]]
    base_cycle = cycles[-1]
    var_names = payload["var_names"]
    prediction = payload["prediction_physical"]
    latitudes = payload["latitudes"]
    longitudes = payload["longitudes"]
    city_mask = build_city_mask(latitudes, longitudes)
    point_idx = nearest_point_index(latitudes, longitudes, target_lat, target_lon)

    idx_t2m = var_names.index("t2m")
    idx_u10 = var_names.index("10u")
    idx_v10 = var_names.index("10v")
    idx_msl = var_names.index("msl")

    openmeteo = fetch_openmeteo_hourly(target_lat, target_lon)
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    point_meta: dict[str, str | float] = {
        "mode": mode,
        "target_lat": round(float(target_lat), 4),
        "target_lon": round(float(target_lon), 4),
        "nearest_lat": round(float(latitudes[point_idx]), 4),
        "nearest_lon": round(float(longitudes[point_idx]), 4),
        "nearest_index": point_idx,
    }

    rows: list[dict[str, str | float]] = []
    for step_idx in range(prediction.shape[1]):
        valid_utc = base_cycle + timedelta(hours=6 * (step_idx + 1))
        if mode == "nearest-point":
            forecast_t2m_c = float(prediction[point_idx, step_idx, idx_t2m] - 273.15)
            forecast_u10 = float(prediction[point_idx, step_idx, idx_u10])
            forecast_v10 = float(prediction[point_idx, step_idx, idx_v10])
            forecast_msl = float(prediction[point_idx, step_idx, idx_msl])
        else:
            forecast_t2m_c = float(prediction[:, step_idx, idx_t2m][city_mask].mean() - 273.15)
            forecast_u10 = float(prediction[:, step_idx, idx_u10][city_mask].mean())
            forecast_v10 = float(prediction[:, step_idx, idx_v10][city_mask].mean())
            forecast_msl = float(prediction[:, step_idx, idx_msl][city_mask].mean())
        forecast_wind = math.sqrt(forecast_u10 * forecast_u10 + forecast_v10 * forecast_v10)

        key = valid_utc.isoformat()
        observed = openmeteo.get(key, {})

        row: dict[str, str | float] = {
            "horizon": f"+{6 * (step_idx + 1)}h",
            "valid_utc": key,
            "same_in_krasnoyarsk": (valid_utc + timedelta(hours=CITY_UTC_OFFSET_HOURS)).strftime("%Y-%m-%d %H:%M"),
            "status": "observed/recent" if valid_utc <= now_utc else "open-meteo forecast",
            "forecast_t2m_c": round(forecast_t2m_c, 2),
            "forecast_wind_ms": round(forecast_wind, 2),
            "forecast_msl_hpa": round(forecast_msl, 2),
        }

        if observed:
            row["openmeteo_t2m_c"] = round(float(observed["temperature_2m_c"]), 2)
            row["openmeteo_wind_ms"] = round(float(observed["wind_speed_10m_ms"]), 2)
            row["openmeteo_msl_hpa"] = round(float(observed["pressure_msl_hpa"]), 2)
            row["temp_error_c"] = round(float(row["forecast_t2m_c"]) - float(row["openmeteo_t2m_c"]), 2)
            row["wind_error_ms"] = round(float(row["forecast_wind_ms"]) - float(row["openmeteo_wind_ms"]), 2)
            row["msl_error_hpa"] = round(float(row["forecast_msl_hpa"]) - float(row["openmeteo_msl_hpa"]), 2)
        rows.append(row)
    return rows, point_meta


def write_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "horizon",
        "valid_utc",
        "same_in_krasnoyarsk",
        "status",
        "forecast_t2m_c",
        "openmeteo_t2m_c",
        "temp_error_c",
        "forecast_wind_ms",
        "openmeteo_wind_ms",
        "wind_error_ms",
        "forecast_msl_hpa",
        "openmeteo_msl_hpa",
        "msl_error_hpa",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    rows: list[dict[str, str | float]],
    out_path: Path,
    forecast_path: Path,
    point_name: str,
    point_meta: dict[str, str | float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(point_meta["mode"])
    lines = [
        "# 3-Day Live Forecast vs Open-Meteo",
        "",
        f"Forecast bundle: {forecast_path}",
        f"Reference point: {point_name} ({float(point_meta['target_lat']):.2f}, {float(point_meta['target_lon']):.2f})",
        "Open-Meteo hourly values are fetched in UTC and matched to the forecast by the same UTC timestamp.",
        "The Krasnoyarsk time column is only a human-readable conversion of that same instant to KRAT (UTC+7).",
        (
            "Forecast values are ROI means over [55.5..56.5N, 92..94E]; Open-Meteo values are point-based city hourly weather."
            if mode == "roi-mean"
            else f"Forecast values use the nearest forecast grid point: idx={int(point_meta['nearest_index'])}, lat={float(point_meta['nearest_lat']):.4f}, lon={float(point_meta['nearest_lon']):.4f}."
        ),
        "",
        "| Horizon | Valid UTC | Same Instant In KRAT | Status | Forecast T | Open-Meteo T | ΔT | Forecast Wind | Open-Meteo Wind | ΔWind | Forecast MSL | Open-Meteo MSL | ΔMSL |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {horizon} | {valid_utc} | {same_in_krasnoyarsk} | {status} | {forecast_t2m_c:.2f} | {openmeteo_t2m_c:.2f} | {temp_error_c:.2f} | {forecast_wind_ms:.2f} | {openmeteo_wind_ms:.2f} | {wind_error_ms:.2f} | {forecast_msl_hpa:.2f} | {openmeteo_msl_hpa:.2f} | {msl_error_hpa:.2f} |".format(
                horizon=row.get("horizon", ""),
                valid_utc=row.get("valid_utc", ""),
                same_in_krasnoyarsk=row.get("same_in_krasnoyarsk", ""),
                status=row.get("status", ""),
                forecast_t2m_c=float(row.get("forecast_t2m_c", float("nan"))),
                openmeteo_t2m_c=float(row.get("openmeteo_t2m_c", float("nan"))),
                temp_error_c=float(row.get("temp_error_c", float("nan"))),
                forecast_wind_ms=float(row.get("forecast_wind_ms", float("nan"))),
                openmeteo_wind_ms=float(row.get("openmeteo_wind_ms", float("nan"))),
                wind_error_ms=float(row.get("wind_error_ms", float("nan"))),
                forecast_msl_hpa=float(row.get("forecast_msl_hpa", float("nan"))),
                openmeteo_msl_hpa=float(row.get("openmeteo_msl_hpa", float("nan"))),
                msl_error_hpa=float(row.get("msl_error_hpa", float("nan"))),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    forecast_path = Path(args.forecast)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing forecast bundle: {forecast_path}")

    rows, point_meta = build_rows(forecast_path, args.mode, args.lat, args.lon)
    default_stem = "openmeteo_comparison" if args.mode == "roi-mean" else "openmeteo_point_comparison"
    out_md = Path(args.out_md) if args.out_md else forecast_path.with_name(f"{default_stem}.md")
    out_csv = Path(args.out_csv) if args.out_csv else forecast_path.with_name(f"{default_stem}.csv")

    write_markdown(rows, out_md, forecast_path, args.name, point_meta)
    write_csv(rows, out_csv)

    print(f"[compare] wrote {out_md}")
    print(f"[compare] wrote {out_csv}")


if __name__ == "__main__":
    main()