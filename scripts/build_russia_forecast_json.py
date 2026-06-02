#!/usr/bin/env python3
"""Build Russia ``russia_forecast.json`` (Krasnoyarsk-compatible schema) from
the live 33f GNN forecast and the per-station postproc-v2 corrections.

Output schema mirrors website/static/forecast.json so the same JS can render
both pages.  Aggregation scopes:

  * ``summary_core``   → 3 capital-zone stations (Moscow/SPb/Novosibirsk)
  * ``summary_city``   → 10 major federal-district hubs
  * ``summary_region`` → all 50 postproc-covered stations

``grid_points`` carries one point per station for the map.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MSK_OFFSET_H = 3  # display timezone for Russia page

CORE_USAFS = {"275185", "260630", "296340"}      # Vnukovo, Pulkovo, Tolmachevo
CITY_USAFS = CORE_USAFS | {
    "284400",  # Koltsovo (Yekaterinburg)
    "275950",  # Kazan
    "287220",  # Ufa
    "371710",  # Sochi
    "349290",  # Krasnodar
    "307100",  # Irkutsk
    "218230",  # Yakutsk
}


def wind_dir_text(deg: float) -> str:
    dirs = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
    idx = int(((deg % 360) + 22.5) / 45) % 8
    return dirs[idx]


def to_mmhg(hpa: float) -> float:
    return round(hpa * 0.7500616, 1)


def fmt_local(utc_iso: str) -> str:
    dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(timezone(timedelta(hours=MSK_OFFSET_H)))
    return local.isoformat()


def fmt_utc_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def precip_classify(mm: float, t_c: float = 5.0) -> tuple[str, str, str, str]:
    """Classify precipitation by 6-hour accumulation (mm) and surface temperature.

    Intensity thresholds for 6h accumulation:
      <0.1   — none
      0.1-0.5 — trace / drizzle  ("следы осадков")
      0.5-2   — light            ("слабый …")
      2-6     — moderate         ("умеренный …")
      6-15    — heavy            ("сильный …")
      >15     — downpour         ("ливень" / "обильный снег")
    """
    if mm < 0.1:
        return ("none", "", "", "")

    # Phase by surface temperature.
    if t_c <= 1.0:
        ptype, noun = "snow", "снег"
    elif t_c <= 3.0:
        ptype, noun = "sleet", "мокрый снег"
    else:
        ptype, noun = "rain", "дождь"

    if mm < 0.5:
        intensity = "trace"
        text = "морось" if ptype == "rain" else f"следы {noun}а" if ptype == "snow" else f"следы {noun}а"
        icon = "🌫️" if ptype != "rain" else "🌦️"
        return (ptype, icon, intensity, text)
    if mm < 2.0:
        intensity, label = "light", "слабый"
        icon = "🌦️" if ptype == "rain" else "🌨️"
    elif mm < 6.0:
        intensity, label = "moderate", "умеренный"
        icon = "🌧️" if ptype == "rain" else "🌨️"
    elif mm < 15.0:
        intensity, label = "heavy", "сильный"
        icon = "🌧️" if ptype == "rain" else "❄️"
    else:
        intensity = "downpour"
        label = "ливневый" if ptype == "rain" else "обильный"
        icon = "⛈️" if ptype == "rain" else "❄️"
    return (ptype, icon, intensity, f"{label} {noun}")


def aggregate_step(per_station_step: list[dict]) -> dict:
    """Aggregate one lead step across a list of station-record dicts.

    Each dict expects keys: t2m_C, u10, v10, wind_ms, msl_hPa, tp_mm, valid_time_utc, lead_h.
    """
    t = [r["t2m_C"] for r in per_station_step]
    ws = [r["wind_ms"] for r in per_station_step]
    us = [r["u10"] for r in per_station_step]
    vs = [r["v10"] for r in per_station_step]
    ps = [r["msl_hPa"] for r in per_station_step]
    pr = [max(0.0, r["tp_mm"]) for r in per_station_step]

    mean_u = statistics.fmean(us)
    mean_v = statistics.fmean(vs)
    mean_dir = (math.degrees(math.atan2(-mean_u, -mean_v)) + 360) % 360
    mean_t = statistics.fmean(t)
    mean_ws = statistics.fmean(ws)
    mean_p = statistics.fmean(ps)
    mean_pr = statistics.fmean(pr)

    ptype, picon, pint, pint_text = precip_classify(mean_pr, mean_t)
    sample = per_station_step[0]
    return {
        "valid_time_utc": sample["valid_time_utc"],
        "valid_time_msk": fmt_local(sample["valid_time_utc"]),
        "horizon_hours": sample["lead_h"],
        "t2m_celsius": round(mean_t, 1),
        "t2m_min": round(min(t), 1),
        "t2m_max": round(max(t), 1),
        "wind_speed_ms": round(mean_ws, 2),
        "wind_speed_min": round(min(ws), 1),
        "wind_speed_max": round(max(ws), 1),
        "wind_gust_ms": round(max(ws) * 1.5, 1),  # proxy gust
        "wind_direction_deg": round(mean_dir, 1),
        "wind_direction_text": wind_dir_text(mean_dir),
        "pressure_hpa": round(mean_p, 1),
        "pressure_mmhg": to_mmhg(mean_p),
        "precip_mm": round(mean_pr, 2),
        "precip_max_mm": round(max(pr), 2),
        "precip_type": ptype,
        "precip_type_icon": picon,
        "precip_intensity": pint,
        "precip_intensity_text": pint_text,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default="results/live_russia_33f_5d/forecast.pt")
    p.add_argument("--postproc-json", default="results/live_russia_33f_5d/station_postproc_v2.json")
    p.add_argument("--out", default="website/static/russia_forecast.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fc = torch.load(args.forecast, map_location="cpu", weights_only=False)
    var_names: list[str] = fc["var_names"]
    pred = fc["prediction_physical"]            # (G, AR, 33)
    lats_all = np.asarray(fc["latitudes"])
    lons_all = np.asarray(fc["longitudes"])     # 0..360
    cycles = fc["cycles"]
    base_cycle = datetime.fromisoformat(cycles[-1])
    if base_cycle.tzinfo is None:
        base_cycle = base_cycle.replace(tzinfo=timezone.utc)
    ar_steps = pred.shape[1]

    tp_idx = var_names.index("tp")
    msl_idx = var_names.index("msl")
    t2m_idx = var_names.index("t2m")
    u10_idx = var_names.index("10u")
    v10_idx = var_names.index("10v")

    pp = json.loads(Path(args.postproc_json).read_text())
    stations = pp["stations"]   # {usaf: {...}}

    # Build per-station per-step records using postproc t2m/wind + GNN pressure/precip
    per_station_records: dict[str, list[dict]] = {}

    for usaf, st in stations.items():
        gidx = st["grid_idx"]
        recs: list[dict] = []
        for s in range(ar_steps):
            valid_dt = base_cycle + timedelta(hours=6 * (s + 1))
            t2m_C = st["postproc"]["t2m_C"][s]
            u10 = st["postproc"]["u10"][s]
            v10 = st["postproc"]["v10"][s]
            ws = st["postproc"]["wind_ms"][s]
            wd = (math.degrees(math.atan2(-u10, -v10)) + 360) % 360
            msl_h = float(pred[gidx, s, msl_idx])
            # ERA5 'tp' is in meters per 6h accumulation step → convert to mm.
            tp_raw_m = float(pred[gidx, s, tp_idx])
            tp_mm = max(0.0, tp_raw_m * 1000.0)
            if tp_mm > 200.0:  # sanity cap (>200 mm/6h is unphysical)
                tp_mm = 0.0
            recs.append({
                "valid_time_utc": fmt_utc_z(valid_dt),
                "lead_h": 6 * (s + 1),
                "t2m_C": t2m_C,
                "u10": u10, "v10": v10, "wind_ms": ws, "wind_dir_deg": wd,
                "msl_hPa": msl_h, "tp_mm": tp_mm,
            })
        per_station_records[usaf] = recs

    # ── Dense Russia grid_points from GNN raw output (for overlay interpolation) ──
    # Russia bbox: lat 41..78, lon 19..180 (0..360 convention).
    # Decimate to ~3000 points for tractable interpolation and small JSON.
    russia_mask = (lats_all >= 41.0) & (lats_all <= 78.0) & (lons_all >= 19.0) & (lons_all <= 180.0)
    russia_idx = np.where(russia_mask)[0]
    stride = max(1, len(russia_idx) // 3000)
    grid_idx = russia_idx[::stride]

    # ── IDW residual interpolation onto grid (postproc → grid) ──
    # For each lead step, compute per-station residuals between postproc and raw GNN
    # (Δt, Δu, Δv) and IDW-interpolate them onto every grid point, then add them on
    # top of the raw GNN field. Wind speed/direction recomputed from corrected (u, v).
    st_lat = np.array([st["lat"] for st in stations.values()], dtype=np.float64)
    st_lon_raw = np.array([st["lon"] for st in stations.values()], dtype=np.float64)
    st_lon = np.where(st_lon_raw <= 180.0, st_lon_raw, st_lon_raw - 360.0)
    st_gidx = np.array([st["grid_idx"] for st in stations.values()], dtype=np.int64)

    grid_lats = lats_all[grid_idx].astype(np.float64)
    grid_lons_raw = lons_all[grid_idx].astype(np.float64)
    grid_lons = np.where(grid_lons_raw <= 180.0, grid_lons_raw, grid_lons_raw - 360.0)

    # Build IDW weights once: 8 nearest stations per grid point, power=2, on great-
    # circle-ish chord (cos(lat) longitudinal scale).
    K_NEIGH = 8
    IDW_POW = 2.0
    coslat_mid = np.cos(np.deg2rad((grid_lats[:, None] + st_lat[None, :]) * 0.5))
    dlat = grid_lats[:, None] - st_lat[None, :]
    dlon = (grid_lons[:, None] - st_lon[None, :]) * coslat_mid
    dist2 = dlat * dlat + dlon * dlon  # (G, S) squared degrees, latitude-scaled
    near_idx = np.argpartition(dist2, K_NEIGH, axis=1)[:, :K_NEIGH]   # (G, K)
    rows = np.arange(dist2.shape[0])[:, None]
    near_d2 = dist2[rows, near_idx]                                   # (G, K)
    weights = 1.0 / np.maximum(near_d2, 1e-8) ** (IDW_POW * 0.5)      # 1/d^p
    weights /= weights.sum(axis=1, keepdims=True)

    # Raw GNN at station grid-nodes for residual computation
    pred_st = pred[st_gidx]                                           # (S, AR, 33)
    st_raw_t = pred_st[..., t2m_idx] - 273.15                         # (S, AR) °C
    st_raw_u = pred_st[..., u10_idx]
    st_raw_v = pred_st[..., v10_idx]

    # Postproc arrays
    st_pp_t = np.array([st["postproc"]["t2m_C"] for st in stations.values()])
    st_pp_u = np.array([st["postproc"]["u10"] for st in stations.values()])
    st_pp_v = np.array([st["postproc"]["v10"] for st in stations.values()])

    # Per-step residuals (S, AR)
    res_t = st_pp_t - st_raw_t
    res_u = st_pp_u - st_raw_u
    res_v = st_pp_v - st_raw_v

    grid_points: list[dict] = []
    grid_idx_np = np.asarray(grid_idx, dtype=np.int64)
    for gi, ni in enumerate(grid_idx_np):
        ni = int(ni)
        lat = float(grid_lats[gi])
        lon_disp = float(grid_lons[gi])
        steps_for_map = []
        w = weights[gi]                                               # (K,)
        nidx = near_idx[gi]                                           # (K,)
        for s in range(ar_steps):
            t_raw = float(pred[ni, s, t2m_idx]) - 273.15
            u_raw = float(pred[ni, s, u10_idx])
            v_raw = float(pred[ni, s, v10_idx])
            dt = float(np.dot(w, res_t[nidx, s]))
            du = float(np.dot(w, res_u[nidx, s]))
            dv = float(np.dot(w, res_v[nidx, s]))
            t_C = t_raw + dt
            u = u_raw + du
            v = v_raw + dv
            ws = math.hypot(u, v)
            wd = (math.degrees(math.atan2(-u, -v)) + 360) % 360
            msl_h = float(pred[ni, s, msl_idx])
            # ERA5 'tp' is in meters per 6h accumulation step → convert to mm.
            tp_raw_m = float(pred[ni, s, tp_idx])
            tp_mm = max(0.0, tp_raw_m * 1000.0)
            if tp_mm > 200.0:
                tp_mm = 0.0
            ptype, picon, pint, pint_text = precip_classify(tp_mm)
            steps_for_map.append({
                "t":  round(t_C, 1),
                "ws": round(ws, 2),
                "wd": int(round(wd)),
                "wg": round(ws * 1.5, 1),
                "p":  round(tp_mm, 2),
                "pi": picon,
                "pt": pint_text,
                "pr": to_mmhg(msl_h),
            })
        grid_points.append({
            "lat": lat,
            "lon": lon_disp,
            "steps": steps_for_map,
        })

    # Stations: separate list with names for clickable markers
    stations_list = []
    for usaf, st in stations.items():
        lon_disp = st["lon"] if st["lon"] <= 180 else st["lon"] - 360
        recs = per_station_records[usaf]
        st_steps = []
        for r in recs:
            ptype, picon, pint, pint_text = precip_classify(r["tp_mm"])
            st_steps.append({
                "t":  round(r["t2m_C"], 1),
                "ws": round(r["wind_ms"], 2),
                "wd": int(round(r["wind_dir_deg"])),
                "wg": round(r["wind_ms"] * 1.5, 1),
                "p":  round(r["tp_mm"], 2),
                "pi": picon,
                "pt": pint_text,
                "pr": to_mmhg(r["msl_hPa"]),
            })
        stations_list.append({
            "usaf": usaf, "name": st["name"],
            "lat": st["lat"], "lon": lon_disp,
            "steps": st_steps,
        })

    # Aggregate per scope
    def aggregate_scope(usaf_set: set[str] | None) -> list[dict]:
        steps = []
        for s in range(ar_steps):
            cohort = []
            for usaf, recs in per_station_records.items():
                if usaf_set is not None and usaf not in usaf_set:
                    continue
                cohort.append(recs[s])
            if not cohort:
                continue
            steps.append(aggregate_step(cohort))
        return steps

    summary_core = aggregate_scope(CORE_USAFS)
    summary_city = aggregate_scope(CITY_USAFS)
    summary_region = aggregate_scope(None)

    out_payload = {
        "generated_at": fmt_utc_z(datetime.now(timezone.utc)),
        "last_cycle": fmt_utc_z(base_cycle),
        "mos_applied": True,                # neural postproc v2 applied
        "neural_postproc": "v2",
        "postproc_applied_to_grid": True,   # IDW residual interpolation on grid_points
        "postproc_grid_method": "IDW k=8 power=2 (Δt, Δu, Δv from 50 stations)",
        "warnings": [],
        "n_core_points": len(CORE_USAFS & set(per_station_records.keys())),
        "n_city_points": len(CITY_USAFS & set(per_station_records.keys())),
        "n_region_points": len(per_station_records),
        "n_map_points": len(grid_points),
        "n_stations": len(stations_list),
        "summary_core": summary_core,
        "summary_city": summary_city,
        "summary_region": summary_region,
        "grid_points": grid_points,
        "stations": stations_list,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False))
    print(f"[russia] wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"  steps: {ar_steps}  core={out_payload['n_core_points']} "
          f"city={out_payload['n_city_points']} region={out_payload['n_region_points']}")


if __name__ == "__main__":
    main()
