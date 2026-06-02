#!/usr/bin/env python3
"""Apply neural_postproc_v2 (StationLeadAwareResidualMLP) to a Russia
multires-33f live forecast.

Reads:
  - results/live_russia_33f_5d/forecast.pt   (prediction_physical: G x AR x 33)
  - website/static/russia_stations.json      (50 stations with lat/lon/elev)
  - experiments/neural_postproc_v2/{config.json, scalers.json,
    station_to_idx.json, best_model.pth}

Writes:
  - results/live_russia_33f_5d/station_postproc_v2.json   one entry per station:
        { usaf: {
              "name": ..., "lat": ..., "lon": ..., "elev": ...,
              "lead_h": [6, 12, ..., 120],
              "valid_time_utc": [...],
              "gnn":     {"t2m_C": [...], "u10": [...], "v10": [...], ...},
              "postproc":{"t2m_C": [...], "u10": [...], "v10": [...], "wind_ms": [...]},
          } }
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.postprocessing.neural.models import StationLeadAwareResidualMLP


# --- Feature order must match training (see experiments/neural_postproc_v2/config.json)
FEATURE_COLS = [
    "gnn_t2m", "gnn_u10", "gnn_v10", "gnn_msl", "gnn_sp",
    "gnn_t850", "gnn_t500", "gnn_q850", "gnn_z500",
    "gnn_u850", "gnn_v850", "gnn_u1000", "gnn_v1000",
    "lapse_t850_1000", "dewpoint_depression", "solar_zen",
    "lat", "lon", "elev", "z_surf", "lsm",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "lead_norm",
]

# Map FEATURE_COL -> var_name in GNN var_order
GNN_FROM = {
    "gnn_t2m": "t2m", "gnn_u10": "10u", "gnn_v10": "10v",
    "gnn_msl": "msl", "gnn_sp":  "sp",
    "gnn_t850": "t@850", "gnn_t500": "t@500",
    "gnn_q850": "q@850", "gnn_z500": "z@500",
    "gnn_u850": "u@850", "gnn_v850": "v@850",
    "gnn_u1000": "u@1000", "gnn_v1000": "v@1000",
}


def solar_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Spencer 1971 — same formula used in build_corpus.py."""
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
    time_offset = eq_time + 4.0 * lon_deg
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    ha = math.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    elev = math.asin(
        math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    )
    return math.degrees(elev)


def dewpoint_depression_K(t2m_K: float, q_kg_kg: float, sp_Pa: float) -> float:
    if not np.isfinite(q_kg_kg) or q_kg_kg <= 0 or not np.isfinite(sp_Pa) or sp_Pa <= 0:
        return float("nan")
    e = q_kg_kg * sp_Pa / (0.622 + 0.378 * q_kg_kg)
    e_hPa = e / 100.0
    if e_hPa <= 0:
        return float("nan")
    ln_term = math.log(max(e_hPa / 6.112, 1e-6))
    td_C = 243.5 * ln_term / (17.67 - ln_term)
    return float(t2m_K - (td_C + 273.15))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default="results/live_russia_33f_5d/forecast.pt")
    p.add_argument("--stations", default="website/static/russia_stations.json")
    p.add_argument("--postproc-dir", default="experiments/neural_postproc_v2")
    p.add_argument("--out", default="results/live_russia_33f_5d/station_postproc_v2.json")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fc = torch.load(args.forecast, map_location="cpu", weights_only=False)
    lats = fc["latitudes"]
    lons = fc["longitudes"]
    var_names: list[str] = fc["var_names"]
    pred = fc["prediction_physical"]            # (G, AR, 33)
    cycles = fc["cycles"]
    base_cycle = datetime.fromisoformat(cycles[-1])
    if base_cycle.tzinfo is None:
        base_cycle = base_cycle.replace(tzinfo=timezone.utc)
    ar_steps = pred.shape[1]
    print(f"[postproc] AR steps = {ar_steps}, base cycle = {base_cycle.isoformat()}")

    stations = json.loads(Path(args.stations).read_text())
    if isinstance(stations, dict):
        stations = list(stations.values())

    postproc_dir = Path(args.postproc_dir)
    cfg = json.loads((postproc_dir / "config.json").read_text())
    scalers = json.loads((postproc_dir / "scalers.json").read_text())
    station_to_idx: dict[str, int] = json.loads(
        (postproc_dir / "station_to_idx.json").read_text()
    )

    # Sanity: feature_cols match
    cfg_features = cfg.get("feature_cols") or FEATURE_COLS
    if list(cfg_features) != FEATURE_COLS:
        print("[warn] config.json feature_cols differ from this script's FEATURE_COLS")
        print(" config:", cfg_features)
        print(" script:", FEATURE_COLS)

    device = torch.device(args.device)
    model = StationLeadAwareResidualMLP(
        feature_dim=len(FEATURE_COLS),
        num_stations=len(station_to_idx),
        station_emb_dim=cfg.get("station_emb_dim", 16),
        hidden=cfg.get("hidden", [128, 128]),
        dropout=cfg.get("dropout", 0.1),
        probabilistic=cfg.get("probabilistic", False),
        film_hidden=cfg.get("film_hidden", 32),
    ).to(device)
    state = torch.load(postproc_dir / "best_model.pth", map_location=device)
    if isinstance(state, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[postproc] loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # Map each station to nearest GNN node (longitudes are 0..360 in coords.npz).
    out_data: dict[str, dict] = {}
    matched = 0
    skipped = 0
    for st in stations:
        usaf = str(st["usaf"])
        if usaf not in station_to_idx:
            skipped += 1
            continue
        matched += 1
        lat = float(st["lat"])
        lon = float(st["lon"]) % 360.0
        d2 = (lats - lat) ** 2 + (lons - lon) ** 2
        gidx = int(np.argmin(d2))
        sidx = station_to_idx[usaf]

        # ---- build features for each lead ----
        feats = np.zeros((ar_steps, len(FEATURE_COLS)), dtype=np.float32)
        gnn_targets = {
            "t2m": np.zeros(ar_steps, dtype=np.float32),
            "u10": np.zeros(ar_steps, dtype=np.float32),
            "v10": np.zeros(ar_steps, dtype=np.float32),
        }
        valid_times: list[str] = []
        lead_hours: list[int] = []
        gnn_raw_dump: dict[str, list[float]] = {k: [] for k in GNN_FROM}

        # static z_surf, lsm — same at all steps (and we pin them in inference)
        z_surf_node = float(pred[gidx, 0, var_names.index("z_surf")])
        lsm_node = float(pred[gidx, 0, var_names.index("lsm")])
        # z_surf saved channel is in metres (training side divided by 9.80665);
        # the runtime bundle's z_surf is already in metres (multires data).
        # build_corpus did `df["z_surf"] = df["era5_z_surf"] / 9.80665` — i.e. the
        # corpus z_surf is in *metres*. Our pred z_surf is also in metres.
        z_surf_m = z_surf_node

        for s in range(ar_steps):
            valid_dt = base_cycle + timedelta(hours=6 * (s + 1))
            valid_times.append(valid_dt.isoformat())
            lead_h = 6 * (s + 1)
            lead_hours.append(lead_h)

            row = {}
            for fname, gname in GNN_FROM.items():
                val = float(pred[gidx, s, var_names.index(gname)])
                row[fname] = val
                gnn_raw_dump[fname].append(val)

            row["lapse_t850_1000"] = row["gnn_t850"] - float(
                pred[gidx, s, var_names.index("t@1000")]
            )
            q1000 = float(pred[gidx, s, var_names.index("q@1000")])
            # gnn_sp is in hPa; dewpoint expects Pa
            row["dewpoint_depression"] = dewpoint_depression_K(
                row["gnn_t2m"], q1000, row["gnn_sp"] * 100.0
            )
            row["solar_zen"] = 90.0 - solar_elevation(lat, float(st["lon"]), valid_dt)
            row["lat"] = lat
            row["lon"] = float(st["lon"])
            row["elev"] = float(st.get("elev", 0.0))
            row["z_surf"] = z_surf_m
            row["lsm"] = lsm_node
            h = valid_dt.hour + valid_dt.minute / 60.0
            doy = valid_dt.timetuple().tm_yday
            row["sin_hour"] = math.sin(2 * math.pi * h / 24.0)
            row["cos_hour"] = math.cos(2 * math.pi * h / 24.0)
            row["sin_doy"] = math.sin(2 * math.pi * doy / 365.25)
            row["cos_doy"] = math.cos(2 * math.pi * doy / 365.25)
            row["lead_norm"] = lead_h / 120.0

            # normalize per scalers
            for i, col in enumerate(FEATURE_COLS):
                mu, sigma = scalers[col]
                if not np.isfinite(row[col]):
                    row[col] = 0.0
                feats[s, i] = (row[col] - mu) / max(sigma, 1e-6)

            gnn_targets["t2m"][s] = row["gnn_t2m"]
            gnn_targets["u10"][s] = row["gnn_u10"]
            gnn_targets["v10"][s] = row["gnn_v10"]

        # forward
        with torch.no_grad():
            x = torch.from_numpy(feats).to(device)
            station_idx_t = torch.full((ar_steps,), sidx, dtype=torch.long, device=device)
            lead_norm_t = torch.from_numpy(
                np.array([h / 120.0 for h in lead_hours], dtype=np.float32)
            ).to(device)
            gnn_t = {k: torch.from_numpy(v).to(device) for k, v in gnn_targets.items()}
            out = model(x, station_idx=station_idx_t, lead_norm=lead_norm_t, gnn_targets=gnn_t)

        t2m_corr = out["t2m"].cpu().numpy()
        u10_corr = out["u10"].cpu().numpy()
        v10_corr = out["v10"].cpu().numpy()
        wind_corr = np.sqrt(u10_corr ** 2 + v10_corr ** 2)

        out_data[usaf] = {
            "name": st.get("name", ""),
            "lat": lat,
            "lon": float(st["lon"]),
            "elev": float(st.get("elev", 0.0)),
            "grid_idx": gidx,
            "grid_lat": float(lats[gidx]),
            "grid_lon": float(lons[gidx]),
            "lead_h": lead_hours,
            "valid_time_utc": valid_times,
            "gnn": {
                "t2m_C": [v - 273.15 for v in gnn_raw_dump["gnn_t2m"]],
                "u10": gnn_raw_dump["gnn_u10"],
                "v10": gnn_raw_dump["gnn_v10"],
                "wind_ms": [float(np.sqrt(u * u + v * v)) for u, v in zip(gnn_raw_dump["gnn_u10"], gnn_raw_dump["gnn_v10"])],
                "msl_hPa": gnn_raw_dump["gnn_msl"],
                "sp_hPa": gnn_raw_dump["gnn_sp"],
            },
            "postproc": {
                "t2m_C": [float(v - 273.15) for v in t2m_corr],
                "u10": [float(v) for v in u10_corr],
                "v10": [float(v) for v in v10_corr],
                "wind_ms": [float(v) for v in wind_corr],
            },
        }

    print(f"[postproc] applied to {matched} stations, skipped {skipped} (no entry in station_to_idx)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_cycle": base_cycle.isoformat(),
        "ar_steps": ar_steps,
        "stations": out_data,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"[postproc] wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
