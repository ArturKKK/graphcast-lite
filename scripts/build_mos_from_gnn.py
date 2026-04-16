#!/usr/bin/env python3
"""
Step 2: Retrain MOS using GNN predictions instead of raw ERA5.

Takes the CSV from extract_gnn_at_stations.py (gnn_t2m at 19 stations)
and combines with station observations + ERA5 auxiliary features to train
a new HistGBM that corrects GNN output → station temperature.

Optionally also trains wind MOS from extract_gnn_wind_at_stations.py output
(gnn_ws_ms at stations) to correct GNN wind speed → station wind speed.

Pipeline: ERA5 → GNN → MOS → station forecast
Target t2m: bias_gnn = station_t2m - gnn_t2m_C
Target wind: wind_bias_gnn = station_ws - gnn_ws_ms

Usage:
    python scripts/build_mos_from_gnn.py
    python scripts/build_mos_from_gnn.py --gnn-csv data/temp_train/gnn_predictions_at_stations.csv
    python scripts/build_mos_from_gnn.py --wind-csv data/temp_train/gnn_wind_predictions_at_stations.csv
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, ".")
from scripts.build_learned_mos import (
    STATIONS, FEATURE_COLUMNS, build_features, solar_elevation,
)
from scripts.mos_hparam_sweep import add_extra_features


# Features when using GNN t2m as main predictor
GNN_FEATURE_COLUMNS = [
    "gnn_t2m_C",             # GNN prediction (replaces era5_temperature_2m)
    "era5_temperature_2m",   # still useful as additional signal
    "gnn_era5_diff",         # gnn_t2m - era5_t2m (GNN correction amount)
    "era5_dewpoint_2m",
    "era5_windspeed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "era5_surface_pressure",
    "era5_cloudcover",
    "era5_shortwave_radiation",
    "era5_precipitation",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "solar_elevation",
    "dewpoint_depression",
    "era5_t2m_lag6h",
    "delta_t2m_6h",
    "gnn_t2m_lag6h",         # GNN prediction lag
    "delta_gnn_6h",          # GNN t2m change in 6h
    "station_lat",
    "station_lon",
    "station_elev",
]


# Features for wind MOS using GNN wind as main predictor.
# Only features available BOTH during training AND at runtime (from GNN output).
GNN_WIND_FEATURE_COLUMNS = [
    "gnn_ws_ms",              # GNN wind speed (main predictor)
    "gnn_t2m_C",              # GNN temperature (weather state indicator)
    "wind_dir_sin",
    "wind_dir_cos",
    "era5_surface_pressure",  # at runtime: GNN surface pressure
    "era5_precipitation",     # at runtime: GNN tp
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "solar_elevation",
    "gnn_ws_lag",             # GNN wind speed lag (previous timestep)
    "delta_gnn_ws",           # GNN wind speed change
    "station_lat",
    "station_lon",
    "station_elev",
]


def load_gnn_predictions(csv_path: str) -> pd.DataFrame:
    """Load GNN predictions CSV from extract_gnn_at_stations.py."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["station_usaf"] = df["station_usaf"].astype(str)
    print(f"GNN predictions: {len(df)} rows, "
          f"{df.time.min()} → {df.time.max()}, "
          f"{df.station_usaf.nunique()} stations")
    return df


def load_station_obs(usaf: str, cache_dir: Path,
                     start_year: int = 2010, end_year: int = 2021) -> pd.DataFrame:
    """Load cached station observations."""
    cache_file = cache_dir / f"merged_{usaf}_{start_year}_{end_year}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["time"])
    # Try wider range
    for sy in [2010, 2016]:
        for ey in [2021, 2024]:
            f = cache_dir / f"merged_{usaf}_{sy}_{ey}.csv"
            if f.exists():
                df = pd.read_csv(f, parse_dates=["time"])
                return df
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-csv", default="data/temp_train/gnn_predictions_at_stations.csv")
    ap.add_argument("--wind-csv", default="data/temp_train/gnn_wind_predictions_at_stations.csv",
                    help="CSV with GNN wind predictions (from extract_gnn_wind_at_stations.py)")
    ap.add_argument("--cache-dir", default="data/temp_train/mos_cache")
    ap.add_argument("--output", default="live_runtime_bundle/learned_mos_t2m_gnn.joblib")
    ap.add_argument("--compare-era5", action="store_true", default=True,
                    help="Also train ERA5-only MOS for comparison")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    gnn_df = load_gnn_predictions(args.gnn_csv)

    # ── Merge GNN predictions with station observations ─────────────
    all_parts = []
    for usaf in gnn_df.station_usaf.unique():
        info = STATIONS.get(usaf)
        if info is None:
            continue

        gnn_sub = gnn_df[gnn_df.station_usaf == usaf].copy()
        obs_df = load_station_obs(usaf, cache_dir)
        if obs_df is None:
            print(f"  No cached obs for {usaf} ({info['name']}), skipping")
            continue

        # Merge on time
        merged = gnn_sub.merge(obs_df, on="time", how="inner", suffixes=("_gnn", ""))

        if "station_t2m_C" not in merged.columns and "station_temperature" in merged.columns:
            merged["station_t2m_C"] = merged["station_temperature"]

        if "station_t2m_C" not in merged.columns:
            print(f"  No station_t2m_C for {usaf}, skipping")
            continue

        if "bias" not in merged.columns:
            merged["bias"] = merged["station_t2m_C"] - merged["era5_temperature_2m"]

        # GNN-based target
        merged["bias_gnn"] = merged["station_t2m_C"] - merged["gnn_t2m_C"]
        merged["gnn_era5_diff"] = merged["gnn_t2m_C"] - merged["era5_temperature_2m"]

        # Build features
        merged = build_features(merged, info["lat"], info["lon"], info["elev"])
        merged = add_extra_features(merged)

        # Add GNN-specific features
        merged["gnn_t2m_lag6h"] = merged["gnn_t2m_C"].shift(1)  # 1 step = 6h in this data
        merged["delta_gnn_6h"] = merged["gnn_t2m_C"] - merged["gnn_t2m_lag6h"]

        all_parts.append(merged)
        print(f"  Merged {usaf} ({info['name']:20s}): {len(merged)} rows, "
              f"GNN bias μ={merged.bias_gnn.mean():.3f}°C, "
              f"ERA5 bias μ={merged.bias.mean():.3f}°C")

    if not all_parts:
        raise RuntimeError("No data merged. Check GNN CSV and cached obs.")

    df = pd.concat(all_parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"\nTotal: {len(df)} samples, {df.time.min()} → {df.time.max()}")

    # ── Chronological split ─────────────────────────────────────────
    # GNN data is 2010-2020, station obs 2016-2024, overlap: 2016-2020
    # Split: ≤2018 train, 2019 val, 2020 test
    train_mask = df.time.dt.year <= 2018
    val_mask = df.time.dt.year == 2019
    test_mask = df.time.dt.year == 2020

    print(f"Split: train={train_mask.sum()} | val={val_mask.sum()} | test={test_mask.sum()}")

    # ── Prepare features ────────────────────────────────────────────
    # GNN features
    gnn_feats = [f for f in GNN_FEATURE_COLUMNS if f in df.columns]
    missing = [f for f in GNN_FEATURE_COLUMNS if f not in df.columns]
    if missing:
        print(f"  Missing GNN features (dropped): {missing}")

    df_clean = df.dropna(subset=gnn_feats + ["bias_gnn"])
    train_mask_c = df_clean.time.dt.year <= 2018
    val_mask_c = df_clean.time.dt.year == 2019
    test_mask_c = df_clean.time.dt.year == 2020
    print(f"After dropna: {len(df_clean)} (train={train_mask_c.sum()}, "
          f"val={val_mask_c.sum()}, test={test_mask_c.sum()})")

    X_train = df_clean.loc[train_mask_c, gnn_feats].values
    y_train = df_clean.loc[train_mask_c, "bias_gnn"].values
    X_val = df_clean.loc[val_mask_c, gnn_feats].values
    y_val = df_clean.loc[val_mask_c, "bias_gnn"].values
    X_test = df_clean.loc[test_mask_c, gnn_feats].values
    y_test = df_clean.loc[test_mask_c, "bias_gnn"].values

    # Test reference values
    era5_test = df_clean.loc[test_mask_c, "era5_temperature_2m"].values
    gnn_test = df_clean.loc[test_mask_c, "gnn_t2m_C"].values
    station_test = df_clean.loc[test_mask_c, "station_t2m_C"].values

    # ── Baselines ───────────────────────────────────────────────────
    era5_mae = np.mean(np.abs(station_test - era5_test))
    gnn_mae = np.mean(np.abs(station_test - gnn_test))
    print(f"\nBaselines (test):")
    print(f"  ERA5 → station MAE:  {era5_mae:.4f}°C")
    print(f"  GNN → station MAE:   {gnn_mae:.4f}°C")

    # ── Train GNN-based MOS ─────────────────────────────────────────
    # Use best hyperparams from sweep (will be overridden if sweep results available)
    best_hparams = {
        "max_iter": 1000,
        "max_depth": 10,
        "learning_rate": 0.1,
        "min_samples_leaf": 50,
        "l2_regularization": 0.1,
    }

    # Try to load tuned hparams from sweep results
    sweep_csv = Path("data/temp_train/mos_sweep_results.csv")
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        best_row = sweep_df.loc[sweep_df.val_mae.idxmin()]
        best_hparams = {
            "max_iter": int(best_row.max_iter),
            "max_depth": int(best_row.max_depth),
            "learning_rate": float(best_row.learning_rate),
            "min_samples_leaf": int(best_row.min_samples_leaf),
            "l2_regularization": float(best_row.l2_regularization),
        }
        print(f"  Using sweep-tuned hparams: {best_hparams}")

    print(f"\n{'='*70}")
    print(f"Training GNN-based MOS ({len(gnn_feats)} features)")
    print(f"{'='*70}")

    model_gnn = HistGradientBoostingRegressor(
        **best_hparams,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    t0 = time.time()
    model_gnn.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_bias_gnn = model_gnn.predict(X_test)
    forecast_gnn_mos = gnn_test + pred_bias_gnn  # GNN + MOS correction

    gnn_mos_mae = np.mean(np.abs(station_test - forecast_gnn_mos))
    gnn_mos_rmse = np.sqrt(np.mean((station_test - forecast_gnn_mos)**2))

    print(f"  Trained in {train_time:.1f}s, n_iter={model_gnn.n_iter_}")
    print(f"  GNN+MOS test MAE:  {gnn_mos_mae:.4f}°C")
    print(f"  GNN+MOS test RMSE: {gnn_mos_rmse:.4f}°C")

    # ── Compare: ERA5-based MOS (same split) ────────────────────────
    if args.compare_era5:
        era5_feats = [f for f in FEATURE_COLUMNS if f in df_clean.columns]
        # Also add lag features that sweep found useful
        for extra in ["era5_t2m_lag12h", "delta_t2m_12h",
                       "era5_t2m_lag24h", "delta_t2m_24h"]:
            if extra in df_clean.columns:
                era5_feats.append(extra)

        df_era5 = df_clean.dropna(subset=era5_feats + ["bias"])
        tm = df_era5.time.dt.year <= 2018
        vm = df_era5.time.dt.year == 2019
        tsm = df_era5.time.dt.year == 2020

        model_era5 = HistGradientBoostingRegressor(
            **best_hparams,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        )
        model_era5.fit(
            df_era5.loc[tm, era5_feats].values,
            df_era5.loc[tm, "bias"].values,
        )
        pred_bias_era5 = model_era5.predict(df_era5.loc[tsm, era5_feats].values)
        era5_mos_forecast = df_era5.loc[tsm, "era5_temperature_2m"].values + pred_bias_era5
        era5_mos_station = df_era5.loc[tsm, "station_t2m_C"].values

        era5_mos_mae = np.mean(np.abs(era5_mos_station - era5_mos_forecast))
        era5_mos_rmse = np.sqrt(np.mean((era5_mos_station - era5_mos_forecast)**2))

        print(f"\n{'='*70}")
        print("COMPARISON (test set)")
        print(f"{'='*70}")
        print(f"  {'Method':<25s} {'MAE °C':>8s} {'RMSE °C':>9s}")
        print(f"  {'-'*42}")
        print(f"  {'Raw ERA5':<25s} {era5_mae:8.4f} {'':>9s}")
        print(f"  {'Raw GNN':<25s} {gnn_mae:8.4f} {'':>9s}")
        print(f"  {'ERA5 + MOS':<25s} {era5_mos_mae:8.4f} {era5_mos_rmse:9.4f}")
        print(f"  {'GNN + MOS':<25s} {gnn_mos_mae:8.4f} {gnn_mos_rmse:9.4f}")
        improvement = (era5_mos_mae - gnn_mos_mae) / era5_mos_mae * 100
        print(f"\n  GNN+MOS vs ERA5+MOS: {improvement:+.2f}% MAE")

    # ── Feature importance ──────────────────────────────────────────
    print(f"\nGNN-MOS t2m feature importance (top 10):")
    try:
        imp = model_gnn.feature_importances_
        order = np.argsort(imp)[::-1]
        for rank, i in enumerate(order[:10]):
            print(f"  {rank+1:2d}. {gnn_feats[i]:<28s} {imp[i]:.4f}")
    except AttributeError:
        print("  (feature_importances_ not available)")

    # ── Wind MOS from GNN predictions ───────────────────────────────
    wind_model_gnn = None
    wind_metrics = {}
    wind_csv = Path(args.wind_csv)
    if wind_csv.exists():
        print(f"\n{'='*70}")
        print("WIND MOS: Training on GNN wind errors")
        print(f"{'='*70}")

        wind_df = pd.read_csv(wind_csv, parse_dates=["time"])
        wind_df["station_usaf"] = wind_df["station_usaf"].astype(str)
        print(f"GNN wind predictions: {len(wind_df)} rows, "
              f"{wind_df.time.min()} → {wind_df.time.max()}")

        # Merge with station observations
        wind_parts = []
        for usaf in wind_df.station_usaf.unique():
            info = STATIONS.get(usaf)
            if info is None:
                continue
            w_sub = wind_df[wind_df.station_usaf == usaf].copy()
            obs_df = load_station_obs(usaf, cache_dir)
            if obs_df is None or "station_wind_speed_ms" not in obs_df.columns:
                print(f"  No wind obs for {usaf}, skipping")
                continue

            obs_cols = ["time", "station_wind_speed_ms",
                       "era5_temperature_2m", "era5_dewpoint_2m",
                       "era5_windspeed_10m", "era5_winddirection_10m",
                       "era5_surface_pressure",
                       "era5_cloudcover", "era5_shortwave_radiation",
                       "era5_precipitation"]
            # Only include columns that exist in obs_df
            obs_cols = [c for c in obs_cols if c in obs_df.columns]
            merged_w = w_sub.merge(obs_df[obs_cols], on="time", how="inner")

            if len(merged_w) == 0:
                continue

            # GNN wind bias target
            merged_w["wind_bias_gnn"] = (merged_w["station_wind_speed_ms"]
                                         - merged_w["gnn_ws_ms"])

            # ERA5 wind speed in m/s (cached CSV has km/h)
            if "era5_windspeed_10m" in merged_w.columns:
                merged_w["era5_ws_ms"] = merged_w["era5_windspeed_10m"] / 3.6
            else:
                merged_w["era5_ws_ms"] = merged_w["era5_ws_ms"] if "era5_ws_ms" in merged_w.columns else 0.0

            # GNN - ERA5 wind difference
            merged_w["gnn_wind_era5_diff"] = (merged_w["gnn_ws_ms"]
                                               - merged_w["era5_ws_ms"])

            # Build auxiliary features (time, solar, station info)
            merged_w = build_features(merged_w, info["lat"], info["lon"], info["elev"])

            # Overwrite wind direction with GNN u10/v10 (not ERA5)
            merged_w["wind_dir_sin"] = np.where(
                merged_w["gnn_ws_ms"] > 0.01,
                merged_w["gnn_u10"] / merged_w["gnn_ws_ms"], 0.0)
            merged_w["wind_dir_cos"] = np.where(
                merged_w["gnn_ws_ms"] > 0.01,
                merged_w["gnn_v10"] / merged_w["gnn_ws_ms"], 0.0)

            # GNN wind lags
            merged_w = merged_w.sort_values("time")
            merged_w["gnn_ws_lag"] = merged_w["gnn_ws_ms"].shift(1)
            merged_w["delta_gnn_ws"] = merged_w["gnn_ws_ms"] - merged_w["gnn_ws_lag"]

            wind_parts.append(merged_w)
            bias_mean = merged_w["wind_bias_gnn"].mean()
            print(f"  {usaf} ({info['name']:20s}): {len(merged_w)} rows, "
                  f"wind_bias_gnn={bias_mean:+.3f} m/s")

        if wind_parts:
            df_w = pd.concat(wind_parts, ignore_index=True).sort_values("time")
            # Filter outliers
            df_w = df_w[df_w["wind_bias_gnn"].abs() < 15.0]
            print(f"\nTotal wind samples: {len(df_w)}")
            print(f"Mean wind bias (GNN): {df_w['wind_bias_gnn'].mean():.3f} m/s, "
                  f"Std: {df_w['wind_bias_gnn'].std():.3f}")

            # Determine available features
            w_feats = [f for f in GNN_WIND_FEATURE_COLUMNS if f in df_w.columns]
            missing_w = [f for f in GNN_WIND_FEATURE_COLUMNS if f not in df_w.columns]
            if missing_w:
                print(f"  Missing wind features: {missing_w}")

            df_wc = df_w.dropna(subset=w_feats + ["wind_bias_gnn"])

            # Chronological split (same as t2m)
            wt_train = df_wc[df_wc.time.dt.year <= 2018]
            wt_val = df_wc[df_wc.time.dt.year == 2019]
            wt_test = df_wc[df_wc.time.dt.year == 2020]
            print(f"Split: train={len(wt_train)} | val={len(wt_val)} | test={len(wt_test)}")

            if len(wt_train) > 100 and len(wt_test) > 50:
                Xw_tr = wt_train[w_feats].values
                yw_tr = wt_train["wind_bias_gnn"].values
                Xw_te = wt_test[w_feats].values
                yw_te = wt_test["wind_bias_gnn"].values

                # Baselines
                station_ws_test = wt_test["station_wind_speed_ms"].values
                gnn_ws_test = wt_test["gnn_ws_ms"].values
                era5_ws_test = wt_test["era5_ws_ms"].values

                raw_gnn_mae = np.mean(np.abs(station_ws_test - gnn_ws_test))
                raw_era5_mae = np.mean(np.abs(station_ws_test - era5_ws_test))
                print(f"\nBaselines (test):")
                print(f"  Raw ERA5 → station wind MAE: {raw_era5_mae:.4f} m/s")
                print(f"  Raw GNN  → station wind MAE: {raw_gnn_mae:.4f} m/s")

                wind_model_gnn = HistGradientBoostingRegressor(
                    **best_hparams,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=42,
                )
                t0 = time.time()
                wind_model_gnn.fit(Xw_tr, yw_tr)
                wt = time.time() - t0

                yw_pred = wind_model_gnn.predict(Xw_te)
                gnn_wind_mos = gnn_ws_test + yw_pred
                gnn_wind_mos_mae = np.mean(np.abs(station_ws_test - gnn_wind_mos))
                gnn_wind_mos_rmse = np.sqrt(np.mean((station_ws_test - gnn_wind_mos)**2))

                print(f"\n  Wind MOS trained in {wt:.1f}s, n_iter={wind_model_gnn.n_iter_}")
                print(f"  GNN+Wind MOS MAE:  {gnn_wind_mos_mae:.4f} m/s")
                print(f"  GNN+Wind MOS RMSE: {gnn_wind_mos_rmse:.4f} m/s")
                print(f"  vs Raw GNN:  {(1-gnn_wind_mos_mae/raw_gnn_mae)*100:+.1f}%")
                print(f"  vs Raw ERA5: {(1-gnn_wind_mos_mae/raw_era5_mae)*100:+.1f}%")

                # Feature importance
                print(f"\n  Wind MOS feature importance (top 10):")
                try:
                    wimp = wind_model_gnn.feature_importances_
                    worder = np.argsort(wimp)[::-1]
                    for rank, ii in enumerate(worder[:10]):
                        print(f"    {rank+1:2d}. {w_feats[ii]:<28s} {wimp[ii]:.4f}")
                except AttributeError:
                    pass

                wind_metrics = {
                    "gnn_wind_mos_mae": float(gnn_wind_mos_mae),
                    "gnn_wind_mos_rmse": float(gnn_wind_mos_rmse),
                    "raw_gnn_wind_mae": float(raw_gnn_mae),
                    "raw_era5_wind_mae": float(raw_era5_mae),
                }
            else:
                print("  Not enough wind data for training")
    else:
        print(f"\n  Wind CSV not found: {wind_csv}")
        print("  Run extract_gnn_wind_at_stations.py first to generate it")

    # ── Save model ──────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model_gnn,
        "feature_columns": gnn_feats,
        "target": "bias_gnn",
        "hparams": best_hparams,
        "metrics": {
            "gnn_mos_mae": gnn_mos_mae,
            "gnn_mos_rmse": gnn_mos_rmse,
            "era5_mae": era5_mae,
            "gnn_mae": gnn_mae,
        },
        "train_samples": int(train_mask_c.sum()),
        "n_iter": model_gnn.n_iter_,
    }
    if wind_model_gnn is not None:
        bundle["wind_model"] = wind_model_gnn
        bundle["wind_feature_columns"] = [f for f in GNN_WIND_FEATURE_COLUMNS
                                           if f in df_wc.columns] if 'df_wc' in dir() else []
        bundle["wind_metrics"] = wind_metrics
    joblib.dump(bundle, out_path)
    print(f"\nSaved: {out_path}")
    if wind_model_gnn is not None:
        print(f"  Includes wind MOS (GNN wind MAE: "
              f"{wind_metrics.get('gnn_wind_mos_mae', '?'):.4f} m/s)")


if __name__ == "__main__":
    main()
