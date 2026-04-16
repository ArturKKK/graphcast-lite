#!/usr/bin/env python3
"""
Sweep HistGradientBoostingRegressor hyperparameters for MOS.

Uses cached data from build_learned_mos.py (data/temp_train/mos_cache/).
Runs locally — no GPU needed.

Usage:
    python scripts/mos_hparam_sweep.py
    python scripts/mos_hparam_sweep.py --stations 284935 294670 295810
"""

import argparse
import itertools
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Import stations & features from build_learned_mos ───────────────
sys.path.insert(0, ".")
from scripts.build_learned_mos import (
    STATIONS, FEATURE_COLUMNS, build_features, solar_elevation,
)

# ── Extra features to try ───────────────────────────────────────────
EXTRA_FEATURES = [
    "era5_t2m_lag12h",
    "delta_t2m_12h",
    "era5_t2m_lag24h",
    "delta_t2m_24h",
    "temp_x_solar",       # interaction: t2m × solar_elevation
    "pressure_anomaly",   # surface_pressure - 1013.25
    "month_sin",
    "month_cos",
]


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra candidate features."""
    out = df.copy()
    out["era5_t2m_lag12h"] = out["era5_temperature_2m"].shift(12)
    out["delta_t2m_12h"] = out["era5_temperature_2m"] - out["era5_t2m_lag12h"]
    out["era5_t2m_lag24h"] = out["era5_temperature_2m"].shift(24)
    out["delta_t2m_24h"] = out["era5_temperature_2m"] - out["era5_t2m_lag24h"]
    out["temp_x_solar"] = out["era5_temperature_2m"] * out["solar_elevation"]
    out["pressure_anomaly"] = out["era5_surface_pressure"] - 1013.25
    month = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


# ── Hyperparameter grid ─────────────────────────────────────────────
HPARAM_GRID = {
    "max_iter":          [500, 1000, 2000],
    "max_depth":         [6, 8, 10, 12],
    "learning_rate":     [0.02, 0.05, 0.1],
    "min_samples_leaf":  [10, 20, 50],
    "l2_regularization": [0.01, 0.1, 1.0],
}

FEATURE_SETS = {
    "base20": FEATURE_COLUMNS,
    "base20+lags": FEATURE_COLUMNS + [
        "era5_t2m_lag12h", "delta_t2m_12h",
        "era5_t2m_lag24h", "delta_t2m_24h",
    ],
    "base20+lags+interact": FEATURE_COLUMNS + EXTRA_FEATURES,
}


def load_cached_data(stations: list[str], cache_dir: Path,
                     start_year: int = 2016, end_year: int = 2024) -> pd.DataFrame:
    """Load cached merged CSVs from build_learned_mos.py."""
    all_parts = []
    for usaf in stations:
        info = STATIONS.get(usaf)
        if info is None:
            print(f"  Unknown station {usaf}, skipping")
            continue
        cache_file = cache_dir / f"merged_{usaf}_{start_year}_{end_year}.csv"
        if not cache_file.exists():
            print(f"  No cache for {usaf} at {cache_file} — run build_learned_mos.py first")
            continue
        df = pd.read_csv(cache_file, parse_dates=["time"])
        if "bias" not in df.columns:
            df["bias"] = df["station_t2m_C"] - df["era5_temperature_2m"]
        df = build_features(df, info["lat"], info["lon"], info["elev"])
        df = add_extra_features(df)
        all_parts.append(df)
        print(f"  Loaded {usaf} ({info['name']}): {len(df)} rows")

    if not all_parts:
        raise RuntimeError("No cached data found. Run build_learned_mos.py first.")
    return pd.concat(all_parts, ignore_index=True).sort_values("time").reset_index(drop=True)


def evaluate_config(X_train, y_train, X_val, y_val, X_test, y_test,
                    hparams: dict) -> dict:
    """Train and evaluate a single configuration."""
    model = HistGradientBoostingRegressor(
        max_iter=hparams["max_iter"],
        max_depth=hparams["max_depth"],
        learning_rate=hparams["learning_rate"],
        min_samples_leaf=hparams["min_samples_leaf"],
        l2_regularization=hparams["l2_regularization"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    return {
        "val_mae": mean_absolute_error(y_val, pred_val),
        "val_rmse": np.sqrt(mean_squared_error(y_val, pred_val)),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
        "n_iter": model.n_iter_,
        "train_time": train_time,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="MOS hyperparameter sweep")
    parser.add_argument("--stations", nargs="*",
                        default=list(STATIONS.keys()),
                        help="USAF station IDs (default: all 19)")
    parser.add_argument("--cache-dir", default="data/temp_train/mos_cache")
    parser.add_argument("--quick", action="store_true",
                        help="Run smaller grid for quick test")
    parser.add_argument("--save-best", default=None,
                        help="Save best model to this path")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # ── Load data ───────────────────────────────────────────────────
    print("Loading cached data...")
    df = load_cached_data(args.stations, cache_dir)
    print(f"\nTotal: {len(df)} samples, {df['time'].min()} → {df['time'].max()}")

    # Drop NaN for all possible features
    all_feats = list(set(FEATURE_COLUMNS + EXTRA_FEATURES))
    df = df.dropna(subset=all_feats + ["bias"])
    print(f"After dropna: {len(df)}")

    # ── Split ───────────────────────────────────────────────────────
    train_mask = df["time"].dt.year <= 2022
    val_mask = df["time"].dt.year == 2023
    test_mask = df["time"].dt.year == 2024

    y_train = df.loc[train_mask, "bias"].values
    y_val = df.loc[val_mask, "bias"].values
    y_test = df.loc[test_mask, "bias"].values

    print(f"Split: train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")

    # Baseline
    base_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
    print(f"Baseline (no correction) test MAE: {base_mae:.3f}°C")

    # ── Quick grid ──────────────────────────────────────────────────
    if args.quick:
        grid = {
            "max_iter":          [500, 1000],
            "max_depth":         [8, 10],
            "learning_rate":     [0.05, 0.1],
            "min_samples_leaf":  [20],
            "l2_regularization": [0.1],
        }
    else:
        grid = HPARAM_GRID

    # ── Phase 1: Feature set comparison with default hparams ────────
    print(f"\n{'='*70}")
    print("Phase 1: Feature set comparison (default hparams)")
    print(f"{'='*70}")

    default_hp = {
        "max_iter": 500, "max_depth": 8, "learning_rate": 0.05,
        "min_samples_leaf": 20, "l2_regularization": 0.1,
    }
    best_fset_name = None
    best_fset_val = float("inf")

    for fset_name, fset_cols in FEATURE_SETS.items():
        X_tr = df.loc[train_mask, fset_cols].values
        X_v = df.loc[val_mask, fset_cols].values
        X_te = df.loc[test_mask, fset_cols].values

        res = evaluate_config(X_tr, y_train, X_v, y_val, X_te, y_test, default_hp)
        print(f"  {fset_name:<30s}  val_MAE={res['val_mae']:.4f}  "
              f"test_MAE={res['test_mae']:.4f}  test_RMSE={res['test_rmse']:.4f}  "
              f"iters={res['n_iter']}  {res['train_time']:.1f}s")

        if res["val_mae"] < best_fset_val:
            best_fset_val = res["val_mae"]
            best_fset_name = fset_name

    print(f"\n  → Best feature set: {best_fset_name} (val MAE={best_fset_val:.4f})")
    best_features = FEATURE_SETS[best_fset_name]

    X_train = df.loc[train_mask, best_features].values
    X_val = df.loc[val_mask, best_features].values
    X_test = df.loc[test_mask, best_features].values

    # ── Phase 2: Hyperparameter sweep ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"Phase 2: Hyperparameter sweep ({best_fset_name})")
    print(f"{'='*70}")

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    print(f"Total configurations: {len(combos)}")

    results = []
    best_val_mae = float("inf")
    best_config = None
    best_model = None

    for idx, combo in enumerate(combos):
        hp = dict(zip(keys, combo))

        res = evaluate_config(X_train, y_train, X_val, y_val, X_test, y_test, hp)
        results.append({**hp, **{k: v for k, v in res.items() if k != "model"}})

        marker = ""
        if res["val_mae"] < best_val_mae:
            best_val_mae = res["val_mae"]
            best_config = hp.copy()
            best_model = res["model"]
            marker = " ★"

        if (idx + 1) % 10 == 0 or marker:
            print(f"  [{idx+1:3d}/{len(combos)}] "
                  f"depth={hp['max_depth']:2d} lr={hp['learning_rate']:.2f} "
                  f"iter={hp['max_iter']:4d} leaf={hp['min_samples_leaf']:2d} "
                  f"l2={hp['l2_regularization']:.2f} → "
                  f"val={res['val_mae']:.4f} test={res['test_mae']:.4f} "
                  f"({res['train_time']:.1f}s){marker}")

    # ── Results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    rdf = pd.DataFrame(results).sort_values("val_mae")
    print(f"\nTop 10 configs (by val MAE):")
    print(f"{'depth':>6s} {'lr':>6s} {'iter':>6s} {'leaf':>6s} {'l2':>6s} "
          f"{'val_MAE':>8s} {'test_MAE':>9s} {'test_RMSE':>10s} {'n_iter':>7s}")
    print("-" * 75)
    for _, row in rdf.head(10).iterrows():
        print(f"{row['max_depth']:6.0f} {row['learning_rate']:6.2f} "
              f"{row['max_iter']:6.0f} {row['min_samples_leaf']:6.0f} "
              f"{row['l2_regularization']:6.2f} "
              f"{row['val_mae']:8.4f} {row['test_mae']:9.4f} "
              f"{row['test_rmse']:10.4f} {row['n_iter']:7.0f}")

    print(f"\nBest config (by val MAE): {best_config}")
    print(f"  val MAE:  {best_val_mae:.4f}°C")
    # Retrieve test mae for best config
    best_test = rdf.iloc[0]
    print(f"  test MAE: {best_test['test_mae']:.4f}°C")
    print(f"  test RMSE:{best_test['test_rmse']:.4f}°C")
    print(f"  Baseline: {base_mae:.4f}°C")
    print(f"  Improvement vs baseline: {(1 - best_test['test_mae']/base_mae)*100:.1f}%")

    print(f"\nFeatures used: {best_fset_name} ({len(best_features)} features)")

    # ── Save best ───────────────────────────────────────────────────
    if args.save_best and best_model is not None:
        import joblib
        out_path = Path(args.save_best)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": best_model,
            "feature_columns": best_features,
            "stations_trained": args.stations,
            "hparams": best_config,
            "feature_set": best_fset_name,
            "val_mae": round(float(best_val_mae), 4),
            "test_mae": round(float(best_test["test_mae"]), 4),
            "test_rmse": round(float(best_test["test_rmse"]), 4),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        joblib.dump(bundle, out_path)
        print(f"\n[Saved] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Save sweep results CSV
    csv_path = Path("data/temp_train/mos_sweep_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(csv_path, index=False)
    print(f"[Saved] sweep results → {csv_path}")


if __name__ == "__main__":
    main()
