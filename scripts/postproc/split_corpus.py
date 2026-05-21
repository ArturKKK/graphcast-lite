#!/usr/bin/env python
"""Split postproc corpus into train/val by year of valid_time (or init_time)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input parquet")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019])
    ap.add_argument("--val-years", type=int, nargs="+", default=[2020])
    ap.add_argument("--time-col", default=None,
                    help="time column to use; auto-detected if not given")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"loaded {len(df):,} rows, {len(df.columns)} cols")

    tcol = args.time_col
    if tcol is None:
        for cand in ("valid_time", "init_time", "time", "valid", "init"):
            if cand in df.columns:
                tcol = cand
                break
    if tcol is None:
        print("ERROR: no time column found. Columns:", list(df.columns), file=sys.stderr)
        sys.exit(2)
    print(f"using time column: {tcol}")

    t = pd.to_datetime(df[tcol])
    years = t.dt.year
    print("year distribution:", dict(years.value_counts().sort_index()))

    train = df[years.isin(args.train_years)].reset_index(drop=True)
    val = df[years.isin(args.val_years)].reset_index(drop=True)
    print(f"train rows: {len(train):,}  ({sorted(args.train_years)})")
    print(f"val   rows: {len(val):,}  ({sorted(args.val_years)})")

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out_train, index=False)
    val.to_parquet(args.out_val, index=False)
    print(f"wrote {args.out_train}")
    print(f"wrote {args.out_val}")


if __name__ == "__main__":
    main()
