"""Parquet-backed dataset для (GNN-forecast, station-obs) корпуса.

Ожидаемая схема Parquet (см. RFC §4.2):
    station_usaf:        str
    init_time_utc:       datetime64[ns]
    lead_h:              float32
    valid_time_utc:      datetime64[ns]
    source:              str  ('hindcast' | 'live_gdas')

    # raw GNN snapshot (denormalised, physical units):
    gnn_t2m, gnn_u10, gnn_v10, gnn_msl, gnn_sp,
    gnn_t850, gnn_t500, gnn_q850, gnn_z500,
    gnn_u850, gnn_v850, gnn_u1000, gnn_v1000

    # derived dynamic features:
    lapse_t850_1000, dewpoint_depression, solar_zen

    # static per-station:
    lat, lon, elev, urban_flag, dist_to_coast, z_surf, lsm

    # calendar / lead:
    sin_hour, cos_hour, sin_doy, cos_doy, lead_norm

    # targets:
    obs_t2m, obs_u10, obs_v10

См. docs/postprocessing_rfc.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


GNN_TARGET_COLS = ("gnn_t2m", "gnn_u10", "gnn_v10")
TARGET_COLS = ("obs_t2m", "obs_u10", "obs_v10")


DEFAULT_FEATURES: Tuple[str, ...] = (
    # GNN snapshot (raw — modelи увидит и target-каналы тоже как фичи)
    "gnn_t2m",
    "gnn_u10",
    "gnn_v10",
    "gnn_msl",
    "gnn_sp",
    "gnn_t850",
    "gnn_t500",
    "gnn_q850",
    "gnn_z500",
    "gnn_u850",
    "gnn_v850",
    "gnn_u1000",
    "gnn_v1000",
    # derived
    "lapse_t850_1000",
    "dewpoint_depression",
    "solar_zen",
    # static
    "lat",
    "lon",
    "elev",
    "urban_flag",
    "dist_to_coast",
    "z_surf",
    "lsm",
    # temporal / lead
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
    "lead_norm",
)


class StationCorpusDataset(Dataset):
    """In-memory Parquet dataset (для S3 корпус 3-5M строк помещается в RAM).

    Если корпус не влезает — на следующей итерации заменить на
    `pyarrow.dataset` + lazy IterableDataset.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        feature_cols: Sequence[str] = DEFAULT_FEATURES,
        target_cols: Sequence[str] = TARGET_COLS,
        scalers: Optional[Dict[str, Tuple[float, float]]] = None,
        filter_expr: Optional[str] = None,
        drop_obs_missing: bool = True,
    ):
        path = Path(parquet_path)
        if path.is_dir():
            df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)

        if filter_expr:
            df = df.query(filter_expr).copy()
        if drop_obs_missing:
            df = df.dropna(subset=list(target_cols)).copy()

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing columns: {missing}")

        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.station_ids = df["station_usaf"].astype(str).to_numpy()

        # to float32 numpy для быстрого индекса
        self.X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.Y = df[self.target_cols].to_numpy(dtype=np.float32, copy=True)
        # raw GNN баз для residual-головы (физические единицы)
        self.G = df[list(GNN_TARGET_COLS)].to_numpy(dtype=np.float32, copy=True)
        self.df_meta = df[["station_usaf", "init_time_utc", "lead_h", "valid_time_utc"]].reset_index(
            drop=True
        )

        # scaling: либо передан, либо считаем на лету (только для train!)
        if scalers is None:
            mean = self.X.mean(axis=0)
            std = self.X.std(axis=0)
            std[std < 1e-6] = 1.0
            self.feature_mean = mean.astype(np.float32)
            self.feature_std = std.astype(np.float32)
        else:
            self.feature_mean = np.array(
                [scalers[c][0] for c in self.feature_cols], dtype=np.float32
            )
            self.feature_std = np.array(
                [scalers[c][1] for c in self.feature_cols], dtype=np.float32
            )

        self.X_norm = (self.X - self.feature_mean) / self.feature_std

    def export_scalers(self) -> Dict[str, Tuple[float, float]]:
        return {
            c: (float(self.feature_mean[i]), float(self.feature_std[i]))
            for i, c in enumerate(self.feature_cols)
        }

    def save_scalers(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.export_scalers(), f, indent=2)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.X_norm[idx]),
            "gnn_t2m": torch.tensor(self.G[idx, 0], dtype=torch.float32),
            "gnn_u10": torch.tensor(self.G[idx, 1], dtype=torch.float32),
            "gnn_v10": torch.tensor(self.G[idx, 2], dtype=torch.float32),
            "t2m": torch.tensor(self.Y[idx, 0], dtype=torch.float32),
            "u10": torch.tensor(self.Y[idx, 1], dtype=torch.float32),
            "v10": torch.tensor(self.Y[idx, 2], dtype=torch.float32),
        }


def build_balanced_sampler(dataset: StationCorpusDataset) -> WeightedRandomSampler:
    """Per-station balanced sampler: вероятность семпла обратна частоте станции.

    Обоснование — RFC §3.1a (вместо явного bias-reg).
    """
    stations = dataset.station_ids
    unique, counts = np.unique(stations, return_counts=True)
    freq = dict(zip(unique, counts))
    weights = np.array([1.0 / freq[s] for s in stations], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
    )
