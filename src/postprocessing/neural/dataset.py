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
import re
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


# Признаки-наблюдения, которые строит scripts/postproc/add_obs_lags.py. Список
# DEFAULT_FEATURES писался раньше него, поэтому сеть их не видела вовсе — тот же
# пробел, что был в базовых линиях, где он стоил температуре 0,5% против 11,3%.
# Отбираем по образцу имени; образец узкий, под него не попадают сами наблюдения
# (obs_u10, obs_t2m_K), иначе цель оказалась бы среди признаков.
OBS_FEATURE_RE = re.compile(r"^(obs|err)_[a-z0-9]+_(lag\d+|lag_mean|tend24|anom)$")


def observation_features(df) -> List[str]:
    cols = [c for c in df.columns if OBS_FEATURE_RE.match(c)]
    if "obs_lag_age_h" in df.columns:
        cols.append("obs_lag_age_h")
    return cols


COLUMN_ALIASES = {
    "lat": ("station_lat",),
    "lon": ("station_lon",),
    "elev": ("station_elev",),
    "obs_t2m": ("obs_t2m_K",),
}


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
        auto_obs_features: bool = True,
        scalers: Optional[Dict[str, Tuple[float, float]]] = None,
        filter_expr: Optional[str] = None,
        drop_obs_missing: bool = True,
        station_to_idx: Optional[Dict[str, int]] = None,
    ):
        path = Path(parquet_path)
        if path.is_dir():
            df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)

        rename_map = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        if filter_expr:
            df = df.query(filter_expr).copy()
        if drop_obs_missing:
            df = df.dropna(subset=list(target_cols)).copy()

        # Признаки-наблюдения добавляются сами, если они есть в корпусе:
        # иначе сравнение с базовыми линиями было бы нечестным — там они есть.
        feature_cols = list(feature_cols)
        # Переданные нормировки задают набор признаков жёстко: они посчитаны на
        # обучении, и добавлять к ним что-то — значит строить вход не той
        # ширины, чем ждёт модель. Так 28.08.2026 упала оценка абляции: в
        # чекпойнте 26 признаков, а датасет дополнил их до 60.
        if scalers is not None:
            auto_obs_features = False
        if auto_obs_features:
            extra = [c for c in observation_features(df) if c not in feature_cols]
            if extra:
                print(f"[dataset] признаков-наблюдений добавлено: {len(extra)}")
                feature_cols += extra
        # Несоответствие станции ячейке сетки — главный источник поправки.
        # Разбор по станциям 28.08.2026: выигрыш связан с сырой ошибкой на 0,91,
        # с модулем смещения на 0,85 и с высотой станции на 0,72; наибольший — у
        # станций на 420-1850 м с холодным смещением до -4,7 °C. Двигает дело
        # именно РАЗНОСТЬ высоты станции и рельефа модели, а не каждая по
        # отдельности: в списке они есть, но по разным осям и в разных
        # масштабах. Даём её явно, вместе с поправкой на вертикальный градиент.
        if {"elev", "z_surf"} <= set(df.columns) and "dz_station" not in df.columns:
            df["dz_station"] = df["elev"].astype("float32") - df["z_surf"].astype("float32")
            if "lapse_t850_1000" in df.columns:
                # Приблизительный сдвиг температуры из-за этой разности высот.
                df["dz_lapse"] = (df["dz_station"]
                                  * df["lapse_t850_1000"].astype("float32") / 1000.0)
            nb_cols = sorted(x for x in df.columns if x.startswith("nb_"))
            for c in ("dz_station", "dz_lapse", *nb_cols):
                # Условие — «мы на обучении», а не auto_obs_features: тот флаг
                # выключает наблюдения станции, а высота к ним не относится, и
                # абляция по наблюдениям не должна заодно лишаться рельефа.
                if c in df.columns and c not in feature_cols and scalers is None:
                    feature_cols = list(feature_cols) + [c]
                    print(f"[dataset] добавлен признак {c}")

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
            # nan-версии: у ветровых лагов заполнено 65-87%, и обычное среднее
            # обратило бы в NaN весь столбец, а за ним и всю выборку.
            mean = np.nanmean(self.X, axis=0)
            std = np.nanstd(self.X, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std), std, 1.0)
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
        # Пропуск после нормировки становится нулём, то есть средним значением
        # признака: нейтральное значение, не сдвигающее прогноз ни в какую
        # сторону. Свежесть наблюдения сеть видит отдельно, в obs_lag_age_h.
        n_miss = int((~np.isfinite(self.X_norm)).sum())
        if n_miss:
            print(f"[dataset] пропусков в признаках: {n_miss:,} "
                  f"({n_miss / self.X_norm.size * 100:.2f}%) — заменены средним")
        np.nan_to_num(self.X_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # station→idx (for v2 station embedding); if mapping missing keys, raise.
        self.station_to_idx = station_to_idx
        if station_to_idx is not None:
            try:
                self.station_idx_arr = np.array(
                    [station_to_idx[s] for s in self.station_ids], dtype=np.int64
                )
            except KeyError as e:
                raise KeyError(f"station_to_idx missing station_usaf={e}")
        else:
            self.station_idx_arr = None
        # lead_norm column (if present in parquet) for v2 FiLM
        self.lead_norm_arr = (
            df["lead_norm"].to_numpy(dtype=np.float32, copy=True)
            if "lead_norm" in df.columns
            else None
        )

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
        sample = {
            "features": torch.from_numpy(self.X_norm[idx]),
            "gnn_t2m": torch.tensor(self.G[idx, 0], dtype=torch.float32),
            "gnn_u10": torch.tensor(self.G[idx, 1], dtype=torch.float32),
            "gnn_v10": torch.tensor(self.G[idx, 2], dtype=torch.float32),
            "t2m": torch.tensor(self.Y[idx, 0], dtype=torch.float32),
            "u10": torch.tensor(self.Y[idx, 1], dtype=torch.float32),
            "v10": torch.tensor(self.Y[idx, 2], dtype=torch.float32),
        }
        if self.station_idx_arr is not None:
            sample["station_idx"] = torch.tensor(
                int(self.station_idx_arr[idx]), dtype=torch.long
            )
        if self.lead_norm_arr is not None:
            sample["lead_norm"] = torch.tensor(
                float(self.lead_norm_arr[idx]), dtype=torch.float32
            )
        return sample


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
