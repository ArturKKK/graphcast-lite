"""
Chunked timeseries dataset for large grids (512×256 and above).

Ключевое отличие от предыдущего подхода:
  - Данные хранятся как RAW TIMESERIES (float16 memmap), без дублирования.
  - Окна obs/pred создаются на лету в __getitem__.
  - Поддерживает несколько chunk-файлов (батчи по годам).
  - Нормализация (mean/std) применяется на лету из scalers.npz.
  - Для AR-обучения отдаёт (obs_window + ar_steps) последовательных кадров,
    а train.py сам делит на input/target.

Формат данных на диске:
  data/datasets/wb2_512x256_19f_ar/
    data.npy       — memmap float16 (T, 512, 256, 19)  ~81 GB
    scalers.npz    — mean, std (19,), n
    coords.npz     — longitude (512,), latitude (256,)
    variables.json

  Также поддерживает legacy формат с chunk_0.npy, chunk_1.npy, ...
"""

import os
import json
import glob
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.data_configs import DatasetMetadata


class _ConcatMemmap:
    """Concatenates two memmaps along the last (channel) axis on the fly.

    Used when dataset has both `data.npy` (base channels) and `data_extra.npy`
    (extension channels) — loader sees them as a single (T, ..., n_base+n_extra)
    array. Only supports first-axis slicing (which is what the dataset uses).
    """

    def __init__(self, base: np.memmap, extra: np.memmap):
        assert base.shape[:-1] == extra.shape[:-1], (
            f"shape mismatch base={base.shape} extra={extra.shape}")
        self.base = base
        self.extra = extra
        self.shape = base.shape[:-1] + (base.shape[-1] + extra.shape[-1],)
        self.dtype = base.dtype
        self.ndim = base.ndim

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        # Supports time-axis slicing: chunk[a:b] or chunk[i]
        b = self.base[key]
        e = self.extra[key]
        return np.concatenate([b, e], axis=-1)


class TimeseriesChunkDataset(Dataset):
    """
    Dataset that reads raw timeseries chunks and creates sliding windows on the fly.
    
    Each __getitem__ returns:
      X: (grid_nodes, obs_window * n_feat)  — flattened input
      Y: (grid_nodes, total_target * n_feat)  — flattened target (1 or more steps)
    
    Parameters
    ----------
    data_dir : str
        Path to dataset directory with chunk_*.npy, scalers.npz, etc.
    obs_window : int
        Number of input timesteps (e.g. 2 for original GraphCast).
    pred_steps : int
        Number of prediction steps to return as target (for AR training).
        With pred_steps=1, classic single-step training.
        With pred_steps=4, returns 4 consecutive targets for curriculum AR.
    split : str
        'train' or 'test'. Last 20% of total timesteps = test.
    n_features : int or None
        If set, use only first n_features channels.
    """

    def __init__(
        self,
        data_dir: str,
        obs_window: int = 2,
        pred_steps: int = 1,
        split: str = "train",
        n_features: Optional[int] = None,
        test_fraction: float = 0.2,
        time_stride: int = 1,
        obs_stride: int = 0,
    ):
        self.data_dir = data_dir
        self.obs_window = obs_window
        self.pred_steps = pred_steps
        self.split = split
        self.test_fraction = test_fraction
        # Шаг ЦЕЛИ в единицах сроков датасета (6 ч): на сколько вперёд смотрит
        # модель за одно применение. 1 — +6 ч, 4 — +24 ч, 28 — +7 суток.
        self.time_stride = max(1, int(time_stride))
        # Шаг ВХОДА: расстояние между входными кадрами. По умолчанию совпадает с
        # шагом цели — так было до 15.08.2026, и так обязано быть, если модель
        # применяется авторегрессионно (иначе на втором шаге понадобится кадр,
        # который мы перепрыгнули).
        #
        # Но для ПРЯМОГО прогноза итерировать не нужно, и тогда вход можно
        # оставить частым. Это важно: у суточной модели вход шёл через 24 ч, и
        # по двум таким срокам почти не видно, куда и как быстро движутся
        # системы — она проигрывала на +24 ч (1,31 против 1,24 °C у
        # шестичасовой). Развязка шагов возвращает информацию о тенденции,
        # сохраняя нулевое накопление ошибки.
        self.obs_stride = max(1, int(obs_stride if obs_stride else self.time_stride))

        # 1. Load scalers
        scalers = np.load(os.path.join(data_dir, "scalers.npz"))
        self.mean = scalers["mean"].astype(np.float32)  # (n_feat,)
        self.std = scalers["std"].astype(np.float32)    # (n_feat,)

        # 2. Load data files
        # Support both single data.npy (raw memmap) and multi-chunk chunk_*.npy
        single_file = os.path.join(data_dir, "data.npy")
        info_file = os.path.join(data_dir, "dataset_info.json")
        if os.path.exists(single_file) and os.path.exists(info_file):
            # Raw memmap created by build_dataset_512x256.py — no .npy header
            with open(info_file) as f:
                info = json.load(f)
            self.flat_grid = info.get("flat", False)

            n_feat_total = info["n_feat"]
            n_feat_base = info.get("n_feat_base", n_feat_total)
            extra_file = info.get("extra_file")
            n_feat_extra = info.get("n_feat_extra", 0)

            if self.flat_grid:
                # Flat multi-resolution data: (T, N_nodes, C)
                base_shape = (info["n_time"], info["n_nodes"], n_feat_base)
                extra_shape = (info["n_time"], info["n_nodes"], n_feat_extra) if extra_file else None
            else:
                # Regular grid data: (T, n_lon, n_lat, C)
                base_shape = (info["n_time"], info["n_lon"], info["n_lat"], n_feat_base)
                extra_shape = (info["n_time"], info["n_lon"], info["n_lat"], n_feat_extra) if extra_file else None

            base_mm = np.memmap(single_file, dtype=np.float16, mode="r", shape=base_shape)
            if extra_file:
                extra_path = os.path.join(data_dir, extra_file)
                if not os.path.exists(extra_path):
                    raise FileNotFoundError(
                        f"dataset_info.json указывает extra_file={extra_file}, но {extra_path} не найден")
                extra_mm = np.memmap(extra_path, dtype=np.float16, mode="r", shape=extra_shape)
                mm = _ConcatMemmap(base_mm, extra_mm)
                print(f"[ChunkDataset] склеиваем data.npy ({n_feat_base}ch) + {extra_file} ({n_feat_extra}ch) → {mm.shape[-1]}ch")
            else:
                mm = base_mm

            chunk_files = [single_file]
            self.chunks = [mm]
            self.chunk_lengths = [mm.shape[0]]
            total_time = mm.shape[0]
        else:
            self.flat_grid = False
            chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npy")))
            if not chunk_files:
                raise FileNotFoundError(f"No data.npy or chunk_*.npy found in {data_dir}")

            # 3. Open as memory-mapped (not loaded into RAM!)
            self.chunks = []
            self.chunk_lengths = []
            total_time = 0

            for cf in chunk_files:
                mm = np.load(cf, mmap_mode="r")
                self.chunks.append(mm)
                self.chunk_lengths.append(mm.shape[0])
                total_time += mm.shape[0]

        self.total_time = total_time
        if self.flat_grid:
            self.n_nodes = self.chunks[0].shape[1]
            self.n_lon = None
            self.n_lat = None
            self.n_feat_total = self.chunks[0].shape[2]
        else:
            self.n_nodes = None
            self.n_lon = self.chunks[0].shape[1]
            self.n_lat = self.chunks[0].shape[2]
            self.n_feat_total = self.chunks[0].shape[3]
        self.n_feat = n_features if n_features else self.n_feat_total

        # Apply feature subset to scalers too
        self.mean = self.mean[:self.n_feat]
        self.std = self.std[:self.n_feat]

        # 4. Build cumulative index for quick chunk lookup
        self.cum_lengths = np.cumsum(self.chunk_lengths)

        # 5. Determine valid sample indices
        # A sample at global time t needs timesteps [t, t+1, ..., t + obs + pred - 1]
        # But we can't cross chunk boundaries (temporal discontinuity!)
        # Шаг входа и шаг цели независимы (см. комментарий к obs_stride).
        # Входные кадры: t, t+os, …, t+(obs−1)*os
        # Целевые:       последний_вход + ts, +2*ts, …, +pred*ts
        window_size = ((obs_window - 1) * self.obs_stride
                       + pred_steps * self.time_stride + 1)

        self._sample_indices: List[Tuple[int, int]] = []  # (chunk_idx, local_t)
        for ci, chunk in enumerate(self.chunks):
            T_chunk = chunk.shape[0]
            n_valid = T_chunk - window_size + 1
            if n_valid <= 0:
                continue
            for local_t in range(n_valid):
                self._sample_indices.append((ci, local_t))

        # 6. Train/test split (by time, no shuffling)
        total_samples = len(self._sample_indices)
        split_idx = int(total_samples * (1 - test_fraction))

        if split == "train":
            self._sample_indices = self._sample_indices[:split_idx]
        elif split == "test":
            self._sample_indices = self._sample_indices[split_idx:]
        elif split == "val":
            # First half of test set
            test_indices = self._sample_indices[split_idx:]
            val_size = len(test_indices) // 2
            self._sample_indices = test_indices[:val_size]
        elif split == "test_only":
            # Second half of test set (without val)
            test_indices = self._sample_indices[split_idx:]
            val_size = len(test_indices) // 2
            self._sample_indices = test_indices[val_size:]
        elif split == "all":
            pass  # keep all samples — useful for WRF comparison on specific dates
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"[ChunkDataset] {split}: {len(self._sample_indices)} samples, "
              f"{'flat_nodes=' + str(self.n_nodes) if self.flat_grid else 'grid=' + str(self.n_lon) + '×' + str(self.n_lat)}, "
              f"feat={self.n_feat}, obs={obs_window}, pred={pred_steps}")

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx):
        chunk_idx, local_t = self._sample_indices[idx]
        chunk = self.chunks[chunk_idx]

        # Окно кадров с шагом time_stride.
        #
        # При stride=1 это подряд идущие шестичасовые сроки — как было всегда.
        # При stride=4 сетка становится суточной: вход t−24ч и t, цель t+24ч.
        # Шаг обязан быть одинаковым для входа и цели, иначе авторегрессия
        # рвётся на втором шаге: модель выдаст t+24ч, а для следующего входа
        # понадобится t+18ч, которого мы перепрыгнули.
        os_, ts = self.obs_stride, self.time_stride
        if os_ == 1 and ts == 1:
            window = chunk[local_t : local_t + self.obs_window + self.pred_steps]
        elif os_ == ts:
            n_frames = self.obs_window + self.pred_steps
            window = chunk[local_t : local_t + (n_frames - 1) * ts + 1 : ts]
        else:
            # Шаги входа и цели различаются — берём кадры поимённо, читая из
            # memmap только нужные, а не весь занимаемый интервал (при шаге
            # цели в 28 сроков он был бы в тридцать кадров).
            last_obs = local_t + (self.obs_window - 1) * os_
            idx = [local_t + i * os_ for i in range(self.obs_window)] + \
                  [last_obs + (j + 1) * ts for j in range(self.pred_steps)]
            window = chunk[idx]

        if self.flat_grid:
            # Flat data: window shape (obs+pred, N_nodes, feat_total)
            window = window[:, :, :self.n_feat].astype(np.float32)
            window = (window - self.mean) / self.std

            X_frames = window[:self.obs_window]   # (obs, N, feat)
            Y_frames = window[self.obs_window:]   # (pred, N, feat)

            N = self.n_nodes
            # (obs, N, feat) -> (N, obs*feat)
            X = X_frames.transpose(1, 0, 2).reshape(N, self.obs_window * self.n_feat)
            # (pred, N, feat) -> (N, pred*feat)
            Y = Y_frames.transpose(1, 0, 2).reshape(N, self.pred_steps * self.n_feat)

            return torch.from_numpy(X), torch.from_numpy(Y)

        # Regular grid data: window shape (obs+pred, lon, lat, feat_total)
        window = window[:, :, :, :self.n_feat].astype(np.float32)

        # Normalize
        window = (window - self.mean) / self.std

        # Split into X and Y
        X_frames = window[:self.obs_window]   # (obs, lon, lat, feat)
        Y_frames = window[self.obs_window:]   # (pred, lon, lat, feat)

        # Flatten spatial dims: (lat, lon) -> (grid_nodes,)
        # ВАЖНО: порядок (lat, lon) должен совпадать с np.meshgrid(lons, lats).reshape(-1)
        # в create_graphs.py, где lat меняется медленно, lon — быстро (lat-major).
        grid_nodes = self.n_lon * self.n_lat

        # (obs, lon, lat, feat) -> (lat, lon, obs, feat) -> (lat*lon, obs*feat)
        X = X_frames.transpose(2, 1, 0, 3).reshape(grid_nodes, self.obs_window * self.n_feat)

        # (pred, lon, lat, feat) -> (lat, lon, pred, feat) -> (lat*lon, pred*feat)
        Y = Y_frames.transpose(2, 1, 0, 3).reshape(grid_nodes, self.pred_steps * self.n_feat)

        return torch.from_numpy(X), torch.from_numpy(Y)


def load_chunked_datasets(
    data_path: str,
    obs_window: int = 2,
    pred_steps: int = 1,
    n_features: Optional[int] = None,
    test_fraction: float = 0.2,
    test_split: str = "test_only",
    time_stride: int = 1,
    obs_stride: int = 0,
) -> Tuple[Dataset, Dataset, Dataset, DatasetMetadata]:
    """
    Convenience function matching the interface of load_train_and_test_datasets.
    
    Args:
        test_split: which split to use as the "test" dataset returned.
                    Options: "test_only" (default), "val", "test", "train", "all".
    Returns: (train_dataset, val_dataset, test_dataset, dataset_metadata)
    """
    # Load coords for metadata
    coords = np.load(os.path.join(data_path, "coords.npz"))
    lons = coords["longitude"]
    lats = coords["latitude"]
    
    # Detect flat grid
    info_file = os.path.join(data_path, "dataset_info.json")
    is_flat = False
    if os.path.exists(info_file):
        with open(info_file) as f:
            info = json.load(f)
        is_flat = info.get("flat", False)
    
    # Load variable list
    with open(os.path.join(data_path, "variables.json")) as f:
        var_names = json.load(f)
    
    n_feat = n_features if n_features else len(var_names)
    
    train_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split="train", n_features=n_feat, test_fraction=test_fraction,
        time_stride=time_stride, obs_stride=obs_stride,
    )
    val_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split="val", n_features=n_feat, test_fraction=test_fraction,
        time_stride=time_stride, obs_stride=obs_stride,
    )
    test_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split=test_split, n_features=n_feat, test_fraction=test_fraction,
        time_stride=time_stride, obs_stride=obs_stride,
    )
    if time_stride > 1:
        print(f"[ChunkDataset] шаг по времени {time_stride} сроков "
              f"= {6 * time_stride} ч (вход и цель на одной сетке)")

    if is_flat:
        # Flat multi-resolution grid: lats and lons are paired (N,) arrays
        metadata = DatasetMetadata(
            flattened=True,
            num_latitudes=0,  # not meaningful for flat grid
            num_longitudes=0,
            num_features=n_feat,
            obs_window=obs_window,
            pred_window=pred_steps,
        )
        metadata.flat_grid = True
        metadata.num_grid_nodes = len(lats)
        metadata.cordinates = (lats.astype(np.float32), lons.astype(np.float32))
        if "is_regional" in coords:
            metadata.is_regional = coords["is_regional"]
        else:
            metadata.is_regional = None
    else:
        metadata = DatasetMetadata(
            flattened=True,
            num_latitudes=len(lats),
            num_longitudes=len(lons),
            num_features=n_feat,
            obs_window=obs_window,
            pred_window=pred_steps,
        )
        metadata.flat_grid = False
        metadata.cordinates = (lats.astype(np.float32), lons.astype(np.float32))

    return train_ds, val_ds, test_ds, metadata
