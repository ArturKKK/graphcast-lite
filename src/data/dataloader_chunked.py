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
    ):
        self.data_dir = data_dir
        self.obs_window = obs_window
        self.pred_steps = pred_steps
        self.split = split
        self.test_fraction = test_fraction

        # 1. Load scalers
        scalers = np.load(os.path.join(data_dir, "scalers.npz"))
        self.mean = scalers["mean"].astype(np.float32)  # (n_feat,)
        self.std = scalers["std"].astype(np.float32)    # (n_feat,)

        # 2. Load data files
        # Support both single data.npy and multi-chunk chunk_*.npy
        single_file = os.path.join(data_dir, "data.npy")
        if os.path.exists(single_file):
            chunk_files = [single_file]
        else:
            chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npy")))
        if not chunk_files:
            raise FileNotFoundError(f"No data.npy or chunk_*.npy found in {data_dir}")

        # 3. Open as memory-mapped (not loaded into RAM!)
        self.chunks: List[np.memmap] = []
        self.chunk_lengths: List[int] = []
        total_time = 0

        for cf in chunk_files:
            # Peek shape from .npy header
            mm = np.load(cf, mmap_mode="r")  # memory-mapped, read-only
            self.chunks.append(mm)
            self.chunk_lengths.append(mm.shape[0])
            total_time += mm.shape[0]

        self.total_time = total_time
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
        window_size = obs_window + pred_steps
        
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
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"[ChunkDataset] {split}: {len(self._sample_indices)} samples, "
              f"grid={self.n_lon}×{self.n_lat}, feat={self.n_feat}, "
              f"obs={obs_window}, pred={pred_steps}")

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx):
        chunk_idx, local_t = self._sample_indices[idx]
        chunk = self.chunks[chunk_idx]

        # Extract window: [local_t : local_t + obs + pred]
        window = chunk[local_t : local_t + self.obs_window + self.pred_steps]
        # window shape: (obs+pred, lon, lat, feat_total)

        # Convert to float32 and select features
        window = window[:, :, :, :self.n_feat].astype(np.float32)

        # Normalize
        window = (window - self.mean) / self.std

        # Split into X and Y
        X_frames = window[:self.obs_window]   # (obs, lon, lat, feat)
        Y_frames = window[self.obs_window:]   # (pred, lon, lat, feat)

        # Flatten spatial dims: (lon, lat) -> (grid_nodes,)
        grid_nodes = self.n_lon * self.n_lat

        # Flatten obs*feat for X: (obs, lon, lat, feat) -> (lon*lat, obs*feat)
        X = X_frames.transpose(1, 2, 0, 3).reshape(grid_nodes, self.obs_window * self.n_feat)

        # Flatten pred*feat for Y: (pred, lon, lat, feat) -> (lon*lat, pred*feat)
        Y = Y_frames.transpose(1, 2, 0, 3).reshape(grid_nodes, self.pred_steps * self.n_feat)

        return torch.from_numpy(X), torch.from_numpy(Y)


def load_chunked_datasets(
    data_path: str,
    obs_window: int = 2,
    pred_steps: int = 1,
    n_features: Optional[int] = None,
    test_fraction: float = 0.2,
) -> Tuple[Dataset, Dataset, Dataset, DatasetMetadata]:
    """
    Convenience function matching the interface of load_train_and_test_datasets.
    
    Returns: (train_dataset, val_dataset, test_dataset, dataset_metadata)
    """
    # Load coords for metadata
    coords = np.load(os.path.join(data_path, "coords.npz"))
    lons = coords["longitude"]
    lats = coords["latitude"]
    
    # Load variable list
    with open(os.path.join(data_path, "variables.json")) as f:
        var_names = json.load(f)
    
    n_feat = n_features if n_features else len(var_names)
    
    train_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split="train", n_features=n_feat, test_fraction=test_fraction,
    )
    val_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split="val", n_features=n_feat, test_fraction=test_fraction,
    )
    test_ds = TimeseriesChunkDataset(
        data_path, obs_window=obs_window, pred_steps=pred_steps,
        split="test_only", n_features=n_feat, test_fraction=test_fraction,
    )

    metadata = DatasetMetadata(
        flattened=True,
        num_latitudes=len(lats),
        num_longitudes=len(lons),
        num_features=n_feat,
        obs_window=obs_window,
        pred_window=pred_steps,
    )
    # Attach real coordinates
    metadata.cordinates = (lats.astype(np.float32), lons.astype(np.float32))

    return train_ds, val_ds, test_ds, metadata
