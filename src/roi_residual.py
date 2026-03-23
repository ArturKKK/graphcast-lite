from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from scipy.spatial import cKDTree

from src.create_graphs import _compute_mesh_edge_features
from src.models import InteractionNetProcessor, WeatherPrediction, _get_activation


def build_roi_knn_graph(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    roi: Tuple[float, float, float, float],
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    lat_min, lat_max, lon_min, lon_max = roi
    roi_mask = (
        (grid_lats >= lat_min)
        & (grid_lats <= lat_max)
        & (grid_lons >= lon_min)
        & (grid_lons <= lon_max)
    )
    roi_indices = np.where(roi_mask)[0]
    if len(roi_indices) == 0:
        raise ValueError(f"No grid points in ROI {roi}")

    roi_lats = grid_lats[roi_indices]
    roi_lons = grid_lons[roi_indices]

    def to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x, y, z], axis=-1)

    roi_xyz = to_xyz(roi_lats, roi_lons)
    tree = cKDTree(roi_xyz)
    k_eff = min(k + 1, len(roi_indices))
    _, neighbors = tree.query(roi_xyz, k=k_eff)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]

    if neighbors.shape[1] > 1:
        neighbors = neighbors[:, 1:]
    else:
        neighbors = neighbors[:, :0]

    senders = neighbors.reshape(-1)
    receivers = np.repeat(np.arange(len(roi_indices)), neighbors.shape[1])
    edge_index_np = np.stack([senders, receivers], axis=0).astype(np.int64)
    edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
    edge_features = _compute_mesh_edge_features(roi_lats, roi_lons, edge_index_np)

    return roi_mask, roi_indices, edge_index, edge_features


class ROIResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, node_state: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([node_state, skip_features], dim=-1))


class ROIResidualModel(nn.Module):
    def __init__(
        self,
        global_model: WeatherPrediction,
        roi: Tuple[float, float, float, float],
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        device: torch.device,
        hidden_dim: int = 256,
        processor_steps: int = 6,
        roi_k: int = 8,
    ):
        super().__init__()

        self.global_model = global_model
        self.device = device
        self.roi = roi
        self.n_features = global_model.num_features
        self.obs_window = global_model.obs_window
        self.output_channels = global_model.num_features

        roi_mask, roi_indices, roi_edge_index, roi_edge_features = build_roi_knn_graph(
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            roi=roi,
            k=roi_k,
        )
        self.register_buffer("roi_mask", torch.tensor(roi_mask, dtype=torch.bool))
        self.register_buffer("roi_indices", torch.tensor(roi_indices, dtype=torch.int64))
        self.register_buffer("roi_edge_index", roi_edge_index)
        self.register_buffer("roi_edge_features", roi_edge_features)
        self.n_roi_grid = int(roi_mask.sum())

        total_feature_size = self.n_features * self.obs_window
        global_latent_dim = global_model.encoder.output_dim
        self.skip_dim = total_feature_size + global_latent_dim + self.output_channels

        self.input_proj = nn.Sequential(
            nn.Linear(self.skip_dim, hidden_dim),
            _get_activation("swish"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.processor = InteractionNetProcessor(
            node_dim=hidden_dim,
            raw_edge_dim=4,
            edge_latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_steps=processor_steps,
            activation="swish",
            use_layer_norm=True,
        )

        self.decoder = ROIResidualHead(
            input_dim=hidden_dim + self.skip_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_channels,
        )

        self.to(device)

        n_trainable = sum(
            p.numel() for name, p in self.named_parameters()
            if not name.startswith("global_model.") and p.requires_grad
        )
        roi_deg = torch.bincount(self.roi_edge_index[1], minlength=self.n_roi_grid)
        print(f"\n[ROIResidual] Trainable parameters: {n_trainable:,}")
        print(f"[ROIResidual] ROI grid points: {self.n_roi_grid}")
        print(f"[ROIResidual] ROI graph edges: {self.roi_edge_index.shape[1]}")
        print(
            f"[ROIResidual] In-degree: min={roi_deg.min().item()} max={roi_deg.max().item()} "
            f"mean={roi_deg.float().mean().item():.1f}"
        )

    def _get_global_features(self, X_sq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gm = self.global_model
        X_preprocessed = gm._preprocess_input(grid_node_features=X_sq)
        encoded = gm.encoder.forward(X=X_preprocessed, edge_index=gm.encoding_graph)
        global_grid_latent = encoded[: gm._num_grid_nodes]
        return global_grid_latent, X_preprocessed

    def forward(self, X: torch.Tensor, attention_threshold: float = 0.0, **kwargs) -> torch.Tensor:
        assert X.shape[0] == 1, f"ROIResidualModel supports batch_size=1 only, got {X.shape[0]}"

        with torch.no_grad():
            global_pred = self.global_model(X=X, attention_threshold=attention_threshold, **kwargs)
            X_sq = X[0]
            global_grid_latent, _ = self._get_global_features(X_sq)

        global_pred = global_pred.detach()
        roi_raw = X_sq[self.roi_mask]
        roi_global_latent = global_grid_latent[self.roi_mask]
        roi_global_pred = global_pred[self.roi_mask]
        roi_skip = torch.cat([roi_raw, roi_global_latent, roi_global_pred], dim=-1)

        roi_state = self.input_proj(roi_skip)
        roi_state = self.processor(
            x=roi_state,
            edge_index=self.roi_edge_index,
            edge_attr_raw=self.roi_edge_features,
        )
        roi_correction = self.decoder(roi_state, roi_skip)

        correction_full = torch.zeros_like(global_pred)
        correction_full = correction_full.index_add(0, self.roi_indices, roi_correction)
        return global_pred + correction_full