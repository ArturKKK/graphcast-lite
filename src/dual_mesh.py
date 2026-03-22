"""
src/dual_mesh.py

Двухуровневая mesh-модель: глобальный меш (pretrained) + региональный refined меш.

Глобальная модель НЕ МОДИФИЦИРУЕТСЯ — работает на прежней топологии.
Региональный меш (уровень 7/8 икосаэдра в ROI) добавляет spatial DoF
в области интереса. Cross-edges обеспечивают обмен информацией между
двумя мешами на каждом шаге процессора.

Архитектура:
  Grid ──► Global Mesh (pretrained) ──► Grid (глобальный прогноз)
            ↕ cross-edges ↕
  Grid(ROI) ──► Regional Mesh (trainable) ──► Grid(ROI) (уточнение)

Финальный прогноз в ROI = глобальный + региональная поправка.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import LayerNorm as PygLayerNorm
from torch_geometric.utils import scatter

from typing import Tuple, List, Optional
from src.mesh.create_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    TriangularMesh,
    get_edges_from_faces,
)
from src.models import (
    InteractionNetLayer,
    InteractionNetProcessor,
    WeatherPrediction,
    _get_activation,
)
from src.create_graphs import _compute_mesh_edge_features
from src.utils import get_mesh_lat_long


# ─── 1. Построение регионального меша ─────────────────────────────────

def create_regional_mesh(
    roi: Tuple[float, float, float, float],
    level: int = 8,
    buffer_deg: float = 2.0,
) -> Tuple[TriangularMesh, np.ndarray, np.ndarray]:
    """Создаёт regional refined mesh: вершины уровня `level` внутри ROI + buffer.

    Parameters
    ----------
    roi : (lat_min, lat_max, lon_min, lon_max)
    level : int
        Уровень икосаэдра (7 → ~0.55°, 8 → ~0.28°)
    buffer_deg : float
        Расширение ROI для плавного перехода

    Returns
    -------
    regional_mesh : TriangularMesh
        Меш только в области ROI
    reg_lats, reg_lons : np.ndarray
        Координаты вершин регионального меша
    """
    lat_min, lat_max, lon_min, lon_max = roi

    # Строим полную иерархию до нужного уровня
    meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=level)
    finest = meshes[level]

    # Конвертируем 3D координаты в lat/lon
    vertices = finest.vertices  # (V, 3) на единичной сфере
    lats_rad = np.arcsin(np.clip(vertices[:, 2], -1, 1))  # z = sin(lat)
    lons_rad = np.arctan2(vertices[:, 1], vertices[:, 0])  # atan2(y, x)
    lats_deg = np.degrees(lats_rad)
    lons_deg = np.degrees(lons_rad) % 360  # [0, 360)

    # Фильтруем вершины в ROI + buffer
    # Но: отсекаем вершины, которые уже есть в глобальном меше (уровень 6)
    # Уровень 6 содержит первые V(6) = 40962 вершин (они — префикс)
    n_global = len(meshes[min(level, 6)].vertices)  # 40962 для level >= 6

    # Создаём маску: в ROI + buffer И не в глобальном меше
    lat_ok = (lats_deg >= lat_min - buffer_deg) & (lats_deg <= lat_max + buffer_deg)
    lon_ok = (lons_deg >= lon_min - buffer_deg) & (lons_deg <= lon_max + buffer_deg)
    in_roi = lat_ok & lon_ok

    # Вершины уровня level, которых нет в уровне 6
    new_only = np.zeros(len(vertices), dtype=bool)
    new_only[n_global:] = True  # вершины с индексами >= n_global — это новые

    mask = in_roi & new_only
    kept_indices = np.where(mask)[0]

    if len(kept_indices) == 0:
        raise ValueError(
            f"No regional mesh vertices found in ROI {roi} with buffer={buffer_deg}°. "
            f"Try increasing buffer_deg or level."
        )

    # Переиндексируем: строим маппинг old_idx → new_idx
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[kept_indices] = np.arange(len(kept_indices))

    # Фильтруем грани: оставляем только те, где ВСЕ 3 вершины выжили
    faces = finest.faces
    face_mask = np.all(old_to_new[faces] >= 0, axis=1)
    new_faces = old_to_new[faces[face_mask]]

    regional_mesh = TriangularMesh(
        vertices=vertices[kept_indices],
        faces=new_faces.astype(np.int32),
    )

    reg_lats = lats_deg[kept_indices].astype(np.float32)
    reg_lons = lons_deg[kept_indices].astype(np.float32)

    print(f"[RegionalMesh] level={level}, ROI={roi}, buffer={buffer_deg}°")
    print(f"  Total level-{level} vertices: {len(vertices)}")
    print(f"  Global mesh (level 6) vertices: {n_global}")
    print(f"  Regional vertices in ROI: {len(kept_indices)}")
    print(f"  Regional faces: {len(new_faces)}")

    return regional_mesh, reg_lats, reg_lons


# ─── 2. Cross-edges между глобальным и региональным мешом ─────────────

def build_cross_edges(
    global_lats: np.ndarray,
    global_lons: np.ndarray,
    reg_lats: np.ndarray,
    reg_lons: np.ndarray,
    k: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Строит cross-edges между глобальным и региональным мешом.

    Для каждого регионального узла находит k ближайших глобальных узлов.
    Создаёт рёбра в обе стороны (bidirectional).

    Returns
    -------
    cross_edge_index : torch.Tensor [2, E]
        Рёбра: (global_idx, regional_idx) в обе стороны.
        global_idx — индексы в глобальном меше.
        regional_idx — индексы в региональном меше.
    cross_edge_features : torch.Tensor [E, 4]
        Edge features (distance + relative position).
    """
    from scipy.spatial import cKDTree
    from src.utils import lat_lon_deg_to_spherical, spherical_to_cartesian

    # Конвертируем в 3D для KDTree
    def to_xyz(lats, lons):
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x, y, z], axis=-1)

    global_xyz = to_xyz(global_lats, global_lons)
    reg_xyz = to_xyz(reg_lats, reg_lons)

    # Для каждого регионального узла — k ближайших глобальных
    tree = cKDTree(global_xyz)
    distances, global_indices = tree.query(reg_xyz, k=k)

    # Строим edge_index: regional → global и global → regional
    n_reg = len(reg_lats)
    reg_indices = np.repeat(np.arange(n_reg), k)
    glob_indices = global_indices.flatten()

    # Bidirectional: global→regional + regional→global
    # Формат: (sender, receiver) — два массива
    senders = np.concatenate([glob_indices, reg_indices])
    receivers = np.concatenate([reg_indices, glob_indices])

    cross_edge_index = torch.tensor(
        np.stack([senders, receivers], axis=0), dtype=torch.int64
    )

    # Edge features: нужны координаты всех узлов в едином пространстве
    all_lats = np.concatenate([global_lats, reg_lats])
    all_lons = np.concatenate([global_lons, reg_lons])

    # Сдвигаем региональные индексы в edge_index
    n_global = len(global_lats)
    # senders: первая половина = global (без сдвига), вторая = regional (+n_global)
    # receivers: первая половина = regional (+n_global), вторая = global (без сдвига)
    unified_senders = np.concatenate([glob_indices, reg_indices + n_global])
    unified_receivers = np.concatenate([reg_indices + n_global, glob_indices])

    cross_edge_features = _compute_mesh_edge_features(
        mesh_node_lats=all_lats,
        mesh_node_longs=all_lons,
        edge_index=np.stack([unified_senders, unified_receivers], axis=0),
    )

    print(f"[CrossEdges] {len(senders)} edges ({n_reg}×{k} bidirectional)")

    return cross_edge_index, cross_edge_features


# ─── 3. Региональный Grid↔RegMesh рёбра ──────────────────────────────

def build_regional_grid_mesh_edges(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    reg_lats: np.ndarray,
    reg_lons: np.ndarray,
    roi: Tuple[float, float, float, float],
    radius_factor: float = 0.6,
    reg_mesh: TriangularMesh = None,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Строит рёбра Grid(ROI)→RegionalMesh и RegionalMesh→Grid(ROI).

    Returns
    -------
    roi_grid_mask : np.ndarray (bool, N_grid)
        Маска grid-точек, попадающих в ROI
    encoding_edges : torch.Tensor [2, E]
        Grid(ROI) → RegMesh (sender=grid_roi_local, receiver=reg_mesh)
    decoding_edges : torch.Tensor [2, E]
        RegMesh → Grid(ROI) (sender=reg_mesh, receiver=grid_roi_local)
    """
    from scipy.spatial import cKDTree

    lat_min, lat_max, lon_min, lon_max = roi

    # Маска grid-точек в ROI
    roi_mask = (
        (grid_lats >= lat_min) & (grid_lats <= lat_max) &
        (grid_lons >= lon_min) & (grid_lons <= lon_max)
    )
    roi_indices = np.where(roi_mask)[0]
    n_roi = len(roi_indices)

    if n_roi == 0:
        raise ValueError(f"No grid points in ROI {roi}")

    roi_lats = grid_lats[roi_indices]
    roi_lons = grid_lons[roi_indices]

    # --- Encoding: Grid(ROI) → RegMesh (radius-based) ---
    def to_xyz(lats, lons):
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x, y, z], axis=-1)

    reg_xyz = to_xyz(reg_lats, reg_lons)
    roi_xyz = to_xyz(roi_lats, roi_lons)

    # Радиус: макс. расстояние между соседями в региональном меше × factor
    if reg_mesh is not None and len(reg_mesh.faces) > 0:
        edges = get_edges_from_faces(reg_mesh.faces)
        v = reg_mesh.vertices
        edge_lengths = np.linalg.norm(v[edges[0]] - v[edges[1]], axis=-1)
        max_edge_dist = edge_lengths.max()
    else:
        # Fallback: estimate from average distance
        tree_tmp = cKDTree(reg_xyz)
        dists, _ = tree_tmp.query(reg_xyz, k=2)
        max_edge_dist = dists[:, 1].max()

    radius = max_edge_dist * radius_factor

    # Для каждой grid(ROI) точки — все reg_mesh узлы в радиусе
    tree = cKDTree(reg_xyz)
    enc_grid_idx = []
    enc_mesh_idx = []
    for i, xyz in enumerate(roi_xyz):
        neighbors = tree.query_ball_point(xyz, radius)
        for j in neighbors:
            enc_grid_idx.append(i)
            enc_mesh_idx.append(j)

    # Если radius слишком мал — fallback на k=3 nearest
    if len(enc_grid_idx) < n_roi:
        print(f"[WARNING] Radius encoding found {len(enc_grid_idx)} edges for {n_roi} ROI points. Using k=3 fallback.")
        dists, inds = tree.query(roi_xyz, k=3)
        enc_grid_idx = np.repeat(np.arange(n_roi), 3).tolist()
        enc_mesh_idx = inds.flatten().tolist()

    encoding_edges = torch.tensor(
        np.stack([enc_grid_idx, enc_mesh_idx], axis=0), dtype=torch.int64
    )

    # --- Decoding: RegMesh → Grid(ROI) (k=3 nearest) ---
    tree_grid = cKDTree(roi_xyz)
    dists, mesh_neighbors = tree_grid.query(reg_xyz, k=min(3, n_roi))
    if mesh_neighbors.ndim == 1:
        mesh_neighbors = mesh_neighbors[:, None]
    
    dec_mesh_idx = []
    dec_grid_idx = []
    for i in range(len(reg_lats)):
        for j in mesh_neighbors[i]:
            if j < n_roi:
                dec_mesh_idx.append(i)
                dec_grid_idx.append(j)

    decoding_edges = torch.tensor(
        np.stack([dec_mesh_idx, dec_grid_idx], axis=0), dtype=torch.int64
    )

    print(f"[RegionalGridMesh] ROI grid points: {n_roi}")
    print(f"  Encoding edges (grid→reg_mesh): {encoding_edges.shape[1]}")
    print(f"  Decoding edges (reg_mesh→grid): {decoding_edges.shape[1]}")

    return roi_mask, encoding_edges, decoding_edges


# ─── 4. Cross-Message модуль ──────────────────────────────────────────

class CrossMessageLayer(nn.Module):
    """Однонаправленный обмен сообщениями: Global → Regional.

    Для каждого регионального узла собираем информацию от k ближайших
    глобальных mesh-узлов через cross-edges.

    Обратное направление (regional→global) не используется, т.к.
    глобальная модель заморожена и её латенты detached.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 global_latent_dim: int = None, activation: str = "swish"):
        super().__init__()
        act = _get_activation(activation)
        g_dim = global_latent_dim if global_latent_dim is not None else node_dim

        # Global → Regional message
        self.g2r_edge_mlp = nn.Sequential(
            nn.Linear(g_dim + node_dim + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm_reg = PygLayerNorm(node_dim, mode="node")

    def forward(
        self,
        h_global: torch.Tensor,      # (N_glob, D)
        h_regional: torch.Tensor,    # (N_reg, D)
        cross_edge_index: torch.Tensor,  # (2, E_cross)
        cross_edge_attr: torch.Tensor,   # (E_cross, edge_dim)
        n_global: int,
    ):
        """
        cross_edge_index содержит рёбра в обе стороны:
        - первая половина: global→regional  (используем)
        - вторая половина: regional→global  (не используем)
        Индексы: global [0, n_global), regional [0, n_regional)
        """
        n_reg = h_regional.shape[0]
        E = cross_edge_index.shape[1]
        half_E = E // 2

        # --- Global → Regional ---
        g2r_senders = cross_edge_index[0, :half_E]      # global indices
        g2r_receivers = cross_edge_index[1, :half_E]     # regional indices
        g2r_edge_attr = cross_edge_attr[:half_E]

        g2r_input = torch.cat([
            h_global[g2r_senders],
            h_regional[g2r_receivers],
            g2r_edge_attr,
        ], dim=-1)
        g2r_msg = self.g2r_edge_mlp(g2r_input)
        g2r_agg = scatter(g2r_msg, g2r_receivers, dim=0, dim_size=n_reg, reduce="mean")
        h_regional = self.norm_reg(h_regional + g2r_agg)

        return h_regional


# ─── 5. Региональный процессор ────────────────────────────────────────

class RegionalProcessor(nn.Module):
    """Маленький InteractionNet процессор для регионального меша.

    Shared weights (одинаковые на всех шагах) для экономии параметров.
    """

    def __init__(self, node_dim: int, raw_edge_dim: int = 4,
                 hidden_dim: int = 256, num_steps: int = 4,
                 activation: str = "swish"):
        super().__init__()

        act = _get_activation(activation)

        self.edge_encoder = nn.Sequential(
            nn.Linear(raw_edge_dim, node_dim),
            act,
        )

        # Shared weights: один InteractionNetLayer для всех шагов
        self.step = InteractionNetLayer(
            node_dim=node_dim,
            edge_dim=node_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layer_norm=True,
        )
        self.num_steps = num_steps

    def forward(self, x, edge_index, edge_attr_raw):
        edge_attr = self.edge_encoder(edge_attr_raw)
        for _ in range(self.num_steps):
            x, edge_attr = self.step(x, edge_index, edge_attr)
        return x


# ─── 6. Regional Encoder / Decoder ───────────────────────────────────

class RegionalEncoder(nn.Module):
    """Энкодер: Grid(ROI) → Regional Mesh.

    Простой MLP + scatter_mean по рёбрам.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, grid_features: torch.Tensor, edge_index: torch.Tensor,
                n_mesh_nodes: int):
        """
        grid_features: (N_roi, D_in) — фичи grid-точек в ROI
        edge_index: (2, E) — [grid_idx, mesh_idx]
        """
        x = self.mlp(grid_features)
        # Aggregate grid → mesh через edge_index
        grid_msg = x[edge_index[0]]
        mesh_features = scatter(grid_msg, edge_index[1], dim=0,
                                dim_size=n_mesh_nodes, reduce="mean")
        return mesh_features


class RegionalDecoder(nn.Module):
    """Декодер: Regional Mesh → Grid(ROI).

    Собирает фичи с mesh nodes и прогоняет через MLP → output_dim.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, mesh_features: torch.Tensor, edge_index: torch.Tensor,
                n_grid_nodes: int):
        """
        mesh_features: (N_mesh, D)
        edge_index: (2, E) — [mesh_idx, grid_idx]
        """
        mesh_msg = mesh_features[edge_index[0]]
        grid_agg = scatter(mesh_msg, edge_index[1], dim=0,
                           dim_size=n_grid_nodes, reduce="mean")
        return self.mlp(grid_agg)


# ─── 7. DualMeshModel — главная обёртка ──────────────────────────────

class DualMeshModel(nn.Module):
    """Двухуровневая модель: глобальный pretrained + региональный refined.

    Global model (frozen/slow LR) даёт базовый прогноз.
    Regional module добавляет поправку Δy в ROI точках.

    Output = global_pred.clone()
    Output[roi_mask] += regional_correction
    """

    def __init__(
        self,
        global_model: WeatherPrediction,
        roi: Tuple[float, float, float, float],
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        device: torch.device,
        reg_mesh_level: int = 7,
        reg_mesh_buffer: float = 2.0,
        reg_processor_steps: int = 4,
        cross_k: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.global_model = global_model
        self.device = device
        self.roi = roi
        self.n_features = global_model.num_features
        self.obs_window = global_model.obs_window

        # Decoder output = num_features (e.g. 19 channels)
        self.output_channels = global_model.num_features

        # ── 1. Региональный меш ──
        self.reg_mesh, reg_lats, reg_lons = create_regional_mesh(
            roi=roi, level=reg_mesh_level, buffer_deg=reg_mesh_buffer,
        )
        self.n_reg_mesh = len(reg_lats)

        # ── 2. Processing graph для регионального меша ──
        reg_edge_index = torch.tensor(
            get_edges_from_faces(self.reg_mesh.faces), dtype=torch.int64
        )
        reg_edge_features = _compute_mesh_edge_features(
            mesh_node_lats=reg_lats,
            mesh_node_longs=reg_lons,
            edge_index=reg_edge_index.numpy(),
        )
        self.register_buffer("reg_processing_edges", reg_edge_index)
        self.register_buffer("reg_processing_edge_features", reg_edge_features)

        # ── 3. Cross-edges: global mesh ↔ regional mesh ──
        global_mesh_lats = global_model._mesh_nodes_lat
        global_mesh_lons = global_model._mesh_nodes_lon
        n_global_mesh = global_model._num_mesh_nodes

        cross_edge_index, cross_edge_features = build_cross_edges(
            global_lats=global_mesh_lats,
            global_lons=global_mesh_lons,
            reg_lats=reg_lats,
            reg_lons=reg_lons,
            k=cross_k,
        )
        self.register_buffer("cross_edge_index", cross_edge_index)
        self.register_buffer("cross_edge_features", cross_edge_features)
        self.n_global_mesh = n_global_mesh

        # ── 4. Grid(ROI) ↔ RegMesh edges ──
        roi_mask, enc_edges, dec_edges = build_regional_grid_mesh_edges(
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            reg_lats=reg_lats,
            reg_lons=reg_lons,
            roi=roi,
            reg_mesh=self.reg_mesh,
        )
        self.register_buffer("roi_mask", torch.tensor(roi_mask, dtype=torch.bool))
        self.register_buffer("reg_encoding_edges", enc_edges)
        self.register_buffer("reg_decoding_edges", dec_edges)
        self.n_roi_grid = int(roi_mask.sum())

        # ── 5. Regional modules (trainable) ──
        # Input dim для регионального энкодера: T*F + global encoder latent dim
        total_feature_size = self.n_features * self.obs_window
        global_latent_dim = global_model.encoder.output_dim  # 256 из конфига
        reg_enc_input_dim = total_feature_size + global_latent_dim  # raw features + global latent

        self.reg_encoder = RegionalEncoder(
            input_dim=reg_enc_input_dim, hidden_dim=hidden_dim,
        ).to(device)

        self.reg_processor = RegionalProcessor(
            node_dim=hidden_dim,
            raw_edge_dim=4,
            hidden_dim=hidden_dim,
            num_steps=reg_processor_steps,
        ).to(device)

        self.cross_message = CrossMessageLayer(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,  # edges are pre-encoded from 4D → hidden_dim
            hidden_dim=hidden_dim,
            global_latent_dim=global_latent_dim,
        ).to(device)

        # Cross edge features: проецируем 4D → hidden_dim для использования в cross message
        self.cross_edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            _get_activation("swish"),
        ).to(device)

        self.reg_decoder = RegionalDecoder(
            input_dim=hidden_dim,
            output_dim=self.output_channels,
            hidden_dim=hidden_dim,
        ).to(device)

        # Перенос буферов на device
        self.to(device)

        # Подсчёт параметров
        reg_params = sum(
            p.numel() for n, p in self.named_parameters()
            if not n.startswith("global_model.")
        )
        print(f"\n[DualMesh] Regional module parameters: {reg_params:,}")
        print(f"[DualMesh] Global model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
        print(f"[DualMesh] ROI grid points: {self.n_roi_grid}")
        print(f"[DualMesh] Regional mesh nodes: {self.n_reg_mesh}")

    def forward(self, X: torch.Tensor, attention_threshold=0.0, **kwargs):
        """
        X: (1, N_grid, T*F) — входные данные.

        Returns: (N_grid, output_channels)
        """
        # ── 1. Глобальный прогноз (может быть detached если заморожен) ──
        global_pred = self.global_model(X=X, attention_threshold=attention_threshold, **kwargs)
        # global_pred: (N_grid, output_channels)

        # ── 2. Извлекаем глобальные mesh латенты после encoder ──
        # Прогоняем encoder ещё раз, чтобы получить mesh features
        # (или кешируем — но для простоты прогоняем)
        X_sq = X.squeeze()
        X_preprocessed = self.global_model._preprocess_input(grid_node_features=X_sq)
        encoded = self.global_model.encoder.forward(
            X=X_preprocessed, edge_index=self.global_model.encoding_graph,
        )
        global_grid_latent = encoded[:self.global_model._num_grid_nodes]
        global_mesh_latent = encoded[self.global_model._num_grid_nodes:]

        # ── 3. Regional encoding ──
        # Берём raw features ROI grid точек + их глобальные латенты
        roi_raw = X_sq[self.roi_mask]  # (n_roi, T*F)
        roi_global_latent = global_grid_latent[self.roi_mask]  # (n_roi, D)
        roi_input = torch.cat([roi_raw, roi_global_latent], dim=-1)

        reg_mesh_features = self.reg_encoder(
            grid_features=roi_input,
            edge_index=self.reg_encoding_edges,
            n_mesh_nodes=self.n_reg_mesh,
        )

        # ── 4. Cross-message: global mesh → regional mesh ──
        cross_edge_attr = self.cross_edge_encoder(self.cross_edge_features)
        global_mesh_detached = global_mesh_latent.detach()  # не хотим градиенты через глобальный encoder
        reg_mesh_features = self.cross_message(
            h_global=global_mesh_detached,
            h_regional=reg_mesh_features,
            cross_edge_index=self.cross_edge_index,
            cross_edge_attr=cross_edge_attr,
            n_global=self.n_global_mesh,
        )

        # ── 5. Regional processing ──
        reg_mesh_features = self.reg_processor(
            x=reg_mesh_features,
            edge_index=self.reg_processing_edges,
            edge_attr_raw=self.reg_processing_edge_features,
        )

        # ── 6. Regional decoding → correction for ROI grid ──
        roi_correction = self.reg_decoder(
            mesh_features=reg_mesh_features,
            edge_index=self.reg_decoding_edges,
            n_grid_nodes=self.n_roi_grid,
        )

        # ── 7. Combine: global + regional correction ──
        output = global_pred.clone()
        output[self.roi_mask] = output[self.roi_mask] + roi_correction

        return output
