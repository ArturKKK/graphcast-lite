# Запрос на ревью архитектуры DualMesh для регионального уточнения погодного прогноза

## Контекст проекта

GraphCast-lite — GNN-модель для прогноза погоды на сетке. Глобальная модель (InteractionNet v2, 5.9M параметров, 12 MP шагов, hidden=256, Swish) обучена на ERA5 2010–2021 при разрешении 0.7° (64×32 → 512×256 multires). Работает по принципу GraphCast: Grid → Mesh (encoding) → Mesh processing (message passing) → Mesh → Grid (decoding).

Для региона Красноярск (lat 50–60, lon 83–98) мы делали fine-tuning глобальной модели (freeze6 — заморозили первые 6 из 12 MP слоёв процессора). Результат freeze6:
- **t2m RMSE: 0.96°C**, skill 75.82%, ACC 0.721 (по 200 samples, горизонт +6ч)

Мы попробовали архитектуру **DualMesh** — добавить отдельный региональный меш поверх frozen глобальной модели, чтобы выучить поправку в ROI. После 4 циклов обучения (каждый раз исправляя баги) DualMesh дал **хуже** freeze6.

**Вопрос: Это архитектурный тупик, или в коде есть фундальная ошибка?**

---

## Архитектура DualMesh (словами)

```
Input X: (1, 133279, 19*2=38)  — 133K grid nodes, 19 variables, obs_window=2

1. Global model (frozen, no_grad):
   - global_pred = global_model(X)  → (133279, 19)
   - Повторно прогоняем encoder для получения global_mesh_latent (40962, 256)
     и global_grid_latent (133279, 256)

2. Regional Encoder:
   - Берём ROI grid точки (~2501 из 133K): roi_raw (2501, 38) + roi_global_latent (2501, 256)
   - Конкатенация → (2501, 294)
   - MLP (294→256→256) → (2501, 256)
   - scatter_mean по encoding_edges (mesh-centric KNN k=4) → (2329, 256)
   Каждый из 2329 mesh-узлов агрегирует фичи от 4 ближайших grid-точек

3. Cross-Message (Global → Regional):
   - Для каждого reg mesh-узла берём k=3 ближайших global mesh-узла
   - MLP(global_latent[sender] || reg_feat[receiver] || edge_feat) → message
   - scatter_mean → LayerNorm(residual + aggregated)
   - Только Global→Regional, обратного нет (global frozen)

4. Regional Processor:
   - 4 шага InteractionNet (shared weights) на рёбрах регионального меша
   - Меш = level-8 icosphere, отфильтрованный в ROI+buffer (2329 вершин)
   - edge_dim=4 → энкодер → hidden_dim=256

5. Regional Decoder:
   - scatter_mean по decoding_edges (grid-centric KNN k=3) → (2501, 256)
   - MLP(256→256→19) → roi_correction (2501, 19)
   - **Последний Linear слой zero-init (weight=0, bias=0)**

6. Combine:
   - correction_full = zeros_like(global_pred)  # (133279, 19)
   - correction_full[roi_mask] = roi_correction
   - output = global_pred + correction_full
```

Обучаемые параметры: 938,515 (regional modules only). Global model полностью заморожена.

Loss считается **ТОЛЬКО по ROI точкам**:
```python
loss = weighted_mse_loss(out[:, roi_mask, :], y[:, roi_mask, :], None)
```

---

## Результаты

### DualMesh (после всех фиксов, 30 эпох, lr=5e-4):
```
REGIONAL overall: RMSE=0.074204 | Skill=71.31% | ACC=0.9691

Per-channel (физические единицы, +6h):
  t2m:    1.16°C    (freeze6: 0.96°C)  — ХУЖЕ
  10u:    0.41 m/s  (freeze6: 0.43)    — чуть лучше
  10v:    0.37 m/s  (freeze6: 0.35)    — чуть хуже
  msl:    0.67 Pa   (freeze6: 0.63)    — хуже
  z@500:  skill 79.70%
```

### Наблюдения при обучении:
- Loss падает крайне медленно: 0.007042 → 0.006822 за 7 эпох
- Regional correction по сути ≈ 0 — модель не учится добавлять полезную поправку
- Обучение сходится к тому, что correction ≈ 0, и output ≈ global_pred

---

## Исправленные баги (4 итерации)

1. **Gradient dilution (~48x)**: Loss считался по ВСЕМ 133K точкам, но regional module меняет только 2501 → 97.9% loss от frozen global. Fix: ROI-only loss.

2. **In-place combine**: `global_pred.clone()` + in-place `[roi_mask] = correction`. Fix: functional `zeros_like + addition`.

3. **40% dead mesh nodes**: radius-based encoding оставлял 941/2329 mesh-узлов без входящих рёбер (буферная зона далеко от grid). Fix: mesh-centric KNN (k=4) для encoding, grid-centric KNN (k=3) для decoding → 0 dead nodes гарантировано.

4. **Random init decoder**: Последний Linear слой имел случайные Xavier веса → correction начинается как шум, модель тратит эпохи чтобы обнулить его. Fix: zero-init weight+bias.

---

## Моя гипотеза (может быть не верна)

Два `scatter_mean` (encoder + decoder) уничтожают пространственную высокочастотную информацию. Residual (ошибка глобальной модели) — сложный и неоднородный паттерн, который нельзя передать через усреднение по 3–4 соседям. Mesh как промежуточное представление для коррекции не работает, потому что:
- Encoder: средуем 4 grid-фичи → теряем локальную структуру
- Decoder: средуем 3 mesh-фичи → размазываем коррекцию

---

## Полный код

### src/dual_mesh.py (700 строк)

```python
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
    n_global = len(meshes[min(level, 6)].vertices)  # 40962 для level >= 6

    # Создаём маску: в ROI + buffer
    lat_ok = (lats_deg >= lat_min - buffer_deg) & (lats_deg <= lat_max + buffer_deg)
    lon_ok = (lons_deg >= lon_min - buffer_deg) & (lons_deg <= lon_max + buffer_deg)
    in_roi = lat_ok & lon_ok

    # Вершины уровня level, которых нет в уровне 6
    new_only = np.zeros(len(vertices), dtype=bool)
    new_only[n_global:] = True

    mask = in_roi & new_only
    kept_indices = np.where(mask)[0]

    if len(kept_indices) == 0:
        raise ValueError(
            f"No regional mesh vertices found in ROI {roi} with buffer={buffer_deg}°. "
            f"Try increasing buffer_deg or level."
        )

    # Переиндексируем
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
    """
    from scipy.spatial import cKDTree
    from src.utils import lat_lon_deg_to_spherical, spherical_to_cartesian

    def to_xyz(lats, lons):
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x, y, z], axis=-1)

    global_xyz = to_xyz(global_lats, global_lons)
    reg_xyz = to_xyz(reg_lats, reg_lons)

    tree = cKDTree(global_xyz)
    distances, global_indices = tree.query(reg_xyz, k=k)

    n_reg = len(reg_lats)
    reg_indices = np.repeat(np.arange(n_reg), k)
    glob_indices = global_indices.flatten()

    # Bidirectional
    senders = np.concatenate([glob_indices, reg_indices])
    receivers = np.concatenate([reg_indices, glob_indices])

    cross_edge_index = torch.tensor(
        np.stack([senders, receivers], axis=0), dtype=torch.int64
    )

    # Edge features
    all_lats = np.concatenate([global_lats, reg_lats])
    all_lons = np.concatenate([global_lons, reg_lons])

    n_global = len(global_lats)
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
    k_encode: int = 4,
    k_decode: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Строит рёбра Grid(ROI)→RegionalMesh и RegionalMesh→Grid(ROI).

    Encoding (Grid→Mesh): mesh-centric KNN k=4 — каждый mesh-узел получает данные.
    Decoding (Mesh→Grid): grid-centric KNN k=3 — каждая grid-точка получает прогноз.
    """
    from scipy.spatial import cKDTree

    lat_min, lat_max, lon_min, lon_max = roi

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

    def to_xyz(lats, lons):
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x, y, z], axis=-1)

    reg_xyz = to_xyz(reg_lats, reg_lons)
    roi_xyz = to_xyz(roi_lats, roi_lons)
    n_mesh = len(reg_lats)

    # Encoding: mesh-centric KNN
    tree_grid = cKDTree(roi_xyz)
    k_enc = min(k_encode, n_roi)
    _, grid_neighbors = tree_grid.query(reg_xyz, k=k_enc)
    if grid_neighbors.ndim == 1:
        grid_neighbors = grid_neighbors[:, None]

    enc_grid_idx = grid_neighbors.flatten().tolist()
    enc_mesh_idx = np.repeat(np.arange(n_mesh), k_enc).tolist()

    encoding_edges = torch.tensor(
        np.stack([enc_grid_idx, enc_mesh_idx], axis=0), dtype=torch.int64
    )

    # Decoding: grid-centric KNN
    tree_mesh = cKDTree(reg_xyz)
    k_dec = min(k_decode, n_mesh)
    _, mesh_neighbors = tree_mesh.query(roi_xyz, k=k_dec)
    if mesh_neighbors.ndim == 1:
        mesh_neighbors = mesh_neighbors[:, None]

    dec_mesh_idx = mesh_neighbors.flatten().tolist()
    dec_grid_idx = np.repeat(np.arange(n_roi), k_dec).tolist()

    decoding_edges = torch.tensor(
        np.stack([dec_mesh_idx, dec_grid_idx], axis=0), dtype=torch.int64
    )

    print(f"[RegionalGridMesh] ROI grid points: {n_roi}")
    print(f"  Encoding edges (grid→reg_mesh): {encoding_edges.shape[1]}")
    print(f"  Decoding edges (reg_mesh→grid): {decoding_edges.shape[1]}")

    return roi_mask, encoding_edges, decoding_edges


# ─── 4. Cross-Message модуль ──────────────────────────────────────────

class CrossMessageLayer(nn.Module):
    """Однонаправленный обмен: Global → Regional."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 global_latent_dim: int = None, activation: str = "swish"):
        super().__init__()
        act = _get_activation(activation)
        g_dim = global_latent_dim if global_latent_dim is not None else node_dim

        self.g2r_edge_mlp = nn.Sequential(
            nn.Linear(g_dim + node_dim + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm_reg = PygLayerNorm(node_dim, mode="node")

    def forward(self, h_global, h_regional, cross_edge_index, cross_edge_attr, n_global):
        n_reg = h_regional.shape[0]
        E = cross_edge_index.shape[1]
        half_E = E // 2

        g2r_senders = cross_edge_index[0, :half_E]
        g2r_receivers = cross_edge_index[1, :half_E]
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
    """InteractionNet процессор для регионального меша. Shared weights."""

    def __init__(self, node_dim: int, raw_edge_dim: int = 4,
                 hidden_dim: int = 256, num_steps: int = 4,
                 activation: str = "swish"):
        super().__init__()
        act = _get_activation(activation)
        self.edge_encoder = nn.Sequential(
            nn.Linear(raw_edge_dim, node_dim), act,
        )
        self.step = InteractionNetLayer(
            node_dim=node_dim, edge_dim=node_dim, hidden_dim=hidden_dim,
            activation=activation, use_layer_norm=True,
        )
        self.num_steps = num_steps

    def forward(self, x, edge_index, edge_attr_raw):
        edge_attr = self.edge_encoder(edge_attr_raw)
        for _ in range(self.num_steps):
            x, edge_attr = self.step(x, edge_index, edge_attr)
        return x


# ─── 6. Regional Encoder / Decoder ───────────────────────────────────

class RegionalEncoder(nn.Module):
    """Энкодер: Grid(ROI) → Regional Mesh через scatter_mean."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, grid_features, edge_index, n_mesh_nodes):
        x = self.mlp(grid_features)
        grid_msg = x[edge_index[0]]
        mesh_features = scatter(grid_msg, edge_index[1], dim=0,
                                dim_size=n_mesh_nodes, reduce="mean")
        return mesh_features


class RegionalDecoder(nn.Module):
    """Декодер: Regional Mesh → Grid(ROI). Zero-init output."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Correction starts at exactly zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mesh_features, edge_index, n_grid_nodes):
        mesh_msg = mesh_features[edge_index[0]]
        grid_agg = scatter(mesh_msg, edge_index[1], dim=0,
                           dim_size=n_grid_nodes, reduce="mean")
        return self.mlp(grid_agg)


# ─── 7. DualMeshModel — главная обёртка ──────────────────────────────

class DualMeshModel(nn.Module):
    """
    Output = global_pred + regional_correction (only in ROI).
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
        self.output_channels = global_model.num_features

        # 1. Regional mesh
        self.reg_mesh, reg_lats, reg_lons = create_regional_mesh(
            roi=roi, level=reg_mesh_level, buffer_deg=reg_mesh_buffer,
        )
        self.n_reg_mesh = len(reg_lats)

        # 2. Processing graph
        reg_edge_index = torch.tensor(
            get_edges_from_faces(self.reg_mesh.faces), dtype=torch.int64
        )
        reg_edge_features = _compute_mesh_edge_features(
            mesh_node_lats=reg_lats, mesh_node_longs=reg_lons,
            edge_index=reg_edge_index.numpy(),
        )
        self.register_buffer("reg_processing_edges", reg_edge_index)
        self.register_buffer("reg_processing_edge_features", reg_edge_features)

        # 3. Cross-edges: global mesh ↔ regional mesh
        global_mesh_lats = global_model._mesh_nodes_lat
        global_mesh_lons = global_model._mesh_nodes_lon
        n_global_mesh = global_model._num_mesh_nodes

        cross_edge_index, cross_edge_features = build_cross_edges(
            global_lats=global_mesh_lats, global_lons=global_mesh_lons,
            reg_lats=reg_lats, reg_lons=reg_lons, k=cross_k,
        )
        self.register_buffer("cross_edge_index", cross_edge_index)
        self.register_buffer("cross_edge_features", cross_edge_features)
        self.n_global_mesh = n_global_mesh

        # 4. Grid(ROI) ↔ RegMesh edges
        roi_mask, enc_edges, dec_edges = build_regional_grid_mesh_edges(
            grid_lats=grid_lats, grid_lons=grid_lons,
            reg_lats=reg_lats, reg_lons=reg_lons, roi=roi,
        )
        self.register_buffer("roi_mask", torch.tensor(roi_mask, dtype=torch.bool))
        self.register_buffer("reg_encoding_edges", enc_edges)
        self.register_buffer("reg_decoding_edges", dec_edges)
        self.n_roi_grid = int(roi_mask.sum())

        # 5. Regional modules (trainable)
        total_feature_size = self.n_features * self.obs_window
        global_latent_dim = global_model.encoder.output_dim
        reg_enc_input_dim = total_feature_size + global_latent_dim

        self.reg_encoder = RegionalEncoder(
            input_dim=reg_enc_input_dim, hidden_dim=hidden_dim,
        ).to(device)

        self.reg_processor = RegionalProcessor(
            node_dim=hidden_dim, raw_edge_dim=4, hidden_dim=hidden_dim,
            num_steps=reg_processor_steps,
        ).to(device)

        self.cross_message = CrossMessageLayer(
            node_dim=hidden_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim,
            global_latent_dim=global_latent_dim,
        ).to(device)

        self.cross_edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), _get_activation("swish"),
        ).to(device)

        self.reg_decoder = RegionalDecoder(
            input_dim=hidden_dim, output_dim=self.output_channels,
            hidden_dim=hidden_dim,
        ).to(device)

        self.to(device)

    def forward(self, X: torch.Tensor, attention_threshold=0.0, **kwargs):
        assert X.shape[0] == 1, f"Batch size must be 1, got {X.shape[0]}"

        # 1. Global prediction (no_grad)
        with torch.no_grad():
            global_pred = self.global_model(X=X, attention_threshold=attention_threshold, **kwargs)

            # 2. Извлекаем global mesh латенты
            X_sq = X[0]  # (G, T*F)
            X_preprocessed = self.global_model._preprocess_input(grid_node_features=X_sq)
            encoded = self.global_model.encoder.forward(
                X=X_preprocessed, edge_index=self.global_model.encoding_graph,
            )
            global_grid_latent = encoded[:self.global_model._num_grid_nodes]
            global_mesh_latent = encoded[self.global_model._num_grid_nodes:]

        global_pred = global_pred.detach()

        # 3. Regional encoding
        roi_raw = X_sq[self.roi_mask]        # (n_roi, T*F)
        roi_global_latent = global_grid_latent[self.roi_mask]  # (n_roi, D)
        roi_input = torch.cat([roi_raw, roi_global_latent], dim=-1)

        reg_mesh_features = self.reg_encoder(
            grid_features=roi_input,
            edge_index=self.reg_encoding_edges,
            n_mesh_nodes=self.n_reg_mesh,
        )

        # 4. Cross-message: global mesh → regional mesh
        cross_edge_attr = self.cross_edge_encoder(self.cross_edge_features)
        reg_mesh_features = self.cross_message(
            h_global=global_mesh_latent,
            h_regional=reg_mesh_features,
            cross_edge_index=self.cross_edge_index,
            cross_edge_attr=cross_edge_attr,
            n_global=self.n_global_mesh,
        )

        # 5. Regional processing
        reg_mesh_features = self.reg_processor(
            x=reg_mesh_features,
            edge_index=self.reg_processing_edges,
            edge_attr_raw=self.reg_processing_edge_features,
        )

        # 6. Regional decoding → correction
        roi_correction = self.reg_decoder(
            mesh_features=reg_mesh_features,
            edge_index=self.reg_decoding_edges,
            n_grid_nodes=self.n_roi_grid,
        )

        # 7. Combine
        correction_full = torch.zeros_like(global_pred)
        correction_full[self.roi_mask] = roi_correction
        output = global_pred + correction_full

        return output
```

### scripts/train_dual_mesh.py — training loop (ключевые части)

```python
def train_dual_epoch(model, dataloader, optimizer, device, lat_weights=None,
                     use_residual=False, check_grads=False):
    model.train()
    model.global_model.eval()  # Global всегда eval

    roi_mask = model.roi_mask  # (G,) bool tensor

    total_loss = 0
    n_batches = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.squeeze(0) if y.dim() == 4 else y

        N, G, feat_dim = X.shape
        C = feat_dim // model.obs_window

        # Берём первый шаг из target
        total_target_steps = y.shape[-1] // C
        if total_target_steps > 1:
            y_step0 = y.view(N, G, total_target_steps, C)[:, :, 0, :]
        else:
            y_step0 = y

        optimizer.zero_grad()
        pred = model(X)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)

        if use_residual:
            X_reshaped = X.view(N, G, model.obs_window, C)
            X_last = X_reshaped[:, :, -1, :]
            out = X_last + pred
        else:
            out = pred

        # Loss ТОЛЬКО по ROI
        out_roi = out[:, roi_mask, :]
        y_roi = y_step0[:, roi_mask, :]
        loss = weighted_mse_loss(out_roi, y_roi, None)

        loss.backward()

        # Gradient check (первая эпоха)
        if check_grads and n_batches == 0:
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gn = p.grad.norm().item()
                    if gn > 0:
                        print(f"  {name}: grad_norm={gn:.6f}")
                        break
            else:
                print("  WARNING: все градиенты нулевые!")

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
```

---

## Конкретные вопросы

1. **Есть ли баги в коде**, из-за которых gradient flow пробит или correction не может обучаться?

2. **Правильная ли архитектура** для задачи residual learning поверх frozen global model? Может ли scatter_mean в encoder/decoder быть бутылочным горлышком?

3. **Cross-message делается ОДИН раз перед processor.** Должен ли он быть interleaved (на каждом шаге processor)?

4. **Global mesh латенты detached** (через `no_grad`). Это значит, что cross-message получает «мёртвые» латенты без актуальной информации о текущем input. Это проблема?

5. **Decoder scatter_mean**: для каждой grid-точки берём mean от 3 ближайших mesh-узлов. Но коррекция — это точечная операция (каждая grid-точка нужна своя). Scatter_mean размазывает. Стоит ли делать decoder без aggregation (как attention или bilinear interpolation)?

6. **Какую альтернативную архитектуру** вы бы предложили для regional refinement поверх frozen global GNN, если DualMesh не работает?

7. **Freeze6 fine-tuning дал 75.82% skill** на том же датасете. Почему DualMesh с 938K params (vs ~3M trainable in freeze6) не может хотя бы достичь того же уровня?
