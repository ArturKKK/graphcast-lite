"""Multi-task residual MLP for neural postprocessing.

Архитектура: shared trunk + per-target heads (t2m, wind=(u,v)).
Выход — *residual* поверх GNN-предсказания. Опционально probabilistic head (μ, log σ).

См. docs/postprocessing_rfc.md §3.1a.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for i, h in enumerate(hidden):
        layers.append(nn.Linear(prev, h))
        layers.append(nn.GELU())
        if i == 0:
            layers.append(nn.LayerNorm(h))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


class ProbabilisticHead(nn.Module):
    """Gaussian head: возвращает (μ, log σ). Используется для CRPS-loss."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.log_sigma = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"mu": self.mu(x), "log_sigma": self.log_sigma(x).clamp(min=-6.0, max=4.0)}


class MultiTaskResidualMLP(nn.Module):
    """Residual postprocessor: y_pred = y_gnn + Δ, где Δ = MLP(features).

    Параметры:
      feature_dim: размерность входного вектора фич (см. dataset.py).
      hidden: размеры слоёв общего ствола, по умолчанию [128, 128].
      dropout: dropout общий ствол.
      probabilistic: если True — головы выдают (μ, log σ) вместо точечного значения.
      gnn_indices: словарь {target_name: index_in_features} — позиция raw GNN-канала
                   во входе, чтобы добавить как residual baseline.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: Optional[List[int]] = None,
        dropout: float = 0.1,
        probabilistic: bool = False,
        gnn_indices: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        hidden = hidden or [128, 128]
        self.trunk = _mlp(feature_dim, hidden, dropout)
        trunk_out = hidden[-1]
        self.probabilistic = probabilistic
        self.gnn_indices = gnn_indices or {}

        if probabilistic:
            self.head_t2m = ProbabilisticHead(trunk_out, 1)
            self.head_wind = ProbabilisticHead(trunk_out, 2)
        else:
            self.head_t2m = nn.Linear(trunk_out, 1)
            self.head_wind = nn.Linear(trunk_out, 2)

        # инициализация: residual-голова стартует около нуля,
        # чтобы модель в первой итерации копировала raw GNN.
        for m in [self.head_t2m, self.head_wind]:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, ProbabilisticHead):
                nn.init.zeros_(m.mu.weight)
                nn.init.zeros_(m.mu.bias)
                nn.init.constant_(m.log_sigma.weight, 0.0)
                nn.init.constant_(m.log_sigma.bias, 0.0)

    def forward(
        self, features: torch.Tensor, gnn_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """features: (B, D); gnn_targets: {'t2m': (B,), 'u10': (B,), 'v10': (B,)} (denormalised)."""
        h = self.trunk(features)
        out: Dict[str, torch.Tensor] = {}

        # --- t2m ---
        if self.probabilistic:
            h_t2m = self.head_t2m(h)
            mu = h_t2m["mu"].squeeze(-1)
            if gnn_targets is not None and "t2m" in gnn_targets:
                mu = gnn_targets["t2m"] + mu
            out["t2m_mu"] = mu
            out["t2m_log_sigma"] = h_t2m["log_sigma"].squeeze(-1)
        else:
            delta_t2m = self.head_t2m(h).squeeze(-1)
            if gnn_targets is not None and "t2m" in gnn_targets:
                out["t2m"] = gnn_targets["t2m"] + delta_t2m
            else:
                out["t2m"] = delta_t2m
            out["t2m_delta"] = delta_t2m

        # --- wind (u, v) ---
        if self.probabilistic:
            h_w = self.head_wind(h)
            mu_uv = h_w["mu"]  # (B, 2)
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                mu_uv = base + mu_uv
            out["wind_mu"] = mu_uv  # (B, 2)
            out["wind_log_sigma"] = h_w["log_sigma"]  # (B, 2)
        else:
            delta_uv = self.head_wind(h)  # (B, 2)
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                uv = base + delta_uv
            else:
                uv = delta_uv
            out["u10"] = uv[:, 0]
            out["v10"] = uv[:, 1]
            out["wind_delta"] = delta_uv

        return out
