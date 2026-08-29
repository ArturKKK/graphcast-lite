"""Multi-task residual MLP for neural postprocessing.

Архитектура: shared trunk + per-target heads (t2m, wind=(u,v)).
Выход — *residual* поверх GNN-предсказания. Опционально probabilistic head (μ, log σ).

См. docs/postprocessing_rfc.md §3.1a.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


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


class StationLeadAwareResidualMLP(nn.Module):
    """v2 residual postprocessor with station embedding + lead-FiLM.

    Архитектура:
      • [features | station_emb] → Linear → GELU → LayerNorm
      • FiLM(lead_norm): γ, β = MLP(lead_norm) → (γ⊙h + β) → Dropout
      • Linear → GELU → Dropout → heads (residual поверх GNN-baseline)

    Параметры:
      feature_dim:   D без station emb (lead_norm уже внутри features).
      num_stations:  размер словаря станций.
      station_emb_dim: 16 по умолчанию.
      hidden:        размеры скрытых слоёв ствола.
      dropout:       dropout по стволу.
      probabilistic: использовать Gaussian-головы (μ, log σ).
    """

    def __init__(
        self,
        feature_dim: int,
        num_stations: int,
        station_emb_dim: int = 16,
        hidden: Optional[List[int]] = None,
        dropout: float = 0.1,
        probabilistic: bool = False,
        film_hidden: int = 32,
    ):
        super().__init__()
        hidden = hidden or [128, 128]
        self.probabilistic = probabilistic
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        nn.init.normal_(self.station_emb.weight, mean=0.0, std=0.05)

        in_dim = feature_dim + station_emb_dim
        first_h = hidden[0]
        self.fc1 = nn.Linear(in_dim, first_h)
        self.ln1 = nn.LayerNorm(first_h)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # FiLM на lead_norm (скаляр) → (γ, β) для первого слоя
        self.film = nn.Sequential(
            nn.Linear(1, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * first_h),
        )
        # γ инициализируем ≈ 1, β ≈ 0 → нейтральная модуляция в старте
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # хвост ствола
        tail_layers: List[nn.Module] = []
        prev = first_h
        for h in hidden[1:]:
            tail_layers.append(nn.Linear(prev, h))
            tail_layers.append(nn.GELU())
            if dropout > 0:
                tail_layers.append(nn.Dropout(dropout))
            prev = h
        self.tail = nn.Sequential(*tail_layers) if tail_layers else nn.Identity()
        trunk_out = prev

        if probabilistic:
            self.head_t2m = ProbabilisticHead(trunk_out, 1)
            self.head_wind = ProbabilisticHead(trunk_out, 2)
        else:
            self.head_t2m = nn.Linear(trunk_out, 1)
            self.head_wind = nn.Linear(trunk_out, 2)

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
        self,
        features: torch.Tensor,
        station_idx: torch.Tensor,
        lead_norm: torch.Tensor,
        gnn_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        emb = self.station_emb(station_idx)  # (B, E)
        x = torch.cat([features, emb], dim=-1)
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)

        film = self.film(lead_norm.unsqueeze(-1))  # (B, 2*H)
        gamma, beta = film.chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta  # zero-init → identity at start
        h = self.drop(h)
        h = self.tail(h)

        out: Dict[str, torch.Tensor] = {}
        if self.probabilistic:
            h_t = self.head_t2m(h)
            mu = h_t["mu"].squeeze(-1)
            if gnn_targets is not None and "t2m" in gnn_targets:
                mu = gnn_targets["t2m"] + mu
            out["t2m_mu"] = mu
            out["t2m_log_sigma"] = h_t["log_sigma"].squeeze(-1)
            h_w = self.head_wind(h)
            mu_uv = h_w["mu"]
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                mu_uv = base + mu_uv
            out["wind_mu"] = mu_uv
            out["wind_log_sigma"] = h_w["log_sigma"]
        else:
            delta_t = self.head_t2m(h).squeeze(-1)
            if gnn_targets is not None and "t2m" in gnn_targets:
                out["t2m"] = gnn_targets["t2m"] + delta_t
            else:
                out["t2m"] = delta_t
            out["t2m_delta"] = delta_t
            delta_uv = self.head_wind(h)
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                uv = base + delta_uv
            else:
                uv = delta_uv
            out["u10"] = uv[:, 0]
            out["v10"] = uv[:, 1]
            out["wind_delta"] = delta_uv
        return out


class StationLeadBiasResidualMLP(nn.Module):
    """v3 postprocessor: v2 + explicit per-station additive bias head.

    Мотивация: v2 даёт частичный residual через MLP+FiLM, но не имеет чисто
    станционного аддитивного параметра. В результате остаточный bias на уровне
    конкретной станции (-1.9 → -1.6 в Новосибирске) не вычищается полностью.

    Изменения относительно v2:
      • bias_emb: nn.Embedding(num_stations, 3) — фиксированный per-station
        аддитивный сдвиг для (t2m, u10, v10), складывается с residual выходом;
      • поддерживается всё то же, что в v2: features+station_emb, lead-FiLM,
        residual поверх GNN baseline.

    Архитектура:
      ŷ_t2m = gnn_t2m + Δ_mlp(features, station, lead) + b_station[t2m]
      ŷ_u   = gnn_u   + Δ_mlp + b_station[u]
      ŷ_v   = gnn_v   + Δ_mlp + b_station[v]
    """

    def __init__(
        self,
        feature_dim: int,
        num_stations: int,
        station_emb_dim: int = 32,
        hidden: Optional[List[int]] = None,
        dropout: float = 0.1,
        probabilistic: bool = False,
        film_hidden: int = 32,
        bias_init_std: float = 0.0,
    ):
        super().__init__()
        hidden = hidden or [192, 192, 128]
        self.probabilistic = probabilistic
        # Режим БЕЗ ПРИВЯЗКИ К СТАНЦИИ: station_emb_dim=0 убирает и вложение, и
        # добавочную голову смещения. Тогда модель описывает площадку только
        # признаками — широтой, долготой, высотой, разностью высот со рельефом
        # модели, — и её можно применить к станции, которой она никогда не
        # видела. С вложением это невозможно в принципе: для новой станции
        # попросту нет строки, и никакое приближение её не заменит.
        #
        # Вопрос не теоретический: если постпроцессор работает только там, где у
        # него есть годы наблюдений, поставить его на новую площадку нельзя.
        self.station_free = station_emb_dim <= 0
        if self.station_free:
            self.station_emb = None
            self.bias_emb = None
        else:
            self.station_emb = nn.Embedding(num_stations, station_emb_dim)
            nn.init.normal_(self.station_emb.weight, mean=0.0, std=0.05)

            # PER-STATION ADDITIVE BIAS HEAD (the key v3 innovation)
            # Init at zero so it does no harm at epoch 0; trains via gradient.
            self.bias_emb = nn.Embedding(num_stations, 3)
            if bias_init_std > 0.0:
                nn.init.normal_(self.bias_emb.weight, mean=0.0, std=bias_init_std)
            else:
                nn.init.zeros_(self.bias_emb.weight)

        in_dim = feature_dim + max(0, station_emb_dim)
        first_h = hidden[0]
        self.fc1 = nn.Linear(in_dim, first_h)
        self.ln1 = nn.LayerNorm(first_h)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.film = nn.Sequential(
            nn.Linear(1, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * first_h),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        tail_layers: List[nn.Module] = []
        prev = first_h
        for h in hidden[1:]:
            tail_layers.append(nn.Linear(prev, h))
            tail_layers.append(nn.GELU())
            if dropout > 0:
                tail_layers.append(nn.Dropout(dropout))
            prev = h
        self.tail = nn.Sequential(*tail_layers) if tail_layers else nn.Identity()
        trunk_out = prev

        if probabilistic:
            self.head_t2m = ProbabilisticHead(trunk_out, 1)
            self.head_wind = ProbabilisticHead(trunk_out, 2)
        else:
            self.head_t2m = nn.Linear(trunk_out, 1)
            self.head_wind = nn.Linear(trunk_out, 2)

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
        self,
        features: torch.Tensor,
        station_idx: torch.Tensor,
        lead_norm: torch.Tensor,
        gnn_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.station_free:
            x = features
        else:
            x = torch.cat([features, self.station_emb(station_idx)], dim=-1)
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)

        film = self.film(lead_norm.unsqueeze(-1))
        gamma, beta = film.chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = self.drop(h)
        h = self.tail(h)

        # per-station additive bias (t2m, u, v)
        if self.station_free:
            b_t = features.new_zeros(features.shape[0])
            b_uv = features.new_zeros(features.shape[0], 2)
        else:
            b = self.bias_emb(station_idx)  # (B, 3)
            b_t = b[:, 0]
            b_uv = b[:, 1:3]

        out: Dict[str, torch.Tensor] = {}
        if self.probabilistic:
            h_t = self.head_t2m(h)
            mu = h_t["mu"].squeeze(-1) + b_t
            if gnn_targets is not None and "t2m" in gnn_targets:
                mu = gnn_targets["t2m"] + mu
            out["t2m_mu"] = mu
            out["t2m_log_sigma"] = h_t["log_sigma"].squeeze(-1)
            h_w = self.head_wind(h)
            mu_uv = h_w["mu"] + b_uv
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                mu_uv = base + mu_uv
            out["wind_mu"] = mu_uv
            out["wind_log_sigma"] = h_w["log_sigma"]
        else:
            delta_t = self.head_t2m(h).squeeze(-1) + b_t
            if gnn_targets is not None and "t2m" in gnn_targets:
                out["t2m"] = gnn_targets["t2m"] + delta_t
            else:
                out["t2m"] = delta_t
            out["t2m_delta"] = delta_t
            delta_uv = self.head_wind(h) + b_uv
            if gnn_targets is not None and "u10" in gnn_targets and "v10" in gnn_targets:
                base = torch.stack([gnn_targets["u10"], gnn_targets["v10"]], dim=-1)
                uv = base + delta_uv
            else:
                uv = delta_uv
            out["u10"] = uv[:, 0]
            out["v10"] = uv[:, 1]
            out["wind_delta"] = delta_uv
        return out
