"""Losses for neural postprocessing.

См. docs/postprocessing_rfc.md §3.1a, §3.5.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Smooth-L1 / Huber. Robust к outliers ISD-Lite (особенно ws_obs)."""
    return F.smooth_l1_loss(pred, target, beta=delta, reduction="mean")


def hybrid_wind_loss(
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor,
    alpha: float = 0.5,
    ws_clamp: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Hybrid loss: MSE(u,v) + α · ws_true · (1 − cos Δθ).

    Множитель — ws_true (магнитуда истинного ветра), с clamp снизу,
    чтобы низковетреные часы (шумная direction) не доминировали в angle-части.
    """
    mse = F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)

    ws_true = torch.sqrt(u_true.pow(2) + v_true.pow(2)).clamp(min=ws_clamp)
    ws_pred = torch.sqrt(u_pred.pow(2) + v_pred.pow(2)).clamp(min=eps)

    cos_dtheta = (u_pred * u_true + v_pred * v_true) / (ws_pred * ws_true + eps)
    cos_dtheta = cos_dtheta.clamp(min=-1.0, max=1.0)
    angle_term = (ws_true * (1.0 - cos_dtheta)).mean()

    return mse + alpha * angle_term


def crps_gaussian(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS для нормального предиктива (Gneiting 2007).

    CRPS = σ · [ z·(2·Φ(z) − 1) + 2·φ(z) − 1/√π ],  где z = (y − μ)/σ.
    """
    sigma = log_sigma.exp().clamp(min=1e-6)
    z = (y - mu) / sigma
    # standard normal pdf/cdf
    phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean()


def compute_total_loss(
    out: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    probabilistic: bool = False,
    w_t2m: float = 1.0,
    w_wind: float = 1.0,
    huber_delta: float = 1.0,
    wind_alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Считает совокупный loss + per-task для логов."""
    losses: Dict[str, torch.Tensor] = {}

    if probabilistic:
        l_t = crps_gaussian(out["t2m_mu"], out["t2m_log_sigma"], targets["t2m"])
        # CRPS для (u, v) считаем независимо по компонентам (упрощение, EMOS-стандарт)
        l_w_u = crps_gaussian(out["wind_mu"][:, 0], out["wind_log_sigma"][:, 0], targets["u10"])
        l_w_v = crps_gaussian(out["wind_mu"][:, 1], out["wind_log_sigma"][:, 1], targets["v10"])
        l_w = l_w_u + l_w_v
    else:
        l_t = huber_loss(out["t2m"], targets["t2m"], delta=huber_delta)
        l_w = hybrid_wind_loss(
            out["u10"], out["v10"], targets["u10"], targets["v10"], alpha=wind_alpha
        )

    total = w_t2m * l_t + w_wind * l_w
    losses["loss"] = total
    losses["loss_t2m"] = l_t.detach()
    losses["loss_wind"] = l_w.detach()
    return losses
