
# src/assimilation/nudging.py
# Minimal, self-contained nudging (Newtonian relaxation) for inference-time DA.

from typing import List, Optional, Sequence, Tuple
import torch

def build_feature_mask(
    all_features: Sequence[str],
    assimilate_features: Sequence[str],
    pred_window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds a boolean mask over the flattened (P * C) channel dimension.

    Args:
        all_features: list of all feature names (length C) in the *prediction order*.
        assimilate_features: which features we nudge toward observations.
        pred_window: P
        device: torch device

    Returns:
        mask_flat: shape [P*C], dtype=bool
    """
    C = len(all_features)
    select = torch.zeros(C, dtype=torch.bool)
    name_to_idx = {n: i for i, n in enumerate(all_features)}
    for name in assimilate_features:
        if name not in name_to_idx:
            continue
        select[name_to_idx[name]] = True
    # repeat across lead times
    mask = select.repeat(pred_window)
    return mask.to(device)

def build_feature_mask_from_indices(
    indices: Sequence[int],
    num_features: int,
    pred_window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Same as build_feature_mask but from integer indices (0..C-1).
    """
    C = int(num_features)
    select = torch.zeros(C, dtype=torch.bool)
    for idx in indices:
        if 0 <= idx < C:
            select[idx] = True
    mask = select.repeat(pred_window)
    return mask.to(device)

def cosine_taper_2d(lon: int, lat: int, border: int) -> torch.Tensor:
    """
    Returns a [lon,lat] taper in [0,1] which is 1 inside and smoothly
    decays to 0 at the outer 'border' cells using a raised cosine (Hann) window.
    If border == 0, returns all-ones.
    """
    if border <= 0:
        return torch.ones(lon, lat)
    import math
    w_lon = torch.ones(lon)
    w_lat = torch.ones(lat)
    for i in range(border):
        a = 0.5 * (1 - math.cos(math.pi * (i + 1) / (border + 1)))
        w_lon[i] *= a
        w_lon[-(i+1)] *= a
        w_lat[i] *= a
        w_lat[-(i+1)] *= a
    taper = torch.ger(w_lon, w_lat)  # outer product -> [lon, lat]
    # Normalize so center ~1 (it already is), but clip to [0,1]
    return taper.clamp(0, 1)

class NudgingAssimilator:
    """
    Applies nudging update on the *model output* using available observations.

        y_a = y_f + alpha * M ⊙ (y_obs - y_f)

    where M is a boolean mask over variables we assimilate (broadcast over space/time),
    and 'alpha' is in (0,1], equivalent to (Δt / τ) for a relaxation time τ.

    All tensors are expected to be:
        - pred_flat: [G, P*C]
        - obs_flat:  [G, P*C]   (aligned with pred_flat)
    """
    def __init__(
        self,
        alpha: float = 0.5,
        feature_mask_flat: Optional[torch.Tensor] = None,  # [P*C] bool
        grid_lon: Optional[int] = None,
        grid_lat: Optional[int] = None,
        pred_window: Optional[int] = None,
        num_features: Optional[int] = None,
        blend_border: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.mask_flat = feature_mask_flat  # bool [P*C] or None
        self.grid_lon = grid_lon
        self.grid_lat = grid_lat
        self.pred_window = pred_window
        self.num_features = num_features
        self.blend_border = int(blend_border)
        self.device = device or torch.device("cpu")

        if self.mask_flat is not None and self.mask_flat.dim() != 1:
            raise ValueError("feature_mask_flat must be 1D [P*C]")
        if self.mask_flat is not None:
            self.mask_flat = self.mask_flat.to(self.device)

        if (self.blend_border > 0) and (self.grid_lon is not None) and (self.grid_lat is not None):
            self._taper = cosine_taper_2d(self.grid_lon, self.grid_lat, self.blend_border).to(self.device)
            # flatten for broadcasting over G
            self._taper_flat = self._taper.reshape(-1, 1)  # [G,1]
        else:
            self._taper = None
            self._taper_flat = None

    @torch.no_grad()
    def apply(self, pred_flat: torch.Tensor, obs_flat: torch.Tensor) -> torch.Tensor:
        """
        pred_flat: [G, P*C]
        obs_flat:  [G, P*C]
        Returns:   [G, P*C] (analysis)
        """
        if pred_flat.shape != obs_flat.shape:
            raise ValueError(f"pred and obs must have the same shape, got {tuple(pred_flat.shape)} vs {tuple(obs_flat.shape)}")

        if self.mask_flat is None:
            # default: assimilate all variables
            mask = torch.ones(pred_flat.shape[-1], dtype=torch.bool, device=pred_flat.device)
        else:
            mask = self.mask_flat.to(pred_flat.device)

        # Broadcast mask to [G, P*C]
        mask_bc = mask.view(1, -1)

        update = (obs_flat - pred_flat)
        if self._taper_flat is not None:
            update = update * self._taper_flat  # attenuate near borders

        analysis = pred_flat + self.alpha * update * mask_bc
        return analysis
