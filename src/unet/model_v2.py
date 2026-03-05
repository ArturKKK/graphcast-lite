"""
WeatherUNet V2 — serious architecture for regional weather prediction.

Key upgrades over V1:
  1. Residual conv blocks (gradient highway)
  2. Self-attention at bottleneck (global context, 5×7=35 tokens — nearly free)
  3. 4 encoder levels instead of 3 (wider receptive field)
  4. Channel attention (SE blocks) — learns which channels matter per-level
  5. Spectral conv in bottleneck (captures periodic patterns)

Input:  (B, obs_window * C, H, W)
Output: (B, C, H, W) — predicted delta
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────── Building blocks ───────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel importance."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class ResConvBlock(nn.Module):
    """Conv3x3 → GN → GELU → Conv3x3 → GN → GELU + residual + SE."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        g = min(num_groups, out_ch)
        # Make sure out_ch is divisible by g
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x) + self.skip(x))


class SelfAttention2D(nn.Module):
    """Multi-head self-attention on spatial positions."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C) where N=H*W
        tokens = self.norm(tokens)

        qkv = self.qkv(tokens).reshape(B, H * W, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)

        # Residual + reshape back
        out = tokens + out
        return out.transpose(1, 2).reshape(B, C, H, W)


class SpectralConv2d(nn.Module):
    """Fourier layer — learns in frequency domain. 
    Captures periodic patterns (pressure waves, diurnal cycles)."""
    def __init__(self, in_ch, out_ch, modes_h=4, modes_w=4):
        super().__init__()
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1 / (in_ch * out_ch)
        self.weights_re = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w))
        self.weights_im = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w))

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, C_in, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C_in, H, W//2+1)

        mh = min(self.modes_h, H)
        mw = min(self.modes_w, x_ft.shape[-1])

        # Multiply low-frequency modes with learned weights
        out_ft = torch.zeros(B, self.weights_re.shape[1], H, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)

        w = torch.complex(self.weights_re[:, :, :mh, :mw],
                           self.weights_im[:, :, :mh, :mw])
        out_ft[:, :, :mh, :mw] = torch.einsum("bihw,iohw->bohw", x_ft[:, :, :mh, :mw], w)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = F.pad(x, [0, dw, 0, dh])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────── Main Model ───────────

class WeatherUNetV2(nn.Module):
    """
    U-Net V2 for regional weather prediction.

    Architecture (41×61 grid example):
      Level 0: 41×61  →  f channels
      Level 1: 20×30  →  2f channels
      Level 2: 10×15  →  4f channels
      Level 3:  5×7   →  8f channels
      Bottleneck: Self-Attention + SpectralConv at 5×7 (35 tokens)
      Then mirror decoder with skip connections.

    Parameters
    ----------
    in_channels : int
        obs_window * n_features
    out_channels : int
        n_features (model predicts delta)
    base_filters : int
        Channels at first level (default 64)
    attn_heads : int
        Number of attention heads in bottleneck (default 4)
    spectral_modes : int
        Number of Fourier modes in spectral layer (default 4)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 base_filters: int = 64, attn_heads: int = 4,
                 spectral_modes: int = 4):
        super().__init__()
        f = base_filters

        # Encoder
        self.inc = ResConvBlock(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        # Bottleneck: attention + spectral
        self.bottleneck_attn = SelfAttention2D(f * 8, heads=attn_heads)
        self.bottleneck_spectral = SpectralConv2d(f * 8, f * 8,
                                                   modes_h=spectral_modes,
                                                   modes_w=spectral_modes)
        self.bottleneck_mix = ResConvBlock(f * 8 * 2, f * 8)  # concat attn + spectral

        # Decoder
        self.up1 = Up(f * 8 + f * 4, f * 4)
        self.up2 = Up(f * 4 + f * 2, f * 2)
        self.up3 = Up(f * 2 + f, f)

        # Output: 1×1 conv
        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck: parallel attention + spectral → concat → mix
        b_attn = self.bottleneck_attn(x4)
        b_spec = self.bottleneck_spectral(x4)
        b = self.bottleneck_mix(torch.cat([b_attn, b_spec], dim=1))

        x = self.up1(b, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.out_conv(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
