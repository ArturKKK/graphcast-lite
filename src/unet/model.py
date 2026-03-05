"""
U-Net model for regional weather prediction.

Input:  (B, obs_window * C, H, W) — 2D grid
Output: (B, C, H, W) — predicted delta for 1 step (residual learning)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv3x3 → BN → GELU → Conv3x3 → BN → GELU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample → cat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match exactly after upsampling
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = nn.functional.pad(x, [0, dw, 0, dh])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class WeatherUNet(nn.Module):
    """
    U-Net for regional weather prediction on a regular grid.
    
    Parameters
    ----------
    in_channels : int
        obs_window * n_features (e.g. 2 * 19 = 38)
    out_channels : int
        n_features (e.g. 19) — model predicts delta
    base_filters : int
        Number of filters in first layer (doubled at each level)
    """
    def __init__(self, in_channels: int, out_channels: int, base_filters: int = 64):
        super().__init__()
        f = base_filters
        
        self.inc = DoubleConv(in_channels, f)         # 61×41 → f
        self.down1 = Down(f, f * 2)                    # 30×20 → 2f
        self.down2 = Down(f * 2, f * 4)                # 15×10 → 4f
        self.down3 = Down(f * 4, f * 8)                # 7×5  → 8f
        
        self.up1 = Up(f * 8 + f * 4, f * 4)           # 15×10
        self.up2 = Up(f * 4 + f * 2, f * 2)           # 30×20
        self.up3 = Up(f * 2 + f, f)                    # 61×41
        
        self.out_conv = nn.Conv2d(f, out_channels, 1)  # 1×1 conv → C channels
        
    def forward(self, x):
        """x: (B, in_channels, H, W) → (B, out_channels, H, W)"""
        x1 = self.inc(x)       # (B, f, H, W)
        x2 = self.down1(x1)    # (B, 2f, H/2, W/2)
        x3 = self.down2(x2)    # (B, 4f, H/4, W/4)
        x4 = self.down3(x3)    # (B, 8f, H/8, W/8)
        
        x = self.up1(x4, x3)   # (B, 4f, H/4, W/4)
        x = self.up2(x, x2)    # (B, 2f, H/2, W/2)
        x = self.up3(x, x1)    # (B, f, H, W)
        
        return self.out_conv(x) # (B, C, H, W)
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
