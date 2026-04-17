"""Enhanced Style 5 (Dark) — temperature + wind speed arrows + precipitation zones.

Generates multiple variants of the dark theme visualization:
- v1: base (temperature + wind direction/speed arrows)
- v2: + precipitation overlay (hatching for rain/snow zones)
- v3: multi-layer switchable panels (temp, wind, precip, pressure)
"""

import json
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# ── Config ──
FORECAST_JSON = Path(__file__).parent.parent / "website" / "static" / "forecast.json"
OUTPUT_DIR = Path(__file__).parent.parent / "viz_variants"
DPI = 150
FINE_RES = 200

# Color scheme
BG_COLOR = "#0f1923"
GRID_COLOR = "#2a3a4e"
TEXT_COLOR = "#8899aa"
ACCENT_COLOR = "#4fc3f7"

RDYLBU_COLORS = [
    (49/255, 54/255, 149/255),
    (69/255, 117/255, 180/255),
    (116/255, 173/255, 209/255),
    (171/255, 217/255, 233/255),
    (254/255, 224/255, 144/255),
    (253/255, 174/255, 97/255),
    (244/255, 109/255, 67/255),
    (215/255, 48/255, 39/255),
]
cmap_rdylbu = LinearSegmentedColormap.from_list("rdylbu_custom", RDYLBU_COLORS, N=256)

# Wind speed colormap (greens → yellows → reds)
WIND_COLORS = [
    (0.7, 0.9, 0.7),   # calm - light green
    (0.4, 0.8, 0.4),   # light
    (1.0, 0.9, 0.3),   # moderate - yellow
    (1.0, 0.6, 0.2),   # fresh - orange
    (0.9, 0.2, 0.2),   # strong - red
]
cmap_wind = LinearSegmentedColormap.from_list("wind_speed", WIND_COLORS, N=256)


def load_data(step_idx=0):
    with open(FORECAST_JSON) as f:
        data = json.load(f)
    pts = data["grid_points"]
    lats = np.array([p["lat"] for p in pts])
    lons = np.array([p["lon"] for p in pts])
    step_data = {
        "t": np.array([p["steps"][step_idx]["t"] for p in pts]),
        "ws": np.array([p["steps"][step_idx]["ws"] for p in pts]),
        "wd": np.array([p["steps"][step_idx]["wd"] for p in pts]),
        "wg": np.array([p["steps"][step_idx]["wg"] for p in pts]),
        "p": np.array([p["steps"][step_idx]["p"] for p in pts]),
        "pt": [p["steps"][step_idx]["pt"] for p in pts],
        "pr": np.array([p["steps"][step_idx]["pr"] for p in pts]),
    }
    return lats, lons, step_data


def interpolate(lats, lons, values, method="cubic"):
    lat_fine = np.linspace(lats.min(), lats.max(), FINE_RES)
    lon_fine = np.linspace(lons.min(), lons.max(), FINE_RES)
    lon_grid, lat_grid = np.meshgrid(lon_fine, lat_fine)
    grid = griddata((lons, lats), values, (lon_grid, lat_grid), method=method)
    return lon_fine, lat_fine, lon_grid, lat_grid, grid


def setup_dark_ax(fig, ax, title=""):
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlabel("Долгота, °E", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Широта, °N", color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    if title:
        ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_aspect("auto")


def add_city(ax, dark=True):
    color = ACCENT_COLOR if dark else "black"
    edge = "white"
    ax.plot(92.87, 56.01, marker="*", color=color, markersize=14,
            markeredgecolor=edge, markeredgewidth=0.8, zorder=10)
    ax.annotate("Красноярск", (92.87, 56.01), fontsize=9,
                xytext=(5, 5), textcoords="offset points",
                color=color, fontweight="bold")


def dark_colorbar(fig, ax, mappable, label="", shrink=0.8):
    cbar = fig.colorbar(mappable, ax=ax, label=label, shrink=shrink, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")
    return cbar


# ─────────────────────────────────────────────────────────
# V1: Temperature + wind arrows scaled by speed
# ─────────────────────────────────────────────────────────
def render_v1(lats, lons, sd):
    """Dark theme: temp contours + arrows colored & sized by wind speed."""
    lon_f, lat_f, lon_g, lat_g, t_grid = interpolate(lats, lons, sd["t"])

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_dark_ax(fig, ax, "Температура + ветер (скорость → цвет и длина стрелки)")

    vmin = np.floor(sd["t"].min() / 5) * 5 - 2
    vmax = np.ceil(sd["t"].max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    cf = ax.contourf(lon_g, lat_g, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.75)
    ax.contour(lon_g, lat_g, t_grid, levels=levels[::2],
               colors="white", linewidths=0.25, alpha=0.25)

    dark_colorbar(fig, ax, cf, label="°C")

    # Wind arrows — length proportional to speed, color = speed
    step = max(1, len(lats) // 50)
    ws = sd["ws"][::step]
    wd = sd["wd"][::step]
    u = ws * np.sin(np.radians(wd))
    v = ws * np.cos(np.radians(wd))
    ln = lons[::step]
    lt = lats[::step]

    # Normalize wind speed for coloring
    ws_norm = mcolors.Normalize(vmin=0, vmax=max(8, ws.max()))
    arrow_colors = cmap_wind(ws_norm(ws))

    # Scale factor: longer arrows for stronger wind
    q = ax.quiver(ln, lt, u, v, color=arrow_colors,
                  scale=60, width=0.004, headwidth=3.5, headlength=4,
                  alpha=0.85, zorder=6)

    # Wind speed legend (small inset)
    ax_inset = fig.add_axes([0.02, 0.12, 0.015, 0.3])
    cb_wind = matplotlib.colorbar.ColorbarBase(
        ax_inset, cmap=cmap_wind, norm=ws_norm, orientation='vertical')
    cb_wind.set_label("Ветер, м/с", color="white", fontsize=7)
    cb_wind.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(plt.getp(cb_wind.ax.axes, 'yticklabels'), color="white")

    add_city(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style5v1_wind_speed.png", dpi=DPI,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print("  ✓ v1 saved: style5v1_wind_speed.png")


# ─────────────────────────────────────────────────────────
# V2: Temperature + wind + precipitation overlay
# ─────────────────────────────────────────────────────────
def render_v2(lats, lons, sd):
    """Dark theme: temp + wind + precip zones (hatching)."""
    lon_f, lat_f, lon_g, lat_g, t_grid = interpolate(lats, lons, sd["t"])
    _, _, _, _, p_grid = interpolate(lats, lons, sd["p"], method="linear")

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_dark_ax(fig, ax, "Температура + ветер + осадки")

    vmin = np.floor(sd["t"].min() / 5) * 5 - 2
    vmax = np.ceil(sd["t"].max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    # Temperature base
    cf = ax.contourf(lon_g, lat_g, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.7)
    ax.contour(lon_g, lat_g, t_grid, levels=levels[::2],
               colors="white", linewidths=0.2, alpha=0.2)
    dark_colorbar(fig, ax, cf, label="°C")

    # Precipitation overlay — hatched contours
    # Determine precip type by temperature: t < 0 → snow, t > 2 → rain, else mixed
    precip_levels = [0.1, 0.5, 1.0, 2.0, 4.0]

    # Rain zones (where t > 1°C and precip > 0.1)
    rain_mask = np.where((t_grid > 1) & (~np.isnan(p_grid)), p_grid, 0)
    if np.nanmax(rain_mask) > 0.1:
        cs_rain = ax.contourf(lon_g, lat_g, rain_mask, levels=precip_levels,
                              colors='none', hatches=['..', '//', '///', 'xxx'],
                              alpha=0)
        # Add blue-tinted overlay for rain areas
        ax.contourf(lon_g, lat_g, rain_mask, levels=[0.1, 100],
                    colors=[(0.2, 0.4, 0.9)], alpha=0.15)
        ax.contour(lon_g, lat_g, rain_mask, levels=[0.1],
                   colors=[(0.3, 0.5, 1.0)], linewidths=1.2, linestyles='--', alpha=0.6)

    # Snow zones (where t < 1°C and precip > 0.1)
    snow_mask = np.where((t_grid <= 1) & (~np.isnan(p_grid)), p_grid, 0)
    if np.nanmax(snow_mask) > 0.1:
        cs_snow = ax.contourf(lon_g, lat_g, snow_mask, levels=precip_levels,
                              colors='none', hatches=['..', '**', '***', '****'],
                              alpha=0)
        # White-tinted overlay for snow areas
        ax.contourf(lon_g, lat_g, snow_mask, levels=[0.1, 100],
                    colors=[(0.8, 0.85, 0.95)], alpha=0.12)
        ax.contour(lon_g, lat_g, snow_mask, levels=[0.1],
                   colors=[(0.7, 0.8, 0.95)], linewidths=1.0, linestyles=':', alpha=0.5)

    # Wind arrows
    step = max(1, len(lats) // 50)
    ws = sd["ws"][::step]
    wd = sd["wd"][::step]
    u = ws * np.sin(np.radians(wd))
    v = ws * np.cos(np.radians(wd))
    ws_norm = mcolors.Normalize(vmin=0, vmax=max(8, ws.max()))
    arrow_colors = cmap_wind(ws_norm(ws))

    ax.quiver(lons[::step], lats[::step], u, v, color=arrow_colors,
              scale=60, width=0.004, headwidth=3.5, headlength=4,
              alpha=0.8, zorder=6)

    # Legend for precipitation
    legend_items = []
    legend_items.append(mpatches.Patch(facecolor=(0.2, 0.4, 0.9, 0.3),
                                       edgecolor=(0.3, 0.5, 1.0, 0.8),
                                       linestyle='--', label='Дождь'))
    legend_items.append(mpatches.Patch(facecolor=(0.8, 0.85, 0.95, 0.3),
                                       edgecolor=(0.7, 0.8, 0.95, 0.7),
                                       linestyle=':', label='Снег'))
    leg = ax.legend(handles=legend_items, loc='lower left', fontsize=8,
                    facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                    labelcolor="white", framealpha=0.8)

    # Wind speed legend
    ax_inset = fig.add_axes([0.02, 0.15, 0.015, 0.25])
    cb_wind = matplotlib.colorbar.ColorbarBase(
        ax_inset, cmap=cmap_wind, norm=ws_norm, orientation='vertical')
    cb_wind.set_label("м/с", color="white", fontsize=7)
    cb_wind.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(plt.getp(cb_wind.ax.axes, 'yticklabels'), color="white")

    add_city(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style5v2_wind_precip.png", dpi=DPI,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print("  ✓ v2 saved: style5v2_wind_precip.png")


# ─────────────────────────────────────────────────────────
# V3: Multi-panel (switchable layers concept) — dark
# ─────────────────────────────────────────────────────────
def render_v3(lats, lons, sd):
    """Dark 2x2 panel: temp, wind speed, precipitation, pressure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG_COLOR)

    fields = [
        ("Температура, °C",   sd["t"],  cmap_rdylbu, None),
        ("Скорость ветра, м/с", sd["ws"], cmap_wind,  (0, max(8, sd["ws"].max()))),
        ("Осадки, мм",        sd["p"],   plt.cm.YlGnBu, (0, max(1, sd["p"].max()))),
        ("Давление, мм рт.ст.", sd["pr"], plt.cm.RdPu_r, None),
    ]

    for ax, (title, values, cmap, vlim) in zip(axes.flat, fields):
        ax.set_facecolor(BG_COLOR)
        lon_f, lat_f, lon_g, lat_g, grid = interpolate(lats, lons, values)

        if vlim:
            vmin, vmax = vlim
        else:
            vmin = np.nanmin(grid)
            vmax = np.nanmax(grid)

        levels = np.linspace(vmin, vmax, 18)
        cf = ax.contourf(lon_g, lat_g, grid, levels=levels,
                         cmap=cmap, extend="both", alpha=0.8)
        ax.contour(lon_g, lat_g, grid, levels=levels[::3],
                   colors="white", linewidths=0.2, alpha=0.2)

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        # City marker
        ax.plot(92.87, 56.01, marker="*", color=ACCENT_COLOR, markersize=10,
                markeredgecolor="white", markeredgewidth=0.5, zorder=10)

        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.set_aspect("auto")

    # Add wind arrows to wind panel
    ax_wind = axes[0, 1]
    step = max(1, len(lats) // 40)
    ws = sd["ws"][::step]
    wd = sd["wd"][::step]
    u = ws * np.sin(np.radians(wd))
    v = ws * np.cos(np.radians(wd))
    ax_wind.quiver(lons[::step], lats[::step], u, v,
                   color="white", alpha=0.5, scale=70, width=0.003,
                   headwidth=3, zorder=6)

    fig.suptitle("Прогноз по Красноярскому краю (+6ч)",
                 color="white", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "style5v3_panels.png", dpi=DPI,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print("  ✓ v3 saved: style5v3_panels.png")


# ─────────────────────────────────────────────────────────
# V4: Combined — temp base + precip + wind + pressure isobars
# ─────────────────────────────────────────────────────────
def render_v4(lats, lons, sd):
    """All-in-one dark map: temp fill + precip zones + wind arrows + pressure isobars."""
    lon_f, lat_f, lon_g, lat_g, t_grid = interpolate(lats, lons, sd["t"])
    _, _, _, _, p_grid = interpolate(lats, lons, sd["p"], method="linear")
    _, _, _, _, pr_grid = interpolate(lats, lons, sd["pr"], method="cubic")

    fig, ax = plt.subplots(figsize=(12, 7))
    setup_dark_ax(fig, ax, "Полная карта: температура · ветер · осадки · давление")

    vmin = np.floor(sd["t"].min() / 5) * 5 - 2
    vmax = np.ceil(sd["t"].max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    # Temperature base
    cf = ax.contourf(lon_g, lat_g, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.65)
    dark_colorbar(fig, ax, cf, label="°C", shrink=0.7)

    # Pressure isobars (thin white dashed)
    pr_levels = np.arange(np.floor(sd["pr"].min()), np.ceil(sd["pr"].max()) + 1, 1)
    cs_pr = ax.contour(lon_g, lat_g, pr_grid, levels=pr_levels,
                       colors=[(0.6, 0.7, 0.9)], linewidths=0.5,
                       linestyles='-', alpha=0.35)
    ax.clabel(cs_pr, inline=True, fontsize=6, fmt="%.0f",
              colors=[(0.6, 0.7, 0.9)])

    # Precipitation zones
    rain_mask = np.where((t_grid > 1) & (~np.isnan(p_grid)), p_grid, 0)
    snow_mask = np.where((t_grid <= 1) & (~np.isnan(p_grid)), p_grid, 0)

    if np.nanmax(rain_mask) > 0.1:
        ax.contourf(lon_g, lat_g, rain_mask, levels=[0.1, 100],
                    colors=[(0.2, 0.4, 0.9)], alpha=0.15)
        ax.contour(lon_g, lat_g, rain_mask, levels=[0.1],
                   colors=[(0.4, 0.6, 1.0)], linewidths=1.5,
                   linestyles='--', alpha=0.6)

    if np.nanmax(snow_mask) > 0.1:
        ax.contourf(lon_g, lat_g, snow_mask, levels=[0.1, 100],
                    colors=[(0.85, 0.88, 0.95)], alpha=0.1)
        ax.contour(lon_g, lat_g, snow_mask, levels=[0.1],
                   colors=[(0.75, 0.82, 0.95)], linewidths=1.0,
                   linestyles=':', alpha=0.5)

    # Wind arrows colored by speed
    step = max(1, len(lats) // 55)
    ws = sd["ws"][::step]
    wd = sd["wd"][::step]
    u = ws * np.sin(np.radians(wd))
    v = ws * np.cos(np.radians(wd))
    ws_norm = mcolors.Normalize(vmin=0, vmax=max(8, ws.max()))
    arrow_colors = cmap_wind(ws_norm(ws))

    ax.quiver(lons[::step], lats[::step], u, v, color=arrow_colors,
              scale=55, width=0.0035, headwidth=3.5, headlength=4,
              alpha=0.85, zorder=6)

    # Wind speed legend
    ax_inset = fig.add_axes([0.02, 0.15, 0.013, 0.22])
    cb_w = matplotlib.colorbar.ColorbarBase(
        ax_inset, cmap=cmap_wind, norm=ws_norm, orientation='vertical')
    cb_w.set_label("м/с", color="white", fontsize=7)
    cb_w.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(plt.getp(cb_w.ax.axes, 'yticklabels'), color="white")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=(0.2, 0.4, 0.9, 0.3),
                       edgecolor=(0.4, 0.6, 1.0), linestyle='--',
                       label='Зона дождя'),
        mpatches.Patch(facecolor=(0.85, 0.88, 0.95, 0.2),
                       edgecolor=(0.75, 0.82, 0.95), linestyle=':',
                       label='Зона снега'),
        mpatches.Patch(facecolor='none', edgecolor=(0.6, 0.7, 0.9, 0.5),
                       linestyle='-', label='Изобары (мм рт.ст.)'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7,
              facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor="white", framealpha=0.85)

    add_city(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style5v4_full.png", dpi=DPI,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print("  ✓ v4 saved: style5v4_full.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading data...")
    lats, lons, sd = load_data(step_idx=0)
    print(f"  {len(lats)} grid points, step +6h")
    print(f"  Temp: {sd['t'].min():.1f}..{sd['t'].max():.1f} °C")
    print(f"  Wind: {sd['ws'].min():.1f}..{sd['ws'].max():.1f} m/s")
    print(f"  Precip: {sd['p'].min():.1f}..{sd['p'].max():.1f} mm")
    print()

    print("Rendering v1: temp + wind speed arrows...")
    render_v1(lats, lons, sd)

    print("Rendering v2: temp + wind + precipitation zones...")
    render_v2(lats, lons, sd)

    print("Rendering v3: 2×2 dark panels...")
    render_v3(lats, lons, sd)

    print("Rendering v4: all-in-one full map...")
    render_v4(lats, lons, sd)

    print("\nDone! Check viz_variants/style5v*.png")
