"""Generate multiple interpolation visualization styles for weather map.

Reads forecast.json grid_points and produces PNG overlays in different styles.
"""

import json
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ── Config ──
FORECAST_JSON = Path(__file__).parent.parent / "website" / "static" / "forecast.json"
OUTPUT_DIR = Path(__file__).parent.parent / "viz_variants"
STEP_IDX = 0  # first forecast step (+6h)
DPI = 150
FINE_RES = 200  # interpolation grid resolution

# RdYlBu-like colormap (matches website)
RDYLBU_COLORS = [
    (49/255, 54/255, 149/255),   # cold blue
    (69/255, 117/255, 180/255),
    (116/255, 173/255, 209/255),
    (171/255, 217/255, 233/255),
    (254/255, 224/255, 144/255),
    (253/255, 174/255, 97/255),
    (244/255, 109/255, 67/255),
    (215/255, 48/255, 39/255),   # hot red
]
cmap_rdylbu = LinearSegmentedColormap.from_list("rdylbu_custom", RDYLBU_COLORS, N=256)

# Alternative colormaps
cmap_turbo = plt.cm.turbo
cmap_coolwarm = plt.cm.coolwarm
cmap_viridis = plt.cm.viridis


def load_data():
    with open(FORECAST_JSON) as f:
        data = json.load(f)
    pts = data["grid_points"]
    lats = np.array([p["lat"] for p in pts])
    lons = np.array([p["lon"] for p in pts])
    temps = np.array([p["steps"][STEP_IDX]["t"] for p in pts])
    winds = np.array([p["steps"][STEP_IDX]["ws"] for p in pts])
    wind_dirs = np.array([p["steps"][STEP_IDX]["wd"] for p in pts])
    precips = np.array([p["steps"][STEP_IDX]["pr"] for p in pts])
    return lats, lons, temps, winds, wind_dirs, precips, data


def interpolate_field(lats, lons, values, method="cubic"):
    lat_fine = np.linspace(lats.min(), lats.max(), FINE_RES)
    lon_fine = np.linspace(lons.min(), lons.max(), FINE_RES)
    lon_grid, lat_grid = np.meshgrid(lon_fine, lat_fine)
    grid = griddata((lons, lats), values, (lon_grid, lat_grid), method=method)
    return lon_fine, lat_fine, lon_grid, lat_grid, grid


def add_krasnoyarsk_marker(ax):
    """Add a star marker for Krasnoyarsk city center."""
    ax.plot(92.87, 56.01, marker="*", color="black", markersize=12,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.annotate("Красноярск", (92.87, 56.01), fontsize=8,
                xytext=(5, 5), textcoords="offset points",
                color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


def style_1_smooth_contourf(lats, lons, temps, data):
    """Style 1: Smooth contour fill — classic weather map look."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    cf = ax.contourf(lon_grid, lat_grid, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.85)
    # contour lines
    cs = ax.contour(lon_grid, lat_grid, t_grid, levels=levels[::2],
                    colors="black", linewidths=0.3, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f°")

    plt.colorbar(cf, ax=ax, label="Температура, °C", shrink=0.8, pad=0.02)
    ax.scatter(lons, lats, c="black", s=3, alpha=0.3, zorder=5)
    add_krasnoyarsk_marker(ax)
    ax.set_xlabel("Долгота, °E")
    ax.set_ylabel("Широта, °N")
    ax.set_title("Стиль 1: Контурная заливка (contourf)")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style1_contourf.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def style_2_heatmap_smooth(lats, lons, temps, data):
    """Style 2: Smooth heatmap with gaussian-like interpolation."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2

    im = ax.imshow(t_grid, extent=[lon_fine.min(), lon_fine.max(),
                                    lat_fine.min(), lat_fine.max()],
                   origin="lower", cmap=cmap_rdylbu, vmin=vmin, vmax=vmax,
                   aspect="auto", alpha=0.9, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Температура, °C", shrink=0.8, pad=0.02)
    add_krasnoyarsk_marker(ax)
    ax.set_xlabel("Долгота, °E")
    ax.set_ylabel("Широта, °N")
    ax.set_title("Стиль 2: Тепловая карта (heatmap, bilinear)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style2_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def style_3_contour_with_wind(lats, lons, temps, winds, wind_dirs, data):
    """Style 3: Temperature fill + wind barbs overlay."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    cf = ax.contourf(lon_grid, lat_grid, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.8)
    plt.colorbar(cf, ax=ax, label="Температура, °C", shrink=0.8, pad=0.02)

    # Wind arrows (subsample for clarity)
    step = max(1, len(lats) // 40)
    u = winds[::step] * np.sin(np.radians(wind_dirs[::step]))
    v = winds[::step] * np.cos(np.radians(wind_dirs[::step]))
    ax.quiver(lons[::step], lats[::step], u, v,
              color="black", alpha=0.6, scale=80, width=0.003, headwidth=3)

    add_krasnoyarsk_marker(ax)
    ax.set_xlabel("Долгота, °E")
    ax.set_ylabel("Широта, °N")
    ax.set_title("Стиль 3: Температура + ветер (стрелки)")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style3_temp_wind.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def style_4_turbo_cmap(lats, lons, temps, data):
    """Style 4: Turbo colormap — more vivid/scientific look."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 25)

    cf = ax.contourf(lon_grid, lat_grid, t_grid, levels=levels,
                     cmap=cmap_turbo, extend="both", alpha=0.85)
    cs = ax.contour(lon_grid, lat_grid, t_grid, levels=levels[::3],
                    colors="white", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f°", colors="white")

    plt.colorbar(cf, ax=ax, label="Температура, °C", shrink=0.8, pad=0.02)
    add_krasnoyarsk_marker(ax)
    ax.set_xlabel("Долгота, °E")
    ax.set_ylabel("Широта, °N")
    ax.set_title("Стиль 4: Turbo colormap")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style4_turbo.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def style_5_dark_theme(lats, lons, temps, winds, wind_dirs, data):
    """Style 5: Dark theme — matches the website's dark UI."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2
    levels = np.linspace(vmin, vmax, 20)

    cf = ax.contourf(lon_grid, lat_grid, t_grid, levels=levels,
                     cmap=cmap_rdylbu, extend="both", alpha=0.8)
    cs = ax.contour(lon_grid, lat_grid, t_grid, levels=levels[::2],
                    colors="white", linewidths=0.3, alpha=0.3)

    cbar = plt.colorbar(cf, ax=ax, label="°C", shrink=0.8, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    # Wind arrows on dark bg
    step = max(1, len(lats) // 40)
    u = winds[::step] * np.sin(np.radians(wind_dirs[::step]))
    v = winds[::step] * np.cos(np.radians(wind_dirs[::step]))
    ax.quiver(lons[::step], lats[::step], u, v,
              color="white", alpha=0.4, scale=80, width=0.003, headwidth=3)

    # Krasnoyarsk on dark
    ax.plot(92.87, 56.01, marker="*", color="#4fc3f7", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.annotate("Красноярск", (92.87, 56.01), fontsize=9,
                xytext=(5, 5), textcoords="offset points",
                color="#4fc3f7", fontweight="bold")

    ax.set_xlabel("Долгота, °E", color="#8899aa")
    ax.set_ylabel("Широта, °N", color="#8899aa")
    ax.tick_params(colors="#8899aa")
    ax.set_title("Стиль 5: Тёмная тема + ветер", color="white", fontsize=13)
    for spine in ax.spines.values():
        spine.set_color("#2a3a4e")
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style5_dark.png", dpi=DPI, bbox_inches="tight",
                facecolor="#0f1923")
    plt.close(fig)


def style_6_transparent_overlay(lats, lons, temps, data):
    """Style 6: Transparent PNG overlay (for Leaflet ImageOverlay).
    No axes, no labels — just the colored field with alpha."""
    lon_fine, lat_fine, lon_grid, lat_grid, t_grid = interpolate_field(lats, lons, temps)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_alpha(0)
    ax.set_axis_off()

    vmin, vmax = np.floor(temps.min() / 5) * 5 - 2, np.ceil(temps.max() / 5) * 5 + 2

    # Create RGBA image with alpha channel
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap_rdylbu(norm(t_grid))
    rgba[..., 3] = 0.6  # set uniform transparency
    # Mask NaN areas
    nan_mask = np.isnan(t_grid)
    rgba[nan_mask, 3] = 0

    ax.imshow(rgba, extent=[lon_fine.min(), lon_fine.max(),
                            lat_fine.min(), lat_fine.max()],
              origin="lower", aspect="auto", interpolation="bilinear")

    fig.savefig(OUTPUT_DIR / "style6_overlay.png", dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Also save the bounds for Leaflet
    bounds = {
        "south": float(lat_fine.min()),
        "north": float(lat_fine.max()),
        "west": float(lon_fine.min()),
        "east": float(lon_fine.max()),
    }
    with open(OUTPUT_DIR / "overlay_bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)


def style_7_multi_panel(lats, lons, temps, winds, precips, data):
    """Style 7: Multi-panel — temp + wind + pressure side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, values, title, cmap, label in [
        (axes[0], temps, "Температура", cmap_rdylbu, "°C"),
        (axes[1], winds, "Скорость ветра", plt.cm.YlOrRd, "м/с"),
        (axes[2], precips, "Давление", plt.cm.Blues_r, "мм рт.ст."),
    ]:
        _, _, lon_grid, lat_grid, grid = interpolate_field(lats, lons, values)
        vmin = np.floor(values.min()) - 1
        vmax = np.ceil(values.max()) + 1
        cf = ax.contourf(lon_grid, lat_grid, grid, levels=15,
                         cmap=cmap, extend="both", alpha=0.85)
        plt.colorbar(cf, ax=ax, label=label, shrink=0.8)
        ax.plot(92.87, 56.01, marker="*", color="black", markersize=10,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("°E")
        ax.set_ylabel("°N")
        ax.set_aspect("auto")

    fig.suptitle("Прогноз по Красноярскому краю (+6ч)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "style7_multipanel.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    lats, lons, temps, winds, wind_dirs, precips, data = load_data()

    print(f"Grid: {len(lats)} points, lat [{lats.min():.1f}, {lats.max():.1f}], "
          f"lon [{lons.min():.1f}, {lons.max():.1f}]")
    print(f"Temp range: {temps.min():.1f} .. {temps.max():.1f} °C")
    print(f"Wind range: {winds.min():.1f} .. {winds.max():.1f} m/s")

    print("Generating style 1: contourf...")
    style_1_smooth_contourf(lats, lons, temps, data)

    print("Generating style 2: heatmap...")
    style_2_heatmap_smooth(lats, lons, temps, data)

    print("Generating style 3: temp + wind arrows...")
    style_3_contour_with_wind(lats, lons, temps, winds, wind_dirs, data)

    print("Generating style 4: turbo colormap...")
    style_4_turbo_cmap(lats, lons, temps, data)

    print("Generating style 5: dark theme...")
    style_5_dark_theme(lats, lons, temps, winds, wind_dirs, data)

    print("Generating style 6: transparent overlay (for Leaflet)...")
    style_6_transparent_overlay(lats, lons, temps, data)

    print("Generating style 7: multi-panel...")
    style_7_multi_panel(lats, lons, temps, winds, precips, data)

    print(f"\n✅ All 7 styles saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
