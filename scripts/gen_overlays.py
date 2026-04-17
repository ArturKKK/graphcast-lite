"""Generate transparent overlay PNGs for Leaflet for each variable × time step."""

import json
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

FORECAST_JSON = Path(__file__).parent.parent / "website" / "static" / "forecast.json"
OUTPUT_DIR = Path(__file__).parent.parent / "website" / "static" / "overlays"
FINE_RES = 200

# Temperature: vivid blue→cyan→yellow→orange→red
RDYLBU = LinearSegmentedColormap.from_list("rdylbu", [
    (30/255, 40/255, 150/255),   # deep blue (cold)
    (50/255, 100/255, 190/255),
    (80/255, 165/255, 220/255),
    (140/255, 215/255, 230/255),
    (255/255, 235/255, 120/255), # warm yellow
    (255/255, 170/255, 70/255),
    (240/255, 90/255, 50/255),
    (200/255, 30/255, 30/255),   # hot red
], N=256)

# Wind: green→yellow→orange→red→dark red
WIND_CMAP = LinearSegmentedColormap.from_list("wind", [
    (0.3, 0.75, 0.3),   # calm green
    (0.6, 0.9, 0.2),    # light
    (1.0, 0.85, 0.1),   # moderate yellow
    (1.0, 0.5, 0.1),    # fresh orange
    (0.85, 0.15, 0.15), # strong red
    (0.55, 0.0, 0.0),   # gale dark red
], N=256)

# Precipitation: transparent→light blue→blue→dark blue
PRECIP_CMAP = LinearSegmentedColormap.from_list("precip", [
    (0.7, 0.85, 1.0, 0.0),    # transparent
    (0.35, 0.6, 0.95, 0.4),   # light blue
    (0.15, 0.4, 0.85, 0.65),  # medium blue
    (0.05, 0.2, 0.7, 0.8),    # dark blue
    (0.02, 0.05, 0.45, 0.9),  # very dark blue
], N=256)

# Pressure: distinct teal→green→yellow→orange gradient
PRESSURE_CMAP = LinearSegmentedColormap.from_list("pressure", [
    (0.1, 0.3, 0.5),    # low pressure = dark teal
    (0.15, 0.55, 0.55),
    (0.3, 0.75, 0.5),   # medium = green
    (0.7, 0.85, 0.3),   # yellow-green
    (0.95, 0.75, 0.2),  # high = orange
    (0.95, 0.5, 0.15),  # very high = dark orange
], N=256)


def load_forecast():
    with open(FORECAST_JSON) as f:
        data = json.load(f)
    pts = data["grid_points"]
    n_steps = len(pts[0]["steps"])
    lats = np.array([p["lat"] for p in pts])
    lons = np.array([p["lon"] for p in pts])
    return lats, lons, pts, n_steps, data


def interpolate(lats, lons, values, method="cubic"):
    lat_f = np.linspace(lats.min(), lats.max(), FINE_RES)
    lon_f = np.linspace(lons.min(), lons.max(), FINE_RES)
    lon_g, lat_g = np.meshgrid(lon_f, lat_f)
    grid = griddata((lons, lats), values, (lon_g, lat_g), method=method)
    return grid


def save_overlay(grid, cmap, norm, path, alpha=0.65):
    """Save transparent PNG without axes."""
    rgba = cmap(norm(grid))
    # Apply alpha
    if rgba.shape[-1] == 4:
        rgba[..., 3] *= alpha
    else:
        a = np.full(grid.shape + (1,), alpha)
        rgba = np.concatenate([rgba[..., :3], a], axis=-1)
    nan_mask = np.isnan(grid)
    rgba[nan_mask, 3] = 0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.imshow(rgba, origin="lower", aspect="auto", interpolation="bilinear")
    fig.savefig(path, dpi=100, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading forecast.json...")
    lats, lons, pts, n_steps, data = load_forecast()
    print(f"  {len(pts)} points, {n_steps} steps")

    # Compute global ranges for consistent coloring across steps
    all_t = [p["steps"][s]["t"] for p in pts for s in range(n_steps)]
    all_ws = [p["steps"][s]["ws"] for p in pts for s in range(n_steps)]
    all_p = [p["steps"][s]["p"] for p in pts for s in range(n_steps)]
    all_pr = [p["steps"][s]["pr"] for p in pts for s in range(n_steps)]

    t_min = np.floor(min(all_t) / 5) * 5 - 2
    t_max = np.ceil(max(all_t) / 5) * 5 + 2
    ws_max = max(8, max(all_ws))
    p_max = max(1, max(all_p))
    pr_min, pr_max = min(all_pr), max(all_pr)

    norms = {
        "temp": mcolors.Normalize(vmin=t_min, vmax=t_max),
        "wind": mcolors.Normalize(vmin=0, vmax=ws_max),
        "precip": mcolors.Normalize(vmin=0, vmax=p_max),
        "pressure": mcolors.Normalize(vmin=pr_min, vmax=pr_max),
    }

    # Save wind quiver data as JSON for the JS side
    wind_data = {}

    for step in range(n_steps):
        print(f"  Step {step} (+{(step+1)*6}h)...")
        t_vals = np.array([p["steps"][step]["t"] for p in pts])
        ws_vals = np.array([p["steps"][step]["ws"] for p in pts])
        wd_vals = np.array([p["steps"][step]["wd"] for p in pts])
        p_vals = np.array([p["steps"][step]["p"] for p in pts])
        pr_vals = np.array([p["steps"][step]["pr"] for p in pts])

        t_grid = interpolate(lats, lons, t_vals)
        ws_grid = interpolate(lats, lons, ws_vals)
        p_grid = interpolate(lats, lons, p_vals, method="linear")
        pr_grid = interpolate(lats, lons, pr_vals)

        save_overlay(t_grid, RDYLBU, norms["temp"],
                     OUTPUT_DIR / f"temp_{step}.png", alpha=0.75)
        save_overlay(ws_grid, WIND_CMAP, norms["wind"],
                     OUTPUT_DIR / f"wind_{step}.png", alpha=0.7)
        save_overlay(p_grid, PRECIP_CMAP, norms["precip"],
                     OUTPUT_DIR / f"precip_{step}.png", alpha=0.65)
        save_overlay(pr_grid, PRESSURE_CMAP, norms["pressure"],
                     OUTPUT_DIR / f"pressure_{step}.png", alpha=0.7)

        # Subsample wind arrows for quiver overlay
        sub = max(1, len(pts) // 50)
        arrows = []
        for i in range(0, len(pts), sub):
            arrows.append({
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "ws": float(ws_vals[i]),
                "wd": float(wd_vals[i]),
            })
        wind_data[str(step)] = arrows

    # Save bounds
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())],
                    [float(lats.max()), float(lons.max())]],
        "n_steps": n_steps,
        "ranges": {
            "temp": [float(t_min), float(t_max)],
            "wind": [0, float(ws_max)],
            "precip": [0, float(p_max)],
            "pressure": [float(pr_min), float(pr_max)],
        },
        "wind_arrows": wind_data,
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"\nDone! {n_steps * 4} overlays + meta.json in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
