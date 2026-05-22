"""Generate transparent overlay PNGs for Leaflet for each variable × time step."""

import argparse
import json
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from PIL import Image
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

# Precipitation: a hard transparency cutoff at very low values, then a vivid
# blue ramp so even light rain is visible.
PRECIP_CMAP = LinearSegmentedColormap.from_list("precip", [
    (0.55, 0.78, 1.00, 0.55),  # very light rain (just visible)
    (0.35, 0.60, 0.95, 0.75),  # light blue
    (0.15, 0.40, 0.85, 0.85),  # medium blue
    (0.05, 0.20, 0.70, 0.92),  # dark blue
    (0.02, 0.05, 0.45, 0.95),  # very dark blue
], N=256)

# Precipitation visibility threshold (mm) — below this we keep fully transparent.
PRECIP_VISIBLE_MIN_MM = 0.05

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


def mercator_y(lat_deg):
    """Spherical Mercator Y in radians (Leaflet/EPSG:3857-compatible)."""
    lat_deg = np.clip(lat_deg, -85.05112878, 85.05112878)
    lat_rad = np.deg2rad(lat_deg)
    return np.log(np.tan(np.pi * 0.25 + lat_rad * 0.5))


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
    # NOTE: do NOT pass bbox_inches="tight" — it crops fully transparent
    # borders, which for sparsely-precipitating layers shifts the image
    # relative to the geographic bounds Leaflet pins it to.
    fig.savefig(path, dpi=100, transparent=True, pad_inches=0)
    plt.close(fig)


def save_precip_overlay(grid, norm, path, alpha=0.85,
                        threshold_mm=PRECIP_VISIBLE_MIN_MM):
    """Deprecated grid-based variant kept for backward compatibility — not used.
    See save_precip_overlay_voronoi below for the pixel-exact version."""
    rgba = PRECIP_CMAP(norm(grid))
    rgba[..., 3] *= alpha
    invisible = np.isnan(grid) | (grid < threshold_mm)
    rgba[invisible, 3] = 0

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.imshow(rgba, origin="lower", aspect="auto", interpolation="nearest")
    fig.savefig(path, dpi=100, transparent=True, pad_inches=0)
    plt.close(fig)


def save_precip_overlay_voronoi(lats, lons, values, norm, path,
                                alpha=0.85,
                                threshold_mm=PRECIP_VISIBLE_MIN_MM,
                                px_per_deg_lon=8):
    """Render a pixel-exact Voronoi tiling of precipitation.

    Each output pixel is coloured by the value at its single nearest
    grid point in projected WebMercator space (x=lon_rad, y=mercator_y).
    This matches how Leaflet linearly stretches image overlays between
    bounds and prevents latitude-dependent tooltip/overlay drift.
    """
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    # Build pixel grid with isotropic spacing in projected space.
    x_min = np.deg2rad(lon_min)
    x_max = np.deg2rad(lon_max)
    y_min = float(mercator_y(lat_min))
    y_max = float(mercator_y(lat_max))
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    nx = max(64, int(round((lon_max - lon_min) * px_per_deg_lon)))
    ny = max(64, int(round(nx * (y_span / x_span))))

    x_px = x_min + (np.arange(nx) + 0.5) / nx * x_span
    # Row 0 should be NORTH (max latitude / max mercator y)
    y_px = y_max - (np.arange(ny) + 0.5) / ny * y_span
    x_g, y_g = np.meshgrid(x_px, y_px)

    x_pts = np.deg2rad(lons)
    y_pts = mercator_y(lats)
    tree = cKDTree(np.column_stack([x_pts, y_pts]))
    _, idx = tree.query(np.column_stack([x_g.ravel(), y_g.ravel()]))
    grid = values[idx].reshape(ny, nx)

    rgba = PRECIP_CMAP(norm(grid))
    rgba[..., 3] *= alpha
    invisible = np.isnan(grid) | (grid < threshold_mm)
    rgba[invisible, 3] = 0
    rgba8 = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(rgba8)
    img.save(path, optimize=True)


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
    # Clamp precip vmax so a single thunderstorm cell doesn't wash out the rest
    # of the map. 5 mm/6h is already "heavy" — anything beyond is saturated dark.
    p_max = max(0.5, min(5.0, float(np.percentile(all_p, 99.5))))
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
        pr_grid = interpolate(lats, lons, pr_vals)

        save_overlay(t_grid, RDYLBU, norms["temp"],
                     OUTPUT_DIR / f"temp_{step}.png", alpha=0.75)
        save_overlay(ws_grid, WIND_CMAP, norms["wind"],
                     OUTPUT_DIR / f"wind_{step}.png", alpha=0.7)
        # Precip: pixel-exact Voronoi from the raw grid_points —
        # guarantees overlay colour == tooltip's nearest-grid value.
        save_precip_overlay_voronoi(lats, lons, p_vals, norms["precip"],
                                    OUTPUT_DIR / f"precip_{step}.png",
                                    alpha=0.85)
        save_overlay(pr_grid, PRESSURE_CMAP, norms["pressure"],
                     OUTPUT_DIR / f"pressure_{step}.png", alpha=0.7)

        # Spatially-binned wind arrows: 12 lon × 8 lat cells over the bbox,
        # pick the strongest arrow in each cell so the field is uniformly
        # covered instead of bunched along the multires-grid traversal path.
        N_LON, N_LAT = 12, 8
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_bin = np.clip(((lons - lon_min) / (lon_max - lon_min + 1e-9) * N_LON).astype(int), 0, N_LON - 1)
        lat_bin = np.clip(((lats - lat_min) / (lat_max - lat_min + 1e-9) * N_LAT).astype(int), 0, N_LAT - 1)
        best = {}
        for i in range(len(pts)):
            key = (lat_bin[i], lon_bin[i])
            cur = best.get(key)
            if cur is None or ws_vals[i] > cur[0]:
                best[key] = (ws_vals[i], i)
        arrows = []
        for _, (_, i) in best.items():
            arrows.append({
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "ws": float(ws_vals[i]),
                "wd": float(wd_vals[i]),
            })
        wind_data[str(step)] = arrows

    # Save bounds
    import datetime as _dt
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())],
                    [float(lats.max()), float(lons.max())]],
        "n_steps": n_steps,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="Path to forecast.json")
    ap.add_argument("--output-dir", type=Path, help="Output directory for overlays")
    ap.add_argument("--fine-res", type=int, default=None, help="Interpolation grid resolution")
    args = ap.parse_args()
    if args.input:
        FORECAST_JSON = args.input
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.fine_res:
        FINE_RES = args.fine_res
    main()
