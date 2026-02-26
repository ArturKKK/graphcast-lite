#!/usr/bin/env python3
"""
Интерполяция глобальных предсказаний (512×256, ~0.7°) на региональную ERA5
сетку (0.25°) и вычисление метрик.

Workflow:
  # 1) Глобальный инференс на кластере с сохранением:
  python scripts/predict.py experiments/wb2_512x256_19f_ar_v2 \
      --data-dir data/datasets/wb2_512x256_19f_jan2023 \
      --ckpt experiments/wb2_512x256_19f_ar_v2/best_model.pth \
      --ar-steps 4 --per-channel \
      --save predictions_global_jan2023.pt

  # 2) Интерполяция + метрики:
  python scripts/interpolate_to_region.py \
      --predictions predictions_global_jan2023.pt \
      --global-data data/datasets/wb2_512x256_19f_jan2023 \
      --region-data data/datasets/region_krsk_cds_19f \
      --per-channel

  # 3) Опционально — сохранить интерполированные предсказания для WRF:
  ... --out interpolated_krsk.pt
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]


# ─── helpers ──────────────────────────────────────────────────────────
def load_coords(data_dir: Path):
    z = np.load(data_dir / "coords.npz")
    return z["latitude"].astype(np.float64), z["longitude"].astype(np.float64)


def load_scalers(data_dir: Path):
    s = np.load(data_dir / "scalers.npz")
    return s["mean"].astype(np.float64), s["std"].astype(np.float64)


def load_variables(data_dir: Path):
    vf = data_dir / "variables.json"
    return json.loads(vf.read_text()) if vf.exists() else None


def load_dataset_info(data_dir: Path):
    return json.loads((data_dir / "dataset_info.json").read_text())


def flat_to_2d(flat, n_lon, n_lat):
    """
    Dataloader flattens as lon-major: node k  →  lon_idx = k // n_lat,
    lat_idx = k % n_lat.  Reshape (G,) → (n_lon, n_lat).
    """
    return flat.reshape(n_lon, n_lat)


def interpolate_field_2d(field_2d, src_lons, src_lats, dst_lons, dst_lats):
    """RegularGridInterpolator: field_2d[lon, lat] on (src_lons, src_lats)
    → (len(dst_lons), len(dst_lats))."""
    fn = RegularGridInterpolator(
        (src_lons, src_lats), field_2d,
        method="linear", bounds_error=False, fill_value=None,
    )
    lon_g, lat_g = np.meshgrid(dst_lons, dst_lats, indexing="ij")
    pts = np.stack([lon_g.ravel(), lat_g.ravel()], axis=-1)
    return fn(pts).reshape(len(dst_lons), len(dst_lats))


def compute_acc(y_true, y_pred):
    yt = y_true.ravel()
    yp = y_pred.ravel()
    yt_a = yt - yt.mean()
    yp_a = yp - yp.mean()
    return float((yt_a * yp_a).sum() / (np.linalg.norm(yt_a) * np.linalg.norm(yp_a) + 1e-8))


# ─── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Interpolate global predictions to regional grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--predictions", required=True,
                    help="predictions .pt saved by predict.py --save")
    ap.add_argument("--global-data", required=True,
                    help="Global dataset dir (wb2_512x256_19f_jan2023)")
    ap.add_argument("--region-data", required=True,
                    help="Regional dataset dir (region_krsk_cds_19f)")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Save interpolated predictions as .pt")
    args = ap.parse_args()

    global_dir = Path(args.global_data)
    region_dir = Path(args.region_data)

    # ── prediction file ──────────────────────────────────────────────
    sav = torch.load(args.predictions, map_location="cpu", weights_only=False)
    predictions = sav["predictions"].numpy().astype(np.float64)   # (N, G, C*P)
    gt_global   = sav["ground_truth"].numpy().astype(np.float64)  # (N, G, C*P)
    C  = int(sav["n_features"])
    P  = int(sav["ar_steps"])
    n_lon_g = int(sav["n_lon"])
    n_lat_g = int(sav["n_lat"])
    OBS = int(sav.get("obs_window", 2))
    sample_offsets = list(sav.get("sample_offsets", range(len(predictions))))
    N = predictions.shape[0]
    G_global = n_lon_g * n_lat_g

    print(f"Loaded {N} predictions: grid={n_lon_g}×{n_lat_g}, C={C}, P={P}")

    # ── global grid ──────────────────────────────────────────────────
    g_lats, g_lons = load_coords(global_dir)
    g_info = load_dataset_info(global_dir)
    g_start = datetime.strptime(g_info["time_start"], "%Y-%m-%d")
    print(f"Global: {g_info['time_start']}–{g_info['time_end']}  "
          f"lon=[{g_lons[0]:.2f}..{g_lons[-1]:.2f}]  "
          f"lat=[{g_lats[0]:.2f}..{g_lats[-1]:.2f}]")

    # ── regional grid ────────────────────────────────────────────────
    r_lats, r_lons = load_coords(region_dir)
    r_mean, r_std  = load_scalers(region_dir)
    r_info = load_dataset_info(region_dir)
    r_start = datetime.strptime(r_info["time_start"], "%Y-%m-%d")
    n_lon_r = int(r_info["n_lon"])
    n_lat_r = int(r_info["n_lat"])
    n_feat_r = int(r_info.get("n_features", r_info.get("n_feat", C)))
    G_region = n_lon_r * n_lat_r
    print(f"Region: {r_info['time_start']}–{r_info['time_end']}  "
          f"lon=[{r_lons[0]:.2f}..{r_lons[-1]:.2f}]  "
          f"lat=[{r_lats[0]:.2f}..{r_lats[-1]:.2f}]  "
          f"grid={n_lon_r}×{n_lat_r}={G_region}")

    # Temporal offset (6-h steps) between dataset starts
    dt = g_start - r_start
    time_offset = int(dt.total_seconds() / (6 * 3600))
    print(f"Temporal offset: global starts {time_offset} steps after regional")

    # ── load regional ground truth (raw → normalise) ─────────────────
    r_raw = np.memmap(str(region_dir / "data.npy"), dtype=np.float16, mode="r")
    n_time_r = int(r_info["n_time"])
    r_data = r_raw.reshape(n_time_r, n_lon_r, n_lat_r, n_feat_r).copy().astype(np.float64)
    r_norm = (r_data - r_mean) / r_std
    print(f"Regional data loaded: {r_data.shape}")

    # ── variable names ───────────────────────────────────────────────
    vars_ = load_variables(global_dir) or [f"ch{i}" for i in range(C)]

    # ── interpolation loop ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Interpolating {N} global samples → {n_lon_r}×{n_lat_r} regional grid")
    print(f"{'='*70}")

    horizon_preds = [[] for _ in range(P)]
    horizon_gts   = [[] for _ in range(P)]
    horizon_bases = [[] for _ in range(P)]
    matched = 0

    interp_out_all = []

    for s in range(N):
        local_t = sample_offsets[s]
        # prediction covers global timesteps [local_t+OBS .. local_t+OBS+P-1]
        pred_start_r = local_t + OBS + time_offset
        base_t_r     = local_t + OBS - 1 + time_offset

        # bounds check for regional data
        if pred_start_r < 0 or pred_start_r + P > n_time_r or base_t_r < 0:
            continue

        pred_flat = predictions[s]  # (G_global, C*P)
        interp_steps = np.zeros((P, n_lon_r, n_lat_r, C), dtype=np.float64)

        for p in range(P):
            for c in range(C):
                col = p * C + c
                field_2d = flat_to_2d(pred_flat[:, col], n_lon_g, n_lat_g)
                interp_steps[p, :, :, c] = interpolate_field_2d(
                    field_2d, g_lons, g_lats, r_lons, r_lats)

        gt_region   = r_norm[pred_start_r: pred_start_r + P, :, :, :C]
        base_region = r_norm[base_t_r, :, :, :C]

        for p in range(P):
            horizon_preds[p].append(interp_steps[p])
            horizon_gts[p].append(gt_region[p])
            horizon_bases[p].append(base_region)

        interp_out_all.append(interp_steps)
        matched += 1

    print(f"\nMatched samples: {matched}/{N}")
    if matched == 0:
        print("ERROR: no temporal overlap. Check dataset dates.")
        sys.exit(1)

    # ── metrics ──────────────────────────────────────────────────────
    all_pred, all_gt, all_base = [], [], []

    for p in range(P):
        hp = np.array(horizon_preds[p])
        hg = np.array(horizon_gts[p])
        hb = np.array(horizon_bases[p])

        err  = hp - hg
        berr = hb - hg
        rmse   = np.sqrt((err ** 2).mean())
        r_base = np.sqrt((berr ** 2).mean())
        mae    = np.abs(err).mean()
        skill  = 1.0 - rmse / (r_base + 1e-12)
        acc    = compute_acc(hg, hp)

        print(f"\n  +{(p+1)*6:02d}h: RMSE={rmse:.6f} | base={r_base:.6f} | "
              f"skill={skill*100:.2f}% | MAE={mae:.6f} | ACC={acc:.4f}")

        if args.per_channel:
            for c in range(C):
                ce  = hp[:, :, :, c] - hg[:, :, :, c]
                cbe = hb[:, :, :, c] - hg[:, :, :, c]
                cr = np.sqrt((ce ** 2).mean())
                cb = np.sqrt((cbe ** 2).mean())
                cs = 1.0 - cr / (cb + 1e-12)
                ca = compute_acc(hg[:, :, :, c], hp[:, :, :, c])
                print(f"    {c:2d}: {vars_[c]:>8s}  RMSE={cr:.4f} base={cb:.4f} "
                      f"skill={cs*100:+.1f}% ACC={ca:.4f}")

        all_pred.append(hp);  all_gt.append(hg);  all_base.append(hb)

    # overall
    ap_ = np.concatenate(all_pred)
    ag  = np.concatenate(all_gt)
    ab  = np.concatenate(all_base)
    o_rmse  = np.sqrt(((ap_ - ag) ** 2).mean())
    o_base  = np.sqrt(((ab - ag) ** 2).mean())
    o_skill = 1.0 - o_rmse / (o_base + 1e-12)
    o_acc   = compute_acc(ag, ap_)

    print(f"\n{'='*70}")
    print(f"Overall ({matched}×{P} horizons × {G_region} nodes × {C} ch):")
    print(f"  RMSE={o_rmse:.6f} | base={o_base:.6f} | "
          f"skill={o_skill*100:.2f}% | ACC={o_acc:.4f}")
    print(f"{'='*70}")

    # ── save ─────────────────────────────────────────────────────────
    if args.out:
        out = np.array(interp_out_all, dtype=np.float32)
        gts = np.stack([np.array(horizon_gts[p], dtype=np.float32) for p in range(P)], axis=1)
        torch.save({
            "interpolated_predictions": torch.from_numpy(out),
            "regional_ground_truth": torch.from_numpy(gts),
            "variables": vars_[:C],
            "region_lats": r_lats,
            "region_lons": r_lons,
            "n_matched": matched,
            "scalers_mean": r_mean[:C],
            "scalers_std":  r_std[:C],
        }, args.out)
        print(f"Saved interpolated: {args.out}")


if __name__ == "__main__":
    main()
