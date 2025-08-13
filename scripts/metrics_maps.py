#!/usr/bin/env python3
# Визуализация ошибок модели на карте (RMSE/MAE/BIAS/ACC) в физических единицах.
# Пример:
#   python scripts/metrics_maps.py experiments/wb2_64x32_15f \
#     --data-dir data/datasets/wb2_64x32_zq_15f_4obs_1pred \
#     --stat rmse --out-dir experiments/wb2_64x32_15f/maps

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.constants import FileNames
from src.data.dataloader import load_train_and_test_datasets
from src.main import load_model_from_experiment_config

def linspace_lats_lons(num_lat, num_lon):
    lats = np.linspace(-90, 90, num_lat, endpoint=True)
    lons = np.linspace(0, 360, num_lon, endpoint=False)
    return lats, lons

def load_scaler_stats(dataset_dir: Path):
    fn = dataset_dir / "scalers.npz"
    if not fn.exists():
        print(f"[WARN] no scalers.npz at {fn} — metrics will be in standardized units.")
        return None
    z = np.load(fn)
    return {
        "y_mean": z["y_mean"],   # shape [feat * pred_window]
        "y_scale": z["y_scale"], # shape [feat * pred_window]
    }

def inverse_standardize(Y, y_mean, y_scale):
    # Y: torch.Tensor [..., C]; y_mean/scale: np.ndarray [C]
    ym = torch.from_numpy(y_mean).to(Y.device, dtype=Y.dtype)
    ys = torch.from_numpy(y_scale).to(Y.device, dtype=Y.dtype)
    return Y * ys + ym

def unit_label(name):
    # человекочитаемые единицы + конверсия
    # возвращаем (label, factor, offset, special)
    # new_val = (val*factor + offset) if special is None
    # specials: 'z_to_m' (делим на g0), 'q_to_gkg' (умножаем на 1000)
    if name == "t2m" or name.startswith("t@"):
        return ("K", 1.0, 0.0, None)
    if name in ("10u","10v","u@850","u@500","v@850","v@500"):
        return ("m/s", 1.0, 0.0, None)
    if name == "msl":
        return ("hPa", 1/100.0, 0.0, None)   # Па -> гПа
    if name == "tp":
        return ("mm/6h", 1000.0, 0.0, None)  # м -> мм (за 6 часов)
    if name.startswith("z@"):
        return ("m", 1.0, 0.0, "z_to_m")     # геопотенциал -> метры
    if name.startswith("q@"):
        return ("g/kg", 1000.0, 0.0, "q_to_gkg")  # кг/кг -> г/кг
    return ("", 1.0, 0.0, None)

def apply_units(arr, var_name):
    # arr: torch.Tensor [...]; возвращаем torch.Tensor
    label, factor, offset, special = unit_label(var_name)
    out = arr
    if special == "z_to_m":
        g0 = 9.80665
        out = out / g0
    out = out * factor + offset
    return out, label

def compute_stat(pred, true, stat):
    # pred/true: [N, G] тензоры в физических единицах
    err = pred - true
    if stat == "rmse":
        return torch.sqrt(torch.mean(err**2, dim=0))
    if stat == "mae":
        return torch.mean(torch.abs(err), dim=0)
    if stat == "bias":
        return torch.mean(err, dim=0)
    if stat == "acc":
        # spatial anomaly correlation (по времени усредняем корр. по пространству)
        # нормируем каждую врем.выборку по полю
        p = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + 1e-8)
        t = (true - true.mean(dim=1, keepdim=True)) / (true.std(dim=1, keepdim=True) + 1e-8)
        return torch.mean(p * t, dim=0)  # коэффициент для каждого узла
    raise ValueError(stat)

def plot_field(field_1d, num_lon, num_lat, title, out_png):
    # field_1d: torch [G]
    arr = field_1d.detach().cpu().numpy().reshape(num_lon, num_lat).T  # [lat, lon]
    plt.figure(figsize=(10, 4.5))
    im = plt.imshow(arr, origin="lower", aspect="auto",
                    extent=[0,360,-90,90])
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="напр. experiments/wb2_64x32_15f")
    ap.add_argument("--data-dir", required=True, help="каталог с датасетом (содержит X_train.pt,y_train.pt,variables.json,scalers.npz)")
    ap.add_argument("--ckpt", default=None, help="путь к .pth; по умолчанию <exp>/best_model.pth")
    ap.add_argument("--stat", default="rmse", choices=["rmse","mae","bias","acc"])
    ap.add_argument("--out-dir", default=None, help="куда сохранить PNG-карты")
    ap.add_argument("--denorm", action="store_true", help="снять стандартизацию в физические единицы")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    cfg_path = exp_dir / FileNames.EXPERIMENT_CONFIG
    ckpt_path = Path(args.ckpt) if args.ckpt else exp_dir / FileNames.SAVED_MODEL
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or (exp_dir / f"maps_{args.stat}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))
    # загрузка датасета/мета (как в predict.py)
    train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(str(data_dir), exp_cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"[DEVICE] {device}")

    model = load_model_from_experiment_config(exp_cfg, device, meta)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # order of variables
    var_order = json.loads((data_dir / "variables.json").read_text())
    assert len(var_order) == exp_cfg.data.num_features_used, "features mismatch with variables.json"

    scaler = load_scaler_stats(data_dir) if args.denorm else None
    print(f"[INFO] denorm={'on' if scaler else 'off'} | stat={args.stat}")

    # Соберём предсказания/правду на всем тесте (батчами по 1)
    preds, truths = [], []
    with torch.no_grad():
        for X, y in torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False):
            y = y.squeeze(0)
            if len(y.shape) == 3:
                y = y.squeeze(-2)  # [G, C]
            X = X.to(device)
            out = model(X=X, attention_threshold=0.0).cpu()  # [G, C]
            preds.append(out)
            truths.append(y.cpu())
    P = torch.stack(preds, dim=0)   # [N, G, C]
    T = torch.stack(truths, dim=0)  # [N, G, C]

    # Денормализация (по каналам)
    if scaler:
        y_mean = scaler["y_mean"]; y_scale = scaler["y_scale"]
        P = inverse_standardize(P, y_mean, y_scale)
        T = inverse_standardize(T, y_mean, y_scale)

    # Метрики по каждому каналу в узлах (карты)
    C = P.shape[-1]
    num_lon, num_lat = meta.num_longitudes, meta.num_latitudes
    for c in range(C):
        name = var_order[c]
        P_c, T_c = P[..., c], T[..., c]  # [N, G]
        # конвертация единиц (tp -> мм, msl -> гПа, z -> м, q -> г/кг)
        P_c_u, unit = apply_units(P_c, name)
        T_c_u, _    = apply_units(T_c, name)

        stat_map = compute_stat(P_c_u, T_c_u, args.stat)  # [G]
        title = f"{args.stat.upper()} {name} [{unit}]"
        out_png = out_dir / f"{args.stat}_{name}.png"
        plot_field(stat_map, num_lon, num_lat, title, out_png)
        print(f"[SAVE] {out_png}")

    print(f"Done. Maps in: {out_dir}")

if __name__ == "__main__":
    main()
