from pathlib import Path
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting import error_panel, field_panel, shared_limits, symmetric_limit

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pred-base", required=True, help="Контрольный прогноз (плохой)")
    p.add_argument("--pred-best", required=True, help="Лучший прогноз (хороший)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--step-idx", type=int, default=3) # +24h
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Lon, Lat = 61, 41
    G = Lon * Lat
    var_idx = 0 # t2m (Температура)

    print("Loading data for Final Shot...")
    y_full = torch.load(os.path.join(args.data_dir, "y_test.pt"))
    y_test = y_full[y_full.shape[0]//2:]
    
    p_base = torch.load(args.pred_base)
    p_best = torch.load(args.pred_best)

    # Скейлеры
    try:
        sc = np.load(os.path.join(args.data_dir, "scalers.npz"))
        mean = sc["y_mean"].flatten()[var_idx]
        std = sc["y_scale"].flatten()[var_idx]
    except:
        mean, std = 0, 1

    def get_phys_map(tensor, sample_idx):
        flat = tensor.reshape(tensor.shape[0], -1)
        C = 15
        P = flat.shape[1] // (G * C)
        data = flat.view(flat.shape[0], G, P, C)[sample_idx, :, args.step_idx, var_idx]
        raw = data.view(Lon, Lat).cpu().numpy()
        # Denorm K -> C
        return raw * std + mean - 273.15

    idx = -1
    truth = get_phys_map(y_test, idx)
    base = get_phys_map(p_base, idx)
    best = get_phys_map(p_best, idx)

    # Ошибки (Difference)
    err_base = base - truth
    err_best = best - truth

    # Метрики для заголовков (по этому кадру)
    rmse_base = np.sqrt(np.mean(err_base**2))
    rmse_best = np.sqrt(np.mean(err_best**2))
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Шкала ошибок (берем по максимуму Базлайна, чтобы Финал выглядел бледным)
    # Общий предел по обеим картам ошибки, чтобы «финал» и правда выглядел
    # бледнее контроля, а не просто был перекрашен в свою шкалу.
    limit = symmetric_limit(err_base, err_best)

    field_panel(axes[0], truth, title="Ground Truth (ERA5)\nTemperature [°C]")
    error_panel(axes[1], err_base, limit=limit,
                title=f"Error: Baseline (Control)\nRMSE: {rmse_base:.2f}°C")
    error_panel(axes[2], err_best, limit=limit,
                title=f"Error: Final Model (OI + Bounds)\nRMSE: {rmse_best:.2f}°C")

    plt.tight_layout()
    save_path = os.path.join(args.out_dir, "FINAL_RESULT.png")
    plt.savefig(save_path)
    print(f"Saved Final Shot: {save_path}")

if __name__ == "__main__":
    main()