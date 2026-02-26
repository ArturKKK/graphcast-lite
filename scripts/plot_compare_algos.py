import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pred-base", required=True) 
    p.add_argument("--pred-nudge", required=True)
    p.add_argument("--pred-oi", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group-name", required=True)
    
    # --- ЦИФРЫ ДЛЯ ЗАГОЛОВКОВ (Из таблицы) ---
    p.add_argument("--skill-nudge", type=float, default=0.0)
    p.add_argument("--rmse-nudge", type=float, default=0.0)
    
    p.add_argument("--skill-oi", type=float, default=0.0)
    p.add_argument("--rmse-oi", type=float, default=0.0)
    
    p.add_argument("--var-idx", type=int, default=0) # 0 = t2m
    p.add_argument("--step-idx", type=int, default=3) # +24h
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Lon, Lat = 61, 41
    G = Lon * Lat 
    
    # Грузим данные (только последний сэмпл)
    y_full = torch.load(os.path.join(args.data_dir, "y_test.pt"))
    val_size = y_full.shape[0] // 2
    y_test_all = y_full[val_size:]
    
    p_nudge_all = torch.load(args.pred_nudge)
    p_oi_all = torch.load(args.pred_oi)

    # Функция извлечения карты
    def get_map(tensor, sample_idx, step, var):
        flat = tensor.reshape(tensor.shape[0], -1)
        C = 15
        P = flat.shape[1] // (G * C)
        data = flat.view(flat.shape[0], G, P, C)[sample_idx, :, step, var]
        return data.view(Lon, Lat).cpu().numpy()

    idx = -1
    truth_raw = get_map(y_test_all, idx, args.step_idx, args.var_idx)
    nudge_raw = get_map(p_nudge_all, idx, args.step_idx, args.var_idx)
    oi_raw = get_map(p_oi_all, idx, args.step_idx, args.var_idx)

    # Денормализация (t2m)
    try:
        sc = np.load(os.path.join(args.data_dir, "scalers.npz"))
        mean = sc["y_mean"].flatten()[args.var_idx]
        std = sc["y_scale"].flatten()[args.var_idx]
    except:
        mean, std = 0, 1

    def to_phys(x):
        val = x * std + mean
        if args.var_idx == 0: val -= 273.15 # K -> C
        return val

    truth = to_phys(truth_raw)
    err_nudge = to_phys(nudge_raw) - truth
    err_oi = to_phys(oi_raw) - truth

    # Рисуем
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    limit = max(np.abs(err_nudge).max(), np.abs(err_oi).max())

    # Nudging
    im1 = axes[0].imshow(err_nudge.T, origin='lower', cmap='bwr', vmin=-limit, vmax=limit)
    axes[0].set_title(f"NUDGING\nRMSE: {args.rmse_nudge:.3f} | Skill: {args.skill_nudge:.1f}%")
    plt.colorbar(im1, ax=axes[0], label="Error Bias [°C]")

    # OI
    im2 = axes[1].imshow(err_oi.T, origin='lower', cmap='bwr', vmin=-limit, vmax=limit)
    axes[1].set_title(f"OPTIMAL INTERPOLATION (OI)\nRMSE: {args.rmse_oi:.3f} | Skill: {args.skill_oi:.1f}%")
    plt.colorbar(im2, ax=axes[1], label="Error Bias [°C]")

    plt.suptitle(f"Algorithm Comparison: {args.group_name}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(args.out_dir, f"compare_{args.group_name.replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()