import argparse, torch, os, numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=3) # 3 = +24h
    args = p.parse_args()

    # Жестко для НСК (как договаривались)
    Lon, Lat = 61, 41
    G = Lon * Lat
    var_idx = 0 # t2m

    print("Loading...")
    y_full = torch.load(os.path.join(args.data_dir, "y_test.pt"))
    y_test = y_full[y_full.shape[0]//2:] 
    pred_tensor = torch.load(args.pred)

    # Грузим скейлеры для t2m
    sc = np.load(os.path.join(args.data_dir, "scalers.npz"))
    # Берем mean/std для 0-го канала (t2m)
    # Скейлеры обычно плоские или (1,1,1,C), берем flatten
    mean = sc["y_mean"].flatten()[var_idx]
    std = sc["y_scale"].flatten()[var_idx]

    def get_map(tensor):
        flat = tensor.reshape(tensor.shape[0], -1)
        C = 15
        P = flat.shape[1] // (G * C)
        # Берем последний сэмпл (-1)
        data = flat.view(flat.shape[0], G, P, C)[-1, :, args.step, var_idx]
        val = data.view(Lon, Lat).cpu().numpy()
        # Денормализация (K -> C)
        return (val * std + mean) - 273.15

    truth = get_map(y_test)
    pred = get_map(pred_tensor)
    diff = pred - truth

    # Рисуем
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Общая шкала температур
    vmin, vmax = min(truth.min(), pred.min()), max(truth.max(), pred.max())
    # Шкала ошибки (симметричная)
    lim = max(abs(diff.min()), abs(diff.max()))

    im0 = axs[0].imshow(truth.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth (ERA5) [°C]"); plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[1].set_title("Prediction (Baseline)"); plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(diff.T, origin='lower', cmap='bwr', vmin=-lim, vmax=lim)
    axs[2].set_title("Error Map (Pred - Truth)"); plt.colorbar(im2, ax=axs[2])

    os.makedirs(args.out, exist_ok=True)
    save_path = os.path.join(args.out, "slide5_baseline.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()