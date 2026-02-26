import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pred-base", required=True)
    p.add_argument("--pred-exp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--var-idx", type=int, default=0) # 0 = t2m (обычно)
    p.add_argument("--step-idx", type=int, default=3) 
    p.add_argument("--title-suffix", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Размеры сетки
    Lon, Lat = 61, 41
    G = Lon * Lat 
    
    # 2. Грузим данные
    print("Loading data...")
    y_full = torch.load(os.path.join(args.data_dir, "y_test.pt"))
    val_size = y_full.shape[0] // 2
    y_test = y_full[val_size:] 
    
    pred_base = torch.load(args.pred_base)
    pred_exp = torch.load(args.pred_exp)

    # 3. Грузим Скейлеры (Scalers) для перевода в физику
    scaler_path = os.path.join(args.data_dir, "scalers.npz")
    mean = 0.0
    std = 1.0
    unit_name = "Norm Units"
    
    if os.path.exists(scaler_path):
        print("Found scalers.npz, denormalizing...")
        try:
            sc = np.load(scaler_path)
            # Обычно там y_mean / y_scale. Форма может быть разной, берем flatten
            # Предполагаем, что порядок переменных такой же, как в channels
            # Нам нужно достать mean/std конкретно для args.var_idx
            
            # y_scale обычно формы (channels,) или (1, 1, 1, channels)
            # В датасете 15 переменных.
            # Если var_idx = 0 (t2m), берем 0-й элемент.
            
            full_mean = sc["y_mean"].flatten()
            full_std = sc["y_scale"].flatten()
            
            # Учитываем, что в скейлере может быть 15 значений, а в тензоре P*C
            # Скейлеры обычно цикличны или одинаковы для всех шагов времени
            C = 15 # Стандарт
            
            # Индекс переменной внутри скейлера
            scaler_idx = args.var_idx % C
            
            mean = full_mean[scaler_idx]
            std = full_std[scaler_idx]
            
            # Если это температура (обычно индекс 0), переведем в Цельсии
            if args.var_idx == 0:
                unit_name = "°C"
                # В датасете Кельвины нормализованные.
                # Denorm: X * std + mean (получим Кельвины)
                # Kelvin -> Celsius: - 273.15
            elif args.var_idx in [1, 2]: # Ветер
                unit_name = "m/s"
            else:
                unit_name = "Phys Units"
                
        except Exception as e:
            print(f"Error loading scalers: {e}. Using raw values.")
    else:
        print("No scalers found. Plotting raw values.")

    # 4. Функция извлечения и денормализации
    def get_map(tensor, sample_idx, step, var):
        flat = tensor.reshape(tensor.shape[0], -1)
        try:
            C = 15
            P = flat.shape[1] // (G * C)
            view = flat.view(flat.shape[0], G, P, C)
            data_1d = view[sample_idx, :, step, var]
            
            # В numpy
            raw = data_1d.view(Lon, Lat).cpu().numpy()
            
            # Денормализация
            phys = raw * std + mean
            
            # Если температура (индекс 0), конвертируем K -> C
            if var == 0 and "°C" in unit_name:
                phys = phys - 273.15
                
            return phys
            
        except Exception as e:
            print(f"Error reshaping: {e}")
            return np.zeros((Lon, Lat))

    # Рисуем ПОСЛЕДНИЙ сэмпл
    idx = -1 
    
    truth = get_map(y_test, idx, args.step_idx, args.var_idx)
    base  = get_map(pred_base, idx, args.step_idx, args.var_idx)
    exp   = get_map(pred_exp, idx, args.step_idx, args.var_idx)

    # Разница (Bias): Прогноз - Факт
    diff_base = base - truth
    diff_exp = exp - truth

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Шкалы для Полей (чтобы сравнивать цвета)
    vmin_val = min(truth.min(), base.min(), exp.min())
    vmax_val = max(truth.max(), base.max(), exp.max())
    
    # Шкалы для Ошибок (симметричная, чтобы 0 был белым)
    # Находим максимальное отклонение по модулю
    limit = max(np.abs(diff_base).max(), np.abs(diff_exp).max())
    
    # 1. Truth
    im = axes[0,0].imshow(truth.T, origin='lower', cmap='jet', vmin=vmin_val, vmax=vmax_val)
    axes[0,0].set_title(f"Ground Truth (ERA5) [{unit_name}]")
    plt.colorbar(im, ax=axes[0,0])

    # 2. Prediction (Exp)
    im = axes[0,1].imshow(exp.T, origin='lower', cmap='jet', vmin=vmin_val, vmax=vmax_val)
    axes[0,1].set_title(f"Prediction: {args.title_suffix}")
    plt.colorbar(im, ax=axes[0,1])

    # 3. Error Baseline (Bias)
    # bwr = Blue-White-Red (Синий=Холоднее, Красный=Теплее, Белый=Точно)
    im = axes[1,0].imshow(diff_base.T, origin='lower', cmap='bwr', vmin=-limit, vmax=limit)
    axes[1,0].set_title("Error: Baseline (Control)")
    plt.colorbar(im, ax=axes[1,0])

    # 4. Error Exp (Bias)
    im = axes[1,1].imshow(diff_exp.T, origin='lower', cmap='bwr', vmin=-limit, vmax=limit)
    axes[1,1].set_title(f"Error: {args.title_suffix}")
    plt.colorbar(im, ax=axes[1,1])

    plt.suptitle(f"Forecast +{(args.step_idx+1)*6}h | {args.title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"phys_compare_{args.title_suffix}.png"))
    print("Done.")

if __name__ == "__main__":
    main()