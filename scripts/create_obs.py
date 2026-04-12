import argparse
import torch
import numpy as np
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Создание файла разреженных наблюдений")
    parser.add_argument("--y-path", type=str, required=True, help="Путь к файлу y_test.pt")
    parser.add_argument("--vars-path", type=str, required=True, help="Путь к variables.json")
    parser.add_argument("--save-path", type=str, required=True, help="Куда сохранить результат (.pt)")
    
    parser.add_argument("--sparsity", type=float, default=0.1, help="Доля точек (0.1 = 10%)")
    parser.add_argument("--vars", type=str, default="all", help="Переменные через запятую (напр. 't2m,10u') или 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi", nargs=4, type=float, default=None,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Ограничить станции регионом (для multires). Нужен --coords-path.")
    parser.add_argument("--coords-path", type=str, default=None,
                        help="Путь к coords.npz (для --roi на flat grid)")
    
    args = parser.parse_args()
    
    # 1. Загружаем данные и конфиг переменных
    print(f"Загрузка данных из {args.y_path}...")
    y_full = torch.load(args.y_path)
    
    # === [FIX SPLIT] ЭМУЛЯЦИЯ ЛОГИКИ DATALOADER ===
    # dataloader.py делит y_test.pt пополам: первая половина Val, вторая Test.
    # predict.py работает со второй половиной. Мы тоже должны взять вторую.
    total_samples = y_full.shape[0]
    val_size = total_samples // 2
    print(f"[SPLIT] Всего сэмплов в файле: {total_samples}")
    print(f"[SPLIT] Отбрасываем первые {val_size} (Validation). Оставляем последние {total_samples - val_size} (Test).")
    
    y_full = y_full[val_size:]
    # ===============================================
    
    with open(args.vars_path, 'r') as f:
        var_names = json.load(f)
    
    # --- ЛОГИКА ОПРЕДЕЛЕНИЯ РАЗМЕРНОСТИ ---
    shape = y_full.shape
    print(f"Размерность данных для обработки: {shape}")
    
    C = len(var_names) 
    
    # Сценарий 1: Плоский [N, G, PC]
    if len(shape) == 3:
        N, G, PC = shape
        P = PC // C
        y_reshaped = y_full.view(N, G, P, C)
        
    # Сценарий 2: Пространственный [N, Lon, Lat, PC]
    elif len(shape) == 4:
        dim_last = shape[-1]
        if dim_last % C == 0:
            N, Lon, Lat, PC = shape
            G = Lon * Lat
            P = PC // C
            print(f"[INFO] Обнаружена пространственная сетка {Lon}x{Lat}. Сплющиваем в G={G}.")
            y_reshaped = y_full.view(N, G, PC).view(N, G, P, C)
        else:
            N, G, P, C_tensor = shape
            y_reshaped = y_full
            
    # Сценарий 3: Полный [N, Lon, Lat, P, C]
    elif len(shape) == 5:
        N, Lon, Lat, P, C_tensor = shape
        G = Lon * Lat
        y_reshaped = y_full.view(N, G, P, C_tensor)
        
    else:
        raise ValueError(f"Непонятная размерность данных: {shape}")

    print(f"Рабочие параметры: N={N}, G={G}, Steps={P}, Vars={C}")

    # 2. Создаем пустой тензор (все NaN)
    obs_tensor = torch.full_like(y_reshaped, float('nan'))
    
    # 3. Выбираем случайные точки (Sparsity)
    np.random.seed(args.seed)

    # Если задан ROI — станции только внутри региона
    candidate_indices = np.arange(G)
    if args.roi is not None:
        lat_min, lat_max, lon_min, lon_max = args.roi
        if args.coords_path is None:
            print("ОШИБКА: --roi требует --coords-path (coords.npz)")
            return
        coords = np.load(args.coords_path)
        c_lats = coords["latitude"].astype(np.float32)
        c_lons = coords["longitude"].astype(np.float32)
        if c_lats.ndim == 1 and len(c_lats) == G:
            # Flat grid (multires): координаты per-node
            roi_mask = ((c_lats >= lat_min) & (c_lats <= lat_max) &
                        (c_lons >= lon_min) & (c_lons <= lon_max))
        else:
            # Regular grid: meshgrid
            lon_grid, lat_grid = np.meshgrid(c_lons, c_lats)
            roi_mask = ((lat_grid.ravel() >= lat_min) & (lat_grid.ravel() <= lat_max) &
                        (lon_grid.ravel() >= lon_min) & (lon_grid.ravel() <= lon_max))
        candidate_indices = np.where(roi_mask)[0]
        print(f"[ROI] {len(candidate_indices)} узлов в регионе "
              f"[{lat_min:.0f}-{lat_max:.0f}°N, {lon_min:.0f}-{lon_max:.0f}°E]")

    num_stations = int(len(candidate_indices) * args.sparsity)
    if num_stations == 0: num_stations = 1 
    
    station_indices = np.random.choice(candidate_indices, num_stations, replace=False)
    print(f"Выбрано {num_stations} 'станций' из {len(candidate_indices)} узлов ({args.sparsity*100}%)")
    
    # 4. Определяем индексы переменных
    if args.vars == "all":
        var_indices = list(range(C))
        print("Усваиваем ВСЕ переменные.")
    else:
        requested_vars = [v.strip() for v in args.vars.split(",")]
        var_indices = []
        for v in requested_vars:
            if v in var_names:
                var_indices.append(var_names.index(v))
            else:
                found = False
                for i, name in enumerate(var_names):
                    if v == name: 
                        var_indices.append(i)
                        found = True
                        break
                if not found:
                    print(f"[WARN] Переменная '{v}' не найдена! Доступные: {var_names}")

        if not var_indices:
            print("ОШИБКА: Не выбрано ни одной переменной! Проверьте имена.")
            return
        print(f"Индексы усваиваемых переменных: {var_indices}")

    # 5. Заполняем данными
    for v_idx in var_indices:
        obs_tensor[:, station_indices, :, v_idx] = y_reshaped[:, station_indices, :, v_idx]

    # 6. Сохраняем
    obs_final = obs_tensor.reshape(N, G, P*C)
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(obs_final, args.save_path)
    print(f"Готово! Файл сохранен: {args.save_path}")

if __name__ == "__main__":
    main()

# Как использовать:
# python scripts/create_obs.py \
#   --y-path data/datasets/region_nsko_1440x721_15f_4obs_4pred_week/y_test.pt \
#   --vars-path data/datasets/wb2_64x32_zq_15f_4obs_4pred/variables.json \
#   --save-path data/datasets/region_nsko_1440x721_15f_4obs_4pred_week/obs_sparse_all.pt \
#   --sparsity 0.1 \
#   --vars all

# python scripts/create_obs.py \
#   --y-path data/datasets/region_nsko_1440x721_15f_4obs_4pred_week/y_test.pt \
#   --vars-path data/datasets/wb2_64x32_zq_15f_4obs_4pred/variables.json \
#   --save-path data/datasets/region_nsko_1440x721_15f_4obs_4pred_week/obs_sparse_dynamic.pt \
#   --sparsity 0.1 \
#   --vars "t2m,10u,10v,msl"
