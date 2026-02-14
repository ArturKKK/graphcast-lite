# inference
# python scripts/predict.py experiments/demo --data-dir data/datasets/demo
import argparse
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader import load_train_and_test_datasets
from src.data.dataloader_chunked import load_chunked_datasets
from src.main import load_model_from_experiment_config

# --- ИМПОРТЫ АЛГОРИТМОВ УСВОЕНИЯ ---
from src.assimilation.nudging import sequential_nudged_rollout, nudge_sequence_offline, build_boundary_taper_mask, NudgingAssimilator
from src.assimilation.optimal_interpolation import OptimalInterpolation
# -----------------------------------

def linspace_lats_lons(num_lat, num_lon):
    lats = np.linspace(-90, 90, num_lat, endpoint=True)        # как в src.main
    lons = np.linspace(0, 360, num_lon, endpoint=False)
    return lats, lons

def region_node_indices(lat_min, lat_max, lon_min, lon_max, lats, lons):
    # узлы нумеруются как: idx = lon_idx * num_lat + lat_idx
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    num_lat = len(lats)
    idxs = []
    for j in lon_idx:
        for i in lat_idx:
            idxs.append(j * num_lat + i)
    return np.array(idxs, dtype=np.int64)

def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_*: [N, G, C]
    err = (y_pred - y_true)
    mse = torch.mean(err**2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(err)).item()
    return mse, rmse, mae

def _spatial_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_*: [N, G, C]
    # Spatial ACC: по каждому сэмплу нормируем поле по узлам (аномации), считаем корреляцию, потом усредняем по времени.
    eps = 1e-8
    yt = y_true - y_true.mean(dim=1, keepdim=True)
    yp = y_pred - y_pred.mean(dim=1, keepdim=True)
    yt = yt / (y_true.std(dim=1, keepdim=True) + eps)
    yp = yp / (y_pred.std(dim=1, keepdim=True) + eps)
    # корреляция по полю → [N, C]
    corr_t = (yp * yt).mean(dim=1)
    # среднее по времени
    acc_per_c = corr_t.mean(dim=0)            # [C]
    acc_overall = acc_per_c.mean().item()     # float
    return acc_overall, acc_per_c

# === читаем РЕАЛЬНЫЕ оси координат из датасета, если есть ===
def read_coords(meta, data_dir: Path):
    """
    Пытаемся прочитать точные координаты из data_dir/coords.npz (создаётся build_region_wb2.py).
    В этом файле:
      - longitude: np.ndarray [num_lon] (например, 75.0, 75.25, ..., 90.0)
      - latitude:  np.ndarray [num_lat] (например, 50.0, 50.25, ..., 60.0)
    Если файла нет (глобальный WB2 64x32), используем равномерную глобальную сетку.
    """
    npz = data_dir / "coords.npz"
    if npz.exists():
        z = np.load(npz)
        lats = z["latitude"].astype(np.float32)
        lons = z["longitude"].astype(np.float32)
        return lats, lons
    # fallback: равномерная сетка во всю сферу
    return linspace_lats_lons(meta.num_latitudes, meta.num_longitudes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="напр. experiments/demo")
    ap.add_argument("--data-dir", default=None, help="каталог с X_train.pt / y_train.pt / X_test.pt / y_test.pt (напр. data/datasets/demo)")
    ap.add_argument("--ckpt", default=None, help="путь к .pth; по умолчанию <exp>/best_model.pth")
    ap.add_argument("--save", default=None, help="куда сохранить predictions.pt (по умолчанию <exp>/predictions.pt)")
    ap.add_argument("--no-save", action="store_true", help="не сохранять predictions.pt")
    ap.add_argument("--region", nargs=4, type=float, metavar=("LAT_MIN","LAT_MAX","LON_MIN","LON_MAX"),
                    help="если указать, сохранит доп. файл с вырезкой региона <exp>/pred_region.pt и выведет метрики по региону")
    ap.add_argument("--per-channel", action="store_true",
                    help="печать подробных метрик по каждому каналу и горизонту")
    
    # --- НОВЫЕ АРГУМЕНТЫ (УСВОЕНИЕ И ГРАНИЦЫ) ---
    ap.add_argument("--assim-method", default="none", choices=["none", "nudging", "oi"], help="Метод усвоения: nudging или oi")
    ap.add_argument("--obs-path", type=str, default=None, help="Путь к файлу с наблюдениями")
    
    # Nudging
    ap.add_argument("--nudging-alpha", type=float, default=0.25)
    ap.add_argument("--nudging-mode", default="sequential", choices=["sequential", "offline"])
    ap.add_argument("--nudge-first-k", type=int, default=None)

    # OI
    ap.add_argument("--oi-sigma-b", type=float, default=0.8)
    ap.add_argument("--oi-sigma-o", type=float, default=0.5)
    ap.add_argument("--oi-corr-len", type=float, default=10000.0) # 10000 пока лучшее
    
    # Границы
    ap.add_argument("--boundary-blending", action="store_true")
    ap.add_argument("--background-path", type=str, default=None)
    ap.add_argument("--taper-width", type=int, default=5)
    # ---------------------------------------

    args = ap.parse_args()

    exp_dir = args.experiment_dir
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = args.ckpt or os.path.join(exp_dir, FileNames.SAVED_MODEL)
    save_path = args.save or os.path.join(exp_dir, "predictions.pt")

    assert os.path.exists(cfg_path), f"нет конфига: {cfg_path}"
    assert os.path.exists(ckpt_path), f"нет чекпойнта: {ckpt_path}"

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    # Откуда брать тензоры:
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = REPO_ROOT / "data" / "datasets" / exp_cfg.data.dataset_name

    # 1) грузим конфиг и датасет
    # Автоопределение формата: если есть data.npy — chunked (memmap), иначе legacy (.pt)
    is_chunked = (data_dir / "data.npy").exists()

    if is_chunked:
        train_ds, val_ds, test_ds, meta = load_chunked_datasets(
            data_path=str(data_dir),
            obs_window=exp_cfg.data.obs_window_used,
            pred_steps=exp_cfg.data.pred_window_used,
            n_features=exp_cfg.data.num_features_used,
        )
        # У chunked-датасета нет .y — собираем через DataLoader
        print(f"[predict] Collecting test truths from chunked dataset "
              f"({len(test_ds)} samples) ...")
        _all_y = []
        for _, y_i in torch.utils.data.DataLoader(
                test_ds, batch_size=64, shuffle=False, num_workers=0):
            _all_y.append(y_i)
        truths = torch.cat(_all_y, dim=0).cpu()
        del _all_y
    else:
        # Legacy .pt format (64×32 и прочие)
        train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(
            str(data_dir), exp_cfg.data)
        truths = test_ds.y.cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) собираем модель как в обучении
    model = load_model_from_experiment_config(exp_cfg, device, meta)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    G = meta.num_longitudes * meta.num_latitudes
    C = exp_cfg.data.num_features_used
    P = exp_cfg.data.pred_window_used

    # --- ПОДГОТОВКА DA ---
    observations = truths
    if args.assim_method != "none":
        if args.obs_path:
            print(f"[DA] Загрузка наблюдений: {args.obs_path}")
            observations = torch.load(args.obs_path).cpu()
        else:
            print("[DA] ВНИМАНИЕ: Используем y_test (идеальные данные) как наблюдения.")

    oi_solver = None
    if args.assim_method == "oi":
        print(f"[DA] Инициализация OI (sigma_b={args.oi_sigma_b}, L={args.oi_corr_len})...")
        lats, lons = read_coords(meta, data_dir)
        oi_solver = OptimalInterpolation(lats, lons, args.oi_sigma_b, args.oi_sigma_o, args.oi_corr_len, device)

    # --- ПОДГОТОВКА ГРАНИЦ ---
    boundary_mask = None
    background_data = None
    if args.boundary_blending:
        print(f"[Boundary] Сшивание границ (width={args.taper_width})")
        if args.background_path:
            background_data = torch.load(args.background_path).cpu()
            # FIX: Распрямляем фон, если он 4D [N, Lon, Lat, C]
            if background_data.dim() == 4:
                 background_data = background_data.view(background_data.shape[0], -1, background_data.shape[-1])
        else:
            print("[Boundary] ВНИМАНИЕ: Фон не задан, используем y_test.")
            background_data = truths
            
        mask_tensor = build_boundary_taper_mask(meta.num_latitudes, meta.num_longitudes, args.taper_width, args.taper_width)
        boundary_mask = mask_tensor.view(1, G, 1).float()

    # 3) прогоняем весь test + сразу считаем метрики
    preds, baseline = [], []

    print(f"[Main] Запуск инференса (DA: {args.assim_method}, Bounds: {args.boundary_blending})...")

    with torch.no_grad():
        for i, (X, y) in enumerate(torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)):
            # Подготовим ground truth как в train/test
            y = y.squeeze(0)
            if len(y.shape) == 3:
                y = y.squeeze(-2)
            
            # Персистентный базлайн: «следующий шаг = последнее наблюдённое»
            # Для flattened входа: X: [1, G, T*F] -> последний шаг это последние F каналов
            # X_last = X.squeeze(0)[:, -exp_cfg.data.num_features_used:]
            X_last_1 = X.squeeze(0)[:, -C:]                     # [G, C]
            X_last = X_last_1.repeat(1, P)                       # [G, C*P]

            X = X.to(device)
            
            # === ЛОГИКА ПРОГНОЗА (ВСТАВКА) ===
            
            # 1. NUDGING (SEQUENTIAL)
            if args.assim_method == "nudging" and args.nudging_mode == "sequential":
                x0 = X.view(1, X.shape[1], exp_cfg.data.obs_window_used, C)
                y_obs = observations[i].unsqueeze(0)
                out = sequential_nudged_rollout(
                    model=model, x0=x0, y_obs=y_obs, p=P,
                    alpha=args.nudging_alpha, k=args.nudge_first_k, device=device
                )
                out = out.squeeze(0)

            # 2. OI (OPTIMAL INTERPOLATION)
            elif args.assim_method == "oi":
                x0 = X.view(1, X.shape[1], exp_cfg.data.obs_window_used, C)
                curr_state = x0.to(device)
                y_obs_full = observations[i]
                y_obs_steps = y_obs_full.view(y.shape[0], P, C)
                
                test_inp = curr_state.view(1, G, -1)
                test_out = model(test_inp, attention_threshold=0.0).cpu()
                
                if test_out.shape[-1] == P*C:
                    # 4-step: Offline OI
                    out_steps = test_out.view(1, G, P, C)
                    for t in range(P):
                        out_steps[0,:,t,:] = oi_solver.apply(out_steps[0,:,t,:], y_obs_steps[:,t,:])
                    out = out_steps.view(1, G, P*C).squeeze(0)
                else:
                    # 1-step: Rollout OI
                    batch_steps = []
                    for step in range(P):
                        inp = curr_state.view(1, G, -1)
                        out_step = model(inp, attention_threshold=0.0).cpu()
                        if out_step.dim() == 2: out_step = out_step.unsqueeze(0)
                        obs_step = y_obs_steps[:, step, :]
                        out_step[0] = oi_solver.apply(out_step[0], obs_step)
                        batch_steps.append(out_step)
                        out_dev = out_step.to(device)
                        curr_state = torch.cat([curr_state[:, :, 1:, :], out_dev.unsqueeze(2)], dim=2)
                    out = torch.stack(batch_steps, dim=2).view(1, G, -1).squeeze(0)

            # 3. STANDARD
            else:
                out = model(X, attention_threshold=0.0).cpu()   # [G, pred_window*num_features_used]
                if args.assim_method == "nudging" and args.nudging_mode == "offline":
                    out = nudge_sequence_offline(out, observations[i], args.nudging_alpha)

            # --- ГРАНИЦЫ (ИСПРАВЛЕННАЯ ФОРМУЛА) ---
            if boundary_mask is not None:
                bg = background_data[i].cpu()
                b_mask = boundary_mask.squeeze(0).cpu()
                # Центр (mask=1) -> out, Край (mask=0) -> bg
                out = b_mask * out + (1.0 - b_mask) * bg

            preds.append(out)
            baseline.append(X_last.cpu())

    preds = torch.stack(preds, dim=0)          # [N, G, C]
    # truths уже загружен в начале
    baseline = torch.stack(baseline, dim=0)    # [N, G, C]

    # Общие метрики по всему тесту
    mse, rmse, mae = _metrics(truths, preds)
    b_mse, b_rmse, b_mae = _metrics(truths, baseline)
    skill_rmse = 1.0 - (rmse / (b_rmse + 1e-12))

    # ACC (anomaly correlation): ближе к 1 — лучше; инвариантен к масштабу/смещению
    acc_overall, acc_pc = _spatial_acc(truths, preds)
    b_acc_overall, b_acc_pc = _spatial_acc(truths, baseline)

    N, G, CP = preds.shape
    C = exp_cfg.data.num_features_used
    P = exp_cfg.data.pred_window_used
    assert CP == C * P

    print()
    print("=== Inference summary ===")
    print(f"Dataset dir: {data_dir}")
    print(f"Grid: {meta.num_longitudes}x{meta.num_latitudes} (G={G}) | Obs_used={exp_cfg.data.obs_window_used} | Pred_used={exp_cfg.data.pred_window_used} | Features_used={exp_cfg.data.num_features_used}")
    print(f"Test samples: {N} | Features per step: {C} | Horizons: {P} (total targets dim={CP})")
    print(f"Overall: MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f}")
    print(f"Baseline(persistence): RMSE={b_rmse:.6f} | MAE={b_mae:.6f} | Skill(1-RMSE/RMSE_base)={skill_rmse*100:.2f}%")
    print(f"ACC (anomaly corr): overall={acc_overall:.3f} | baseline={b_acc_overall:.3f} | Δ={acc_overall - b_acc_overall:+.3f}")

    # Секции по горизонтам
    if P > 1:
        print("\nPer-horizon metrics (aggregated over channels):")
        for p in range(P):
            sl = slice(p*C, (p+1)*C)
            m, r, a = _metrics(truths[..., sl], preds[..., sl])
            m_b, r_b, a_b = _metrics(truths[..., sl], baseline[..., sl])
            skill_p = 1.0 - (r / (r_b + 1e-12))
            # ACC для горизонта:
            acc_p, _ = _spatial_acc(truths[..., sl], preds[..., sl])
            b_acc_p, _ = _spatial_acc(truths[..., sl], baseline[..., sl])
            hours = (p+1) * 6
            print(f"  +{hours:02d}h: RMSE={r:.6f} | MAE={a:.6f} | base_RMSE={r_b:.6f} | skill={skill_p*100:.2f}% | ACC={acc_p:.3f} (base {b_acc_p:.3f})")

    # 4) опционально — вырезка региона
    if args.region:
        lat_min, lat_max, lon_min, lon_max = args.region

        # NEW: читаем реальные координаты тайла, если это региональный датасет
        coords_path = Path(data_dir) / "coords.npz"
        if coords_path.exists():
            z = np.load(coords_path)
            lats = z["latitude"]    # [num_lat]
            lons = z["longitude"]   # [num_lon]
        else:
            lats, lons = linspace_lats_lons(meta.num_latitudes, meta.num_longitudes)

        idxs = region_node_indices(lat_min, lat_max, lon_min, lon_max, lats, lons)

        region_pred = preds[:, idxs, :]
        region_true = truths[:, idxs, :]
        region_base = baseline[:, idxs, :]

        rmse_r = _metrics(region_true, region_pred)[1]
        rmse_rb = _metrics(region_true, region_base)[1]
        skill_r = 1.0 - (rmse_r / (rmse_rb + 1e-12))

        print(f"\nRegion slice [{lat_min},{lat_max}]N x [{lon_min},{lon_max}]E | nodes={len(idxs)}")
        m_r, r_r, a_r = _metrics(region_true, region_pred)
        print(f"Region: MSE={m_r:.6f} RMSE={r_r:.6f} MAE={a_r:.6f} | base_RMSE={rmse_rb:.6f} skill={skill_r*100:.2f}%")

        # --- NEW: Per-horizon REGION metrics (aggregated over channels) ---
        if P > 1:
            print("\nPer-horizon REGION metrics (aggregated over channels):")
            for p in range(P):
                sl = slice(p*C, (p+1)*C)
                m, r, a = _metrics(region_true[..., sl], region_pred[..., sl])
                m_b, r_b, a_b = _metrics(region_true[..., sl], region_base[..., sl])
                skill_p = 1.0 - (r / (r_b + 1e-12))
                acc_p, _ = _spatial_acc(region_true[..., sl], region_pred[..., sl])
                b_acc_p, _ = _spatial_acc(region_true[..., sl], region_base[..., sl])
                hours = (p+1) * 6
                print(f"  +{hours:02d}h: RMSE={r:.6f} | MAE={a:.6f} | base_RMSE={r_b:.6f} | skill={skill_p*100:.2f}% | ACC={acc_p:.3f} (base {b_acc_p:.3f})")

        # --- NEW: Per-channel REGION metrics by horizon (only if --per-channel) ---
        if args.per_channel:
            # подстрахуемся: если var_order ещё не загружен выше
            try:
                var_order
            except NameError:
                var_path = Path(data_dir) / "variables.json"
                if var_path.exists():
                    var_order = json.loads(var_path.read_text())
                else:
                    var_order = [f"ch{c}" for c in range(C)]

            print("\nPer-channel REGION metrics by horizon:")
            for p in range(P):
                hours = (p+1) * 6
                print(f"\n  === +{hours:02d}h (region) ===")
                for c, name in enumerate(var_order):
                    idx = p*C + c  # канал в плоском CP
                    m, r, a = _metrics(region_true[..., idx:idx+1], region_pred[..., idx:idx+1])
                    m_b, r_b, a_b = _metrics(region_true[..., idx:idx+1], region_base[..., idx:idx+1])
                    skill = 1.0 - (r / (r_b + 1e-12))
                    acc, _ = _spatial_acc(region_true[..., idx:idx+1], region_pred[..., idx:idx+1])
                    b_acc, _ = _spatial_acc(region_true[..., idx:idx+1], region_base[..., idx:idx+1])
                    print(f"    {c:2d}:{name:>8s}  MSE={m:.6f} RMSE={r:.6f} MAE={a:.6f} | base_RMSE={r_b:.6f} skill={skill*100:.2f}% | ACC={acc:.3f} (base {b_acc:.3f})")

        region_path = os.path.join(exp_dir, "pred_region.pt")
        torch.save({"idxs": torch.tensor(idxs), "pred_region": region_pred}, region_path)
        print(f"Saved region slice: {region_path}  idxs={len(idxs)}")

    # 5) сохранить предсказания (можно отключить --no-save)
    if not args.no_save:
        torch.save(preds, save_path)
        print(f"\nSaved predictions: {save_path}  shape={tuple(preds.shape)}")

if __name__ == "__main__":
    main()
