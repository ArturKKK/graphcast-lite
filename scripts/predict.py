#!/usr/bin/env python3
#
# СКРИПТ ДЛЯ ИНФЕРЕНСА (ПРЕДСКАЗАНИЯ) МОДЕЛИ
# ==============================================================================
# Этот скрипт выполняет прогон обученной модели на тестовых данных.
#
# Поддерживает:
# 1. Standard Inference (Обычный прогноз)
# 2. Nudging (Усвоение методом релаксации)
# 3. OI (Оптимальная Интерполяция)
# 4. Boundary Blending (Сшивание границ)
# ==============================================================================

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Добавляем путь к корню проекта
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import FileNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from src.data.dataloader import load_train_and_test_datasets
from src.main import load_model_from_experiment_config

# --- ИМПОРТЫ МОДУЛЕЙ УСВОЕНИЯ ---
try:
    from src.assimilation.nudging import sequential_nudged_rollout, nudge_sequence_offline, build_boundary_taper_mask, NudgingAssimilator
    from src.assimilation.optimal_interpolation import OptimalInterpolation
except ImportError:
    print("[WARN] Модули усвоения не найдены! Функции --assim-method работать не будут.")
# --------------------------------

def linspace_lats_lons(num_lat, num_lon):
    """Генерация равномерной сетки."""
    lats = np.linspace(-90, 90, num_lat, endpoint=True)
    lons = np.linspace(0, 360, num_lon, endpoint=False)
    return lats, lons

def region_node_indices(lat_min, lat_max, lon_min, lon_max, lats, lons):
    """Индексы узлов для вырезания региона (метрики)."""
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
    """Базовые метрики (MSE, RMSE, MAE)."""
    err = (y_pred - y_true)
    mse = torch.mean(err**2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(err)).item()
    return mse, rmse, mae

def _spatial_acc(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Anomaly Correlation Coefficient (ACC)."""
    eps = 1e-8
    yt = y_true - y_true.mean(dim=1, keepdim=True)
    yp = y_pred - y_pred.mean(dim=1, keepdim=True)
    yt = yt / (y_true.std(dim=1, keepdim=True) + eps)
    yp = yp / (y_pred.std(dim=1, keepdim=True) + eps)
    corr_t = (yp * yt).mean(dim=1)
    acc_per_c = corr_t.mean(dim=0)
    acc_overall = acc_per_c.mean().item()
    return acc_overall, acc_per_c

def read_coords(meta, data_dir: Path):
    """Читает координаты из файла или генерирует дефолтные."""
    npz = data_dir / "coords.npz"
    if npz.exists():
        z = np.load(npz)
        lats = z["latitude"].astype(np.float32)
        lons = z["longitude"].astype(np.float32)
        return lats, lons
    return linspace_lats_lons(meta.num_latitudes, meta.num_longitudes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="Путь к эксперименту")
    ap.add_argument("--data-dir", default=None, help="Путь к данным")
    ap.add_argument("--ckpt", default=None, help="Путь к модели (.pth)")
    ap.add_argument("--save", default=None, help="Путь сохранения (.pt)")
    ap.add_argument("--no-save", action="store_true", help="Не сохранять прогноз")
    
    # Анализ
    ap.add_argument("--region", nargs=4, type=float, help="Регион: lat_min lat_max lon_min lon_max")
    ap.add_argument("--per-channel", action="store_true", help="Детальные метрики")
    
    # --- НАСТРОЙКИ УСВОЕНИЯ И ГРАНИЦ ---
    ap.add_argument("--assim-method", default="none", choices=["none", "nudging", "oi"], help="Метод: nudging или oi")
    ap.add_argument("--obs-path", type=str, default=None, help="Файл наблюдений")
    
    # Nudging
    ap.add_argument("--nudging-alpha", type=float, default=0.25)
    ap.add_argument("--nudging-mode", default="sequential", choices=["sequential", "offline"])
    ap.add_argument("--nudge-first-k", type=int, default=None)

    # OI
    ap.add_argument("--oi-sigma-b", type=float, default=0.8)
    ap.add_argument("--oi-sigma-o", type=float, default=0.5)
    ap.add_argument("--oi-corr-len", type=float, default=200000.0)

    # Границы
    ap.add_argument("--boundary-blending", action="store_true")
    ap.add_argument("--background-path", type=str, default=None)
    ap.add_argument("--taper-width", type=int, default=5)
    # -----------------------------------

    args = ap.parse_args()

    # 1. Инициализация
    exp_dir = Path(args.experiment_dir)
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = args.ckpt or os.path.join(exp_dir, FileNames.SAVED_MODEL)
    save_path = args.save or os.path.join(exp_dir, "predictions.pt")

    assert os.path.exists(cfg_path), f"Нет конфига: {cfg_path}"
    assert os.path.exists(ckpt_path), f"Нет чекпойнта: {ckpt_path}"

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))
    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data" / "datasets" / exp_cfg.data.dataset_name
    
    print(f"[Init] Данные: {data_dir}")
    train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(str(data_dir), exp_cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_experiment_config(exp_cfg, device, meta)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    truths = test_ds.y.cpu()
    G = test_ds.X.shape[1]
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

    # --- ПОДГОТОВКА ГРАНИЦ (ИСПРАВЛЕНО) ---
    boundary_mask = None
    background_data = None
    if args.boundary_blending:
        print(f"[Boundary] Сшивание границ (width={args.taper_width})")
        if args.background_path:
            background_data = torch.load(args.background_path).cpu()
        else:
            print("[Boundary] ВНИМАНИЕ: Фон не задан, используем y_test.")
            background_data = truths
        
        # Генерируем маску (это уже TENSOR, не numpy)
        # ВАЖНО: Используем G из truths, чтобы совпало с предсказаниями
        G_size = truths.shape[1]
        
        # build_boundary_taper_mask возвращает [G] Tensor
        mask_tensor = build_boundary_taper_mask(meta.num_latitudes, meta.num_longitudes, args.taper_width, args.taper_width)
        
        # Приводим к [1, G, 1] для корректного умножения на батч [N, G, C]
        # Здесь убираем torch.from_numpy, так как mask_tensor уже тензор
        boundary_mask = mask_tensor.view(1, G_size, 1).float()

    # ==================================================================
    # ОСНОВНОЙ ЦИКЛ (Рабочая логика без лишних функций)
    # ==================================================================
    print(f"[Main] Старт (DA: {args.assim_method}, Bounds: {args.boundary_blending})")
    
    preds = []
    baseline = []

    with torch.no_grad():
        # Проходим по тестовым примерам (батч=1)
        for i, (X, y) in enumerate(torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)):
            y = y.squeeze(0) if y.dim() == 3 else y
            
            # 1. Ветка NUDGING (Sequential)
            if args.assim_method == "nudging" and args.nudging_mode == "sequential":
                x0 = X.view(1, G, exp_cfg.data.obs_window_used, C)
                y_obs = observations[i].unsqueeze(0)
                
                # Вызываем функцию из nudging.py (она уже пропатчена под 4-step)
                out = sequential_nudged_rollout(
                    model=model, x0=x0, y_obs=y_obs, p=P,
                    alpha=args.nudging_alpha, k=args.nudge_first_k, device=device
                )
                out = out.squeeze(0)

            # 2. Ветка OI (Ручной цикл, как в рабочей версии)
            elif args.assim_method == "oi":
                x0 = X.view(1, G, exp_cfg.data.obs_window_used, C)
                curr_state = x0.to(device)
                
                # Наблюдения [G, P, C]
                y_obs_full = observations[i]
                y_obs_steps = y_obs_full.view(y.shape[0], P, C)
                
                # Проверка "умной" модели (4pred)
                test_inp = curr_state.view(1, G, -1)
                test_out = model(test_inp, attention_threshold=0.0).cpu()
                
                if test_out.shape[-1] == P*C:
                    # Модель выдала все шаги. Применяем OI как пост-обработку (Offline).
                    out_steps = test_out.view(1, G, P, C)
                    for t in range(P):
                        out_steps[0,:,t,:] = oi_solver.apply(out_steps[0,:,t,:], y_obs_steps[:,t,:])
                    out = out_steps.view(1, G, P*C).squeeze(0)
                else:
                    # Модель пошаговая. Крутим цикл.
                    batch_steps = []
                    for step in range(P):
                        inp = curr_state.view(1, G, -1)
                        out_step = model(inp, attention_threshold=0.0).cpu()
                        if out_step.dim() == 2: out_step = out_step.unsqueeze(0)
                        
                        # Apply OI
                        obs_step = y_obs_steps[:, step, :]
                        out_step[0] = oi_solver.apply(out_step[0], obs_step)
                        
                        batch_steps.append(out_step)
                        
                        # Авторегрессия
                        out_dev = out_step.to(device)
                        curr_state = torch.cat([curr_state[:, :, 1:, :], out_dev.unsqueeze(2)], dim=2)
                    
                    out = torch.stack(batch_steps, dim=2).view(1, G, -1).squeeze(0)

            # 3. Ветка STANDARD (Без усвоения)
            else:
                X_in = X.to(device)
                out = model(X_in, attention_threshold=0.0).cpu()
                
                # Offline Nudging (если выбран)
                if args.assim_method == "nudging" and args.nudging_mode == "offline":
                    out = nudge_sequence_offline(out, observations[i], args.nudging_alpha)

            # --- СШИВАНИЕ ГРАНИЦ ---
            if boundary_mask is not None:
                # out [G, P*C]
                bg = background_data[i].cpu()
                b_mask = boundary_mask.squeeze(0).cpu() # [G, 1]
                
                out = (1.0 - b_mask) * out + b_mask * bg

            preds.append(out)
            
            # Бейзлайн
            X_last = X.view(1, G, -1, C)[:, :, -1, :].repeat(1, 1, P).view(1, G, -1).squeeze(0)
            baseline.append(X_last)

    # Сборка результатов
    preds = torch.stack(preds, dim=0)
    baseline = torch.stack(baseline, dim=0)

    # Метрики
    mse, rmse, mae = _metrics(truths, preds)
    _, b_rmse, _ = _metrics(truths, baseline)
    skill = 1.0 - (rmse / (b_rmse + 1e-12))
    acc, _ = _spatial_acc(truths, preds)
    _, b_acc = _spatial_acc(truths, baseline)

    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТЫ")
    print(f"RMSE: {rmse:.6f}")
    print(f"Skill: {skill*100:.2f}%")
    print(f"ACC: {acc:.3f}")
    print("="*30 + "\n")

    # Детальные метрики (если надо)
    if P > 1:
        print("По шагам:")
        for p in range(P):
            sl = slice(p*C, (p+1)*C)
            _, r, _ = _metrics(truths[..., sl], preds[..., sl])
            _, br, _ = _metrics(truths[..., sl], baseline[..., sl])
            s = 1.0 - (r / (br + 1e-12))
            ac, _ = _spatial_acc(truths[..., sl], preds[..., sl])
            print(f"  T+{6*(p+1)}h: RMSE={r:.4f} | Skill={s*100:.1f}% | ACC={ac:.3f}")

    # Сохранение
    if not args.no_save:
        torch.save(preds, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()