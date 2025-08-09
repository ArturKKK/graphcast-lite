# inference
# python scripts/predict.py experiments/demo --data-dir data/datasets/demo
import argparse
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
from src.main import load_model_from_experiment_config

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="напр. experiments/demo")
    ap.add_argument("--data-dir", default=None, help="каталог с X_train.pt / y_train.pt / X_test.pt / y_test.pt (напр. data/datasets/demo)")
    ap.add_argument("--ckpt", default=None, help="путь к .pth; по умолчанию <exp>/best_model.pth")
    ap.add_argument("--save", default=None, help="куда сохранить predictions.pt (по умолчанию <exp>/predictions.pt)")
    ap.add_argument("--no-save", action="store_true", help="не сохранять predictions.pt")
    ap.add_argument("--region", nargs=4, type=float, metavar=("LAT_MIN","LAT_MAX","LON_MIN","LON_MAX"),
                    help="если указать, сохранит доп. файл с вырезкой региона <exp>/pred_region.pt и выведет метрики по региону")
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
    # !!! фикс: передаем data_dir, а не exp_dir
    train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(str(data_dir), exp_cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) собираем модель как в обучении
    model = load_model_from_experiment_config(exp_cfg, device, meta)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) прогоняем весь test + сразу считаем метрики
    preds, truths, baseline = [], [], []

    with torch.no_grad():
        for X, y in torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False):
            # Подготовим ground truth как в train/test
            y = y.squeeze(0)
            if len(y.shape) == 3:
                y = y.squeeze(-2)
            # Персистентный базлайн: «следующий шаг = последний наблюдённый»
            # Для flattened входа: X: [1, G, T*F] -> последний шаг это последние F каналов
            X_last = X.squeeze(0)[:, -exp_cfg.data.num_features_used:]

            X = X.to(device)
            out = model(X=X, attention_threshold=0.0)   # [G, pred_window*num_features_used]
            preds.append(out.cpu())
            truths.append(y.cpu())
            baseline.append(X_last.cpu())

    preds = torch.stack(preds, dim=0)          # [N, G, C]
    truths = torch.stack(truths, dim=0)        # [N, G, C]
    baseline = torch.stack(baseline, dim=0)    # [N, G, C]

    # Общие метрики по всему тесту
    mse, rmse, mae = _metrics(truths, preds)
    b_mse, b_rmse, b_mae = _metrics(truths, baseline)
    skill_rmse = 1.0 - (rmse / (b_rmse + 1e-12))

    N, G, C = preds.shape
    print()
    print("=== Inference summary ===")
    print(f"Dataset dir: {data_dir}")
    print(f"Grid: {meta.num_longitudes}x{meta.num_latitudes} (G={G}) | Obs_used={exp_cfg.data.obs_window_used} | Pred_used={exp_cfg.data.pred_window_used} | Features_used={exp_cfg.data.num_features_used}")
    print(f"Test samples: {N} | Channels (C): {C}")
    print(f"Overall: MSE={mse:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f}")
    print(f"Baseline(persistence): RMSE={b_rmse:.6f} | MAE={b_mae:.6f} | Skill(1-RMSE/RMSE_base)={skill_rmse*100:.2f}%")

    # Метрики по каждому каналу
    print("\nPer-channel metrics (channel index 0..C-1):")
    for c in range(C):
        m, r, a = _metrics(truths[..., c], preds[..., c])
        m_b, r_b, a_b = _metrics(truths[..., c], baseline[..., c])
        skill_c = 1.0 - (r / (r_b + 1e-12))
        print(f"  c={c}: MSE={m:.6f} RMSE={r:.6f} MAE={a:.6f} | base_RMSE={r_b:.6f} skill={skill_c*100:.2f}%")

    # 4) опционально — вырезка региона
    # --region 53 57 74 87 - Предсказания только для Новосибирской области
    if args.region:
        lat_min, lat_max, lon_min, lon_max = args.region
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

        # сохраняем регион как раньше
        region_path = os.path.join(exp_dir, "pred_region.pt")
        torch.save({"idxs": torch.tensor(idxs), "pred_region": region_pred}, region_path)
        print(f"Saved region slice: {region_path}  idxs={len(idxs)}")

    # 5) сохранить предсказания (можно отключить --no-save)
    if not args.no_save:
        torch.save(preds, save_path)
        print(f"\nSaved predictions: {save_path}  shape={tuple(preds.shape)}")

if __name__ == "__main__":
    main()


# Базлайн (baseline) — “наивный” прогноз, с которым мы сравниваем модель.
# Я взял персистентность: следующий шаг = последнее наблюдение. В погодных рядах это стандарт для короткого горизонта.
# RMSE = sqrt(mean((ŷ - y)^2)) — корень из среднеквадратичной ошибки.
# Меряет среднюю ошибку, сильнее штрафует выбросы, чем MAE.
# Skill (наша метрика улучшения) = 1 - RMSE(model) / RMSE(baseline).
# > 0 — модель лучше базлайна, < 0 — хуже, = 0 — равно.
