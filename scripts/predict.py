# inference
# python scripts/predict.py experiments/demo --data-dir data/datasets/demo
import argparse
import json as _json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from src.assimilation import NudgingAssimilator, build_feature_mask, build_feature_mask_from_indices

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
    
    # --- Nudging / DA options ---
    ap.add_argument("--nudging", action="store_true", help="включить усвоение (nudging) на y_test (в режиме инференса)")
    ap.add_argument("--nudging-alpha", type=float, default=0.5, help="сила nudging [0..1], по умолчанию 0.5")
    ap.add_argument("--nudging-vars", type=str, default="", help="список переменных через запятую (имена) для усвоения, напр.: t2m,10u,10v")
    ap.add_argument("--nudging-idxs", type=str, default="", help="альтернатива: индексы каналов 0..C-1 для усвоения, через запятую")
    ap.add_argument("--feature-list", type=str, default="", help="путь к файлу со списком фич в порядке датасета (txt или json)")
    ap.add_argument("--blend-border", type=int, default=0, help="ширина тапера у границ для мягкого усвоения (0 = выкл.)")
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
            # Персистентный базлайн: «следующий шаг = последнее наблюдённое»
            # Для flattened входа: X: [1, G, T*F] -> последний шаг это последние F каналов
            # X_last = X.squeeze(0)[:, -exp_cfg.data.num_features_used:]
            C = exp_cfg.data.num_features_used
            P = exp_cfg.data.pred_window_used
            # --- init nudging (once) ---
            if "__nudging_initialized" not in locals():
                nudging_enabled = bool(args.nudging)
                if nudging_enabled:
                    # Build feature mask over [P*C]
                    C = exp_cfg.data.num_features_used
                    P = exp_cfg.data.pred_window_used
                    mask_flat = None
                    # try names
                    if args.nudging_vars and args.feature_list and os.path.exists(args.feature_list):
                        # load feature names
                        try:
                            import json
                            txt = open(args.feature_list, "r", encoding="utf-8").read()
                            if args.feature_list.endswith(".json"):
                                feature_list = json.loads(txt)
                            else:
                                if "\n" in txt and ("," not in txt):
                                    feature_list = [t.strip() for t in txt.splitlines() if t.strip()]
                                else:
                                    feature_list = [t.strip() for t in txt.split(",") if t.strip()]
                        except Exception:
                            feature_list = None
                        if feature_list:
                            sel = [s.strip() for s in args.nudging_vars.split(",") if s.strip()]
                            mask_flat = build_feature_mask(feature_list, sel, P, device)
                    # try indices
                    if (mask_flat is None) and args.nudging_idxs:
                        idxs = [int(s) for s in args.nudging_idxs.split(",") if s.strip().isdigit()]
                        mask_flat = build_feature_mask_from_indices(idxs, C, P, device)

                    nudger = NudgingAssimilator(
                        alpha=float(args.nudging_alpha),
                        feature_mask_flat=mask_flat,
                        grid_lon=None, grid_lat=None,
                        pred_window=P, num_features=C,
                        blend_border=int(args.blend_border),
                        device=device,
                    )
                nudging_flag = True

            X_last_1 = X.squeeze(0)[:, -C:]                     # [G, C]
            X_last = X_last_1.repeat(1, P)                       # [G, C*P]

            X = X.to(device)
            
            out = model(X=X, attention_threshold=0.0)   # [G, pred_window*num_features_used]
            if args.nudging:
                # use y as observations aligned with out; ensure y is flattened [G, P*C]
                y_obs = y
                if y_obs.dim() == 3 and y_obs.shape[-2] == 1:
                    y_obs = y_obs.squeeze(-2)
                out = nudger.apply(out, y_obs)

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
    C = exp_cfg.data.num_features_used
    P = exp_cfg.data.pred_window_used
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

    C = exp_cfg.data.num_features_used
    P = exp_cfg.data.pred_window_used
    # 4) опционально — вырезка региона
    # --region 53 57 74 87 - Предсказания только для Новосибирской области
    if args.region:
        lat_min, lat_max, lon_min, lon_max = args.region

        # NEW: читаем реальные координаты тайла, если это региональный датасет
        coords_path = Path(data_dir) / "coords.npz"
        if coords_path.exists():
            z = np.load(coords_path)
            lats = z["latitude"]    # [num_lat]
            lons = z["longitude"]   # [num_lon]
        else:
            # глобальный WB2 64x32 и пр. — используем линейку как в обучении
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
        C = exp_cfg.data.num_features_used
        P = exp_cfg.data.pred_window_used
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
