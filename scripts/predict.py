# inference (memory-efficient for large grids)
# python scripts/predict.py experiments/demo --data-dir data/datasets/demo
# python scripts/predict.py experiments/wb2_512x256_19f_ar --max-samples 200
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
    lats = np.linspace(-90, 90, num_lat, endpoint=True)
    lons = np.linspace(0, 360, num_lon, endpoint=False)
    return lats, lons

def region_node_indices(lat_min, lat_max, lon_min, lon_max, lats, lons):
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

def read_coords(meta, data_dir: Path):
    npz = data_dir / "coords.npz"
    if npz.exists():
        z = np.load(npz)
        return z["latitude"].astype(np.float32), z["longitude"].astype(np.float32)
    return linspace_lats_lons(meta.num_latitudes, meta.num_longitudes)

# ====================== STREAMING METRICS ======================
class StreamingMetrics:
    """Накапливает MSE/MAE/ACC потоково — без хранения всех сэмплов в RAM."""

    def __init__(self, num_channels: int):
        self.C = num_channels
        self.n = 0
        self.total_elem = 0
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.sum_se_per_ch = np.zeros(num_channels, dtype=np.float64)
        self.elem_per_ch = np.zeros(num_channels, dtype=np.int64)
        self.sum_acc = np.zeros(num_channels, dtype=np.float64)
        self.acc_count = np.zeros(num_channels, dtype=np.int64)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """y_true, y_pred: [G, C*P] or [G, C]"""
        err = y_pred.float() - y_true.float()
        self.sum_se += err.pow(2).sum().item()
        self.sum_ae += err.abs().sum().item()
        self.total_elem += y_true.numel()

        # per-channel SE & spatial ACC
        CP = y_true.shape[1]
        eps = 1e-8
        for c in range(CP):
            yt = y_true[:, c].float()
            yp = y_pred[:, c].float()
            ch = c % self.C
            se_c = (yp - yt).pow(2).sum().item()
            self.sum_se_per_ch[ch] += se_c
            self.elem_per_ch[ch] += yt.numel()
            yt_a = yt - yt.mean()
            yp_a = yp - yp.mean()
            corr = (yt_a * yp_a).sum() / (yt_a.norm() * yp_a.norm() + eps)
            self.sum_acc[ch] += corr.item()
            self.acc_count[ch] += 1
        self.n += 1

    @property
    def mse(self):
        return self.sum_se / max(self.total_elem, 1)

    @property
    def rmse(self):
        return float(np.sqrt(self.mse))

    @property
    def mae(self):
        return self.sum_ae / max(self.total_elem, 1)

    @property
    def acc_per_channel(self):
        return self.sum_acc / np.maximum(self.acc_count, 1)

    @property
    def rmse_per_channel(self):
        """Normalized RMSE per channel."""
        mse_pc = self.sum_se_per_ch / np.maximum(self.elem_per_ch, 1)
        return np.sqrt(mse_pc)

    @property
    def acc(self):
        return float(self.acc_per_channel.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", help="напр. experiments/demo")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ckpt", default=None, help="путь к .pth")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="макс. тестовых сэмплов (default: 200 для больших сеток)")
    ap.add_argument("--region", nargs=4, type=float,
                    metavar=("LAT_MIN","LAT_MAX","LON_MIN","LON_MAX"))
    ap.add_argument("--prune-mesh", action="store_true",
                    help="Обрезать mesh до bounding-box данных + буфер. "
                         "Нужно для инференса на региональных датасетах (вместо полного глобуса).")
    ap.add_argument("--mesh-buffer", type=float, default=15.0,
                    help="Буфер (градусы) вокруг данных при prune-mesh (default: 15)")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--ar-steps", type=int, default=None,
                    help="Число AR-шагов для авторегрессионного инференса. "
                         "Если модель одношаговая (P=1), можно прогнать N шагов, "
                         "подавая выход обратно на вход. Напр. --ar-steps 4 для +24h.")

    # --- УСВОЕНИЕ ---
    ap.add_argument("--assim-method", default="none", choices=["none", "nudging", "oi"])
    ap.add_argument("--obs-path", type=str, default=None)
    ap.add_argument("--nudging-alpha", type=float, default=0.25)
    ap.add_argument("--nudging-mode", default="sequential", choices=["sequential", "offline"])
    ap.add_argument("--nudge-first-k", type=int, default=None)
    ap.add_argument("--oi-sigma-b", type=float, default=0.8)
    ap.add_argument("--oi-sigma-o", type=float, default=0.5)
    ap.add_argument("--oi-corr-len", type=float, default=10000.0)
    ap.add_argument("--boundary-blending", action="store_true")
    ap.add_argument("--background-path", type=str, default=None)
    ap.add_argument("--taper-width", type=int, default=5)

    args = ap.parse_args()

    exp_dir = args.experiment_dir
    cfg_path = os.path.join(exp_dir, FileNames.EXPERIMENT_CONFIG)
    ckpt_path = args.ckpt or os.path.join(exp_dir, FileNames.SAVED_MODEL)
    save_path = args.save or os.path.join(exp_dir, "predictions.pt")

    assert os.path.exists(cfg_path), f"нет конфига: {cfg_path}"
    assert os.path.exists(ckpt_path), f"нет чекпойнта: {ckpt_path}"

    exp_cfg = ExperimentConfig(**load_from_json_file(cfg_path))

    # --- data dir ---
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif getattr(exp_cfg, 'data_dir', None):
        # Multires и другие конфиги с явным data_dir
        data_dir = Path(exp_cfg.data_dir)
    else:
        ds_name = str(exp_cfg.data.dataset_name.value
                      if hasattr(exp_cfg.data.dataset_name, "value")
                      else exp_cfg.data.dataset_name)
        # v2 конфиги используют тот же физический датасет, что и v1
        physical_ds_name = ds_name.replace("_v2", "")
        # Пробуем относительный путь (как main.py), затем абсолютный от REPO_ROOT
        for candidate in [Path("data") / "datasets" / physical_ds_name,
                          REPO_ROOT / "data" / "datasets" / physical_ds_name,
                          Path("data") / "datasets" / ds_name,
                          REPO_ROOT / "data" / "datasets" / ds_name]:
            if candidate.exists():
                data_dir = candidate
                break
        else:
            data_dir = Path("data") / "datasets" / physical_ds_name  # fallback

    # --- load dataset ---
    is_chunked = (data_dir / "data.npy").exists() or (data_dir / "dataset_info.json").exists()

    # Определяем сколько целевых шагов нам нужно из датасета
    ds_pred_steps = args.ar_steps if args.ar_steps else exp_cfg.data.pred_window_used
    print(f"[predict] data_dir={data_dir} (chunked={is_chunked}, pred_steps={ds_pred_steps})")

    if is_chunked:
        train_ds, val_ds, test_ds, meta = load_chunked_datasets(
            data_path=str(data_dir),
            obs_window=exp_cfg.data.obs_window_used,
            pred_steps=ds_pred_steps,
            n_features=exp_cfg.data.num_features_used,
        )
    else:
        train_ds, val_ds, test_ds, meta = load_train_and_test_datasets(
            str(data_dir), exp_cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    # Определяем, используем ли реальные координаты из датасета (для регионов)
    real_coords = None
    region_bounds_for_mesh = None

    # Для flat grid всегда берём координаты из metadata (они уже загружены load_chunked_datasets)
    if getattr(meta, 'flat_grid', False) and hasattr(meta, 'cordinates'):
        real_coords = meta.cordinates

    coords_file = data_dir / "coords.npz"
    if args.prune_mesh and coords_file.exists():
        z = np.load(coords_file)
        real_lats = z["latitude"].astype(np.float32)
        real_lons = z["longitude"].astype(np.float32)
        real_coords = (real_lats, real_lons)
        # Bounding box данных → region_bounds для обрезки mesh
        region_bounds_for_mesh = (
            float(real_lats.min()), float(real_lats.max()),
            float(real_lons.min()), float(real_lons.max()),
        )
        print(f"[prune-mesh] Реальные координаты: "
              f"lat=[{real_lats.min():.2f},{real_lats.max():.2f}] "
              f"lon=[{real_lons.min():.2f},{real_lons.max():.2f}]")

    model = load_model_from_experiment_config(
        exp_cfg, device, meta,
        coordinates=real_coords,
        region_bounds=region_bounds_for_mesh,
        mesh_buffer=args.mesh_buffer,
        flat_grid=getattr(meta, 'flat_grid', False),
    )
    state = torch.load(ckpt_path, map_location=device)
    # При обрезке mesh размер _processing_edge_features меняется — убираем из state_dict,
    # т.к. буфер уже пересчитан при _init_ с правильными размерами.
    if region_bounds_for_mesh is not None:
        state = {k: v for k, v in state.items()
                 if not k.startswith("_processing_edge_features")}
    model.load_state_dict(state, strict=False)
    model = model.to(device)  # ensure ALL buffers (edge features etc.) are on device
    model.eval()

    if getattr(meta, 'flat_grid', False):
        G = meta.num_grid_nodes
    else:
        G = meta.num_longitudes * meta.num_latitudes
    C = exp_cfg.data.num_features_used
    P_model = exp_cfg.data.pred_window_used  # сколько шагов модель выдаёт за раз
    AR_STEPS = args.ar_steps if args.ar_steps else P_model  # сколько горизонтов хотим
    P = AR_STEPS  # общее число горизонтов для метрик
    OBS = exp_cfg.data.obs_window_used

    if AR_STEPS > P_model:
        print(f"[AR-rollout] Модель выдаёт {P_model} шаг(ов), но мы прогоним {AR_STEPS} шагов авторегрессионно.")

    # --- static channels: carry-forward вместо предсказания ---
    static_ch = getattr(exp_cfg, 'static_channels', [])
    if static_ch:
        print(f"[static] Каналы {static_ch} — статика, carry-forward при AR")

    # --- max samples (авто-лимит для больших сеток) ---
    if args.max_samples is not None:
        max_samples = args.max_samples
    elif is_chunked:
        max_samples = min(len(test_ds), 200)
    else:
        max_samples = len(test_ds)

    print(f"[predict] device={device} | grid={meta.num_longitudes}x{meta.num_latitudes} "
          f"| {max_samples}/{len(test_ds)} test samples")

    # --- DA setup ---
    observations = None
    if args.assim_method != "none":
        if args.obs_path:
            observations = torch.load(args.obs_path).cpu()
        elif not is_chunked:
            observations = test_ds.y.cpu()
            print("[DA] Используем y_test как наблюдения.")

    oi_solver = None
    if args.assim_method == "oi":
        lats, lons = read_coords(meta, data_dir)
        oi_solver = OptimalInterpolation(lats, lons, args.oi_sigma_b, args.oi_sigma_o, args.oi_corr_len, device)

    # --- boundary setup ---
    boundary_mask, background_data = None, None
    if args.boundary_blending:
        if args.background_path:
            background_data = torch.load(args.background_path).cpu()
            if background_data.dim() == 4:
                background_data = background_data.view(background_data.shape[0], -1, background_data.shape[-1])
        elif not is_chunked:
            background_data = test_ds.y.cpu()
        else:
            print("[Boundary] ОШИБКА: для chunked нужен --background-path"); sys.exit(1)
        mask_tensor = build_boundary_taper_mask(meta.num_latitudes, meta.num_longitudes, args.taper_width, args.taper_width)
        boundary_mask = mask_tensor.view(1, G, 1).float()

    # --- region ---
    region_idxs = None
    coords_npz = data_dir / "coords.npz"
    if args.region and coords_npz.exists():
        # --region задан явно → фильтруем по bbox из coords.npz (работает и для flat grid)
        z = np.load(coords_npz)
        c_lats = z["latitude"].astype(np.float32)
        c_lons = z["longitude"].astype(np.float32)
        lat_min_r, lat_max_r, lon_min_r, lon_max_r = args.region
        if c_lats.ndim == 1 and len(c_lats) != G:
            # Регулярная сетка (lats, lons — оси)
            region_idxs = region_node_indices(lat_min_r, lat_max_r, lon_min_r, lon_max_r, c_lats, c_lons)
        else:
            # Flat grid (multires): координаты уже (N,)
            mask = ((c_lats >= lat_min_r) & (c_lats <= lat_max_r) &
                    (c_lons >= lon_min_r) & (c_lons <= lon_max_r))
            region_idxs = np.where(mask)[0]
        print(f"[Region] {len(region_idxs)} nodes (--region [{lat_min_r},{lat_max_r}]N x [{lon_min_r},{lon_max_r}]E)")
    elif getattr(meta, 'is_regional', None) is not None:
        # Flat multires grid: region mask из metadata (весь ROI)
        region_idxs = np.where(meta.is_regional)[0]
        print(f"[Region] {len(region_idxs)} nodes (from is_regional mask)")
    elif args.region:
        lats, lons = read_coords(meta, data_dir)
        region_idxs = region_node_indices(*args.region, lats, lons)
        print(f"[Region] {len(region_idxs)} nodes")

    # --- streaming metrics (без хранения всех тензоров) ---
    sm_pred = StreamingMetrics(C)
    sm_base = StreamingMetrics(C)
    sm_pred_r = StreamingMetrics(C) if region_idxs is not None else None
    sm_base_r = StreamingMetrics(C) if region_idxs is not None else None

    sm_pred_rh, sm_base_rh = [], []  # per-horizon region metrics

    sm_pred_h, sm_base_h = [], []
    if AR_STEPS > 1:
        for _ in range(AR_STEPS):
            sm_pred_h.append(StreamingMetrics(C))
            sm_base_h.append(StreamingMetrics(C))
            if region_idxs is not None:
                sm_pred_rh.append(StreamingMetrics(C))
                sm_base_rh.append(StreamingMetrics(C))

    # --- accumulate predictions (if --save) ---
    save_preds_list = [] if (args.save and not args.no_save) else None
    save_gt_list = [] if (args.save and not args.no_save) else None
    save_sample_offsets = [] if (args.save and not args.no_save) else None

    # --- inference loop ---
    print(f"[Main] Инференс ({max_samples} samples, DA={args.assim_method})...")

    with torch.no_grad():
        for i, (X, y) in enumerate(torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)):
            if i >= max_samples:
                break

            y = y.squeeze(0)
            if len(y.shape) == 3:
                y = y.squeeze(-2)

            # Для AR-инференса: если y содержит меньше горизонтов, чем AR_STEPS,
            # пропускаем сэмпл (нет ground truth для всех шагов).
            y_total_steps = y.shape[-1] // C
            if AR_STEPS > y_total_steps:
                # Нет ground truth на столько горизонтов — берём сколько есть
                effective_P = y_total_steps
            else:
                effective_P = AR_STEPS

            X_last = X.squeeze(0)[:, -C:].repeat(1, effective_P)  # persistence baseline

            obs_i = y if observations is None else observations[i]

            X = X.to(device)

            # === PREDICTION ===
            if args.assim_method == "nudging" and args.nudging_mode == "sequential":
                x0 = X.view(1, X.shape[1], exp_cfg.data.obs_window_used, C)
                out = sequential_nudged_rollout(
                    model=model, x0=x0, y_obs=obs_i.unsqueeze(0), p=P,
                    alpha=args.nudging_alpha, k=args.nudge_first_k, device=device
                ).squeeze(0)

            elif args.assim_method == "oi":
                x0 = X.view(1, X.shape[1], exp_cfg.data.obs_window_used, C)
                curr_state = x0.to(device)
                y_obs_steps = obs_i.view(y.shape[0], P, C)
                test_out = model(curr_state.view(1, G, -1), attention_threshold=0.0).cpu()
                if test_out.shape[-1] == P*C:
                    out_steps = test_out.view(1, G, P, C)
                    for t in range(P):
                        out_steps[0,:,t,:] = oi_solver.apply(out_steps[0,:,t,:], y_obs_steps[:,t,:])
                    out = out_steps.view(1, G, P*C).squeeze(0)
                else:
                    batch_steps = []
                    for step in range(P):
                        out_step = model(curr_state.view(1, G, -1), attention_threshold=0.0).cpu()
                        if out_step.dim() == 2: out_step = out_step.unsqueeze(0)
                        out_step[0] = oi_solver.apply(out_step[0], y_obs_steps[:, step, :])
                        batch_steps.append(out_step)
                        curr_state = torch.cat([curr_state[:, :, 1:, :], out_step.to(device).unsqueeze(2)], dim=2)
                    out = torch.stack(batch_steps, dim=2).view(1, G, -1).squeeze(0)
            else:
                if AR_STEPS <= P_model:
                    # Модель сама выдаёт все горизонты
                    out = model(X, attention_threshold=0.0).cpu()
                else:
                    # AR-rollout: модель одношаговая, прогоняем несколько раз
                    # curr_state: [1, G, OBS, C]
                    curr_state = X.view(1, G, OBS, C)
                    ar_outs = []
                    for ar_step in range(AR_STEPS):
                        inp = curr_state.view(1, G, -1)
                        step_out = model(inp, attention_threshold=0.0)  # [G, C] or [1, G, C]
                        if step_out.dim() == 2:
                            step_out = step_out.unsqueeze(0)
                        ar_outs.append(step_out.cpu())
                        # Carry-forward: статические каналы подставляем из последнего входного шага
                        if static_ch:
                            static_vals = curr_state[:, :, -1, :]  # [1, G, C]
                            for ch in static_ch:
                                step_out[:, :, ch] = static_vals[:, :, ch]
                        # Сдвигаем окно: [obs0, obs1] → [obs1, pred]
                        curr_state = torch.cat(
                            [curr_state[:, :, 1:, :], step_out.unsqueeze(2)], dim=2
                        )
                    # Склеиваем все шаги: [1, G, AR_STEPS*C]
                    out = torch.cat(ar_outs, dim=-1).squeeze(0)  # [G, AR_STEPS*C]

                if args.assim_method == "nudging" and args.nudging_mode == "offline":
                    out = nudge_sequence_offline(out, obs_i, args.nudging_alpha)

            if boundary_mask is not None:
                bg = background_data[i].cpu()
                out = boundary_mask.squeeze(0).cpu() * out + (1.0 - boundary_mask.squeeze(0).cpu()) * bg

            # --- update streaming metrics ---
            out_cpu, y_cpu, bl_cpu = out.cpu(), y.cpu(), X_last.cpu()

            # Если out шире, чем y (нет ground truth на все шаги), обрезаем
            if out_cpu.shape[-1] > y_cpu.shape[-1]:
                out_cpu = out_cpu[:, :y_cpu.shape[-1]]
            if bl_cpu.shape[-1] > y_cpu.shape[-1]:
                bl_cpu = bl_cpu[:, :y_cpu.shape[-1]]

            sm_pred.update(y_cpu, out_cpu)
            sm_base.update(y_cpu, bl_cpu)

            if effective_P > 1:
                for p in range(effective_P):
                    sl = slice(p*C, (p+1)*C)
                    sm_pred_h[p].update(y_cpu[:, sl], out_cpu[:, sl])
                    sm_base_h[p].update(y_cpu[:, sl], bl_cpu[:, sl])

            if region_idxs is not None:
                sm_pred_r.update(y_cpu[region_idxs], out_cpu[region_idxs])
                sm_base_r.update(y_cpu[region_idxs], bl_cpu[region_idxs])
                if effective_P > 1:
                    for p in range(effective_P):
                        sl = slice(p*C, (p+1)*C)
                        sm_pred_rh[p].update(y_cpu[region_idxs][:, sl], out_cpu[region_idxs][:, sl])
                        sm_base_rh[p].update(y_cpu[region_idxs][:, sl], bl_cpu[region_idxs][:, sl])

            # --- save raw predictions ---
            if save_preds_list is not None:
                save_preds_list.append(out_cpu.clone())
                save_gt_list.append(y_cpu.clone())
                # Save temporal offset for temporal alignment with other datasets
                if hasattr(test_ds, '_sample_indices') and i < len(test_ds._sample_indices):
                    _, local_t = test_ds._sample_indices[i]
                    save_sample_offsets.append(local_t)
                else:
                    save_sample_offsets.append(i)

            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{max_samples}] RMSE={sm_pred.rmse:.6f} ACC={sm_pred.acc:.4f}")

    # --- persist predictions ---
    if save_preds_list is not None:
        preds_tensor = torch.stack(save_preds_list)  # (N, G, C*P)
        gt_tensor = torch.stack(save_gt_list)         # (N, G, C*P)
        save_dict = {
            "predictions": preds_tensor,
            "ground_truth": gt_tensor,
            "n_features": C,
            "ar_steps": AR_STEPS,
            "obs_window": OBS,
            "n_lon": meta.num_longitudes,
            "n_lat": meta.num_latitudes,
            "sample_offsets": save_sample_offsets,
            "data_dir": str(data_dir),
        }
        torch.save(save_dict, save_path)
        print(f"\n[Save] predictions → {save_path} (pred={preds_tensor.shape}, gt={gt_tensor.shape})")

    # === RESULTS ===
    N = sm_pred.n
    skill = 1.0 - (sm_pred.rmse / (sm_base.rmse + 1e-12))

    print()
    print("=" * 60)
    print(f"=== Inference summary ({N} samples) ===")
    print(f"Grid: {meta.num_longitudes}x{meta.num_latitudes} (G={G}) | C={C} | AR={AR_STEPS} horizons")
    print(f"Overall: MSE={sm_pred.mse:.6f} | RMSE={sm_pred.rmse:.6f} | MAE={sm_pred.mae:.6f}")
    print(f"Baseline: RMSE={sm_base.rmse:.6f} | MAE={sm_base.mae:.6f}")
    print(f"Skill: {skill*100:.2f}%")
    print(f"ACC: {sm_pred.acc:.4f} | base: {sm_base.acc:.4f} | Δ={sm_pred.acc - sm_base.acc:+.4f}")

    if AR_STEPS > 1 and sm_pred_h:
        print(f"\nPer-horizon:")
        for p in range(len(sm_pred_h)):
            sp, sb = sm_pred_h[p], sm_base_h[p]
            sk = 1.0 - (sp.rmse / (sb.rmse + 1e-12))
            print(f"  +{(p+1)*6:02d}h: RMSE={sp.rmse:.6f} | base={sb.rmse:.6f} | skill={sk*100:.2f}% | ACC={sp.acc:.4f} (base {sb.acc:.4f})")

    if args.per_channel:
        var_path = Path(data_dir) / "variables.json"
        var_order = json.loads(var_path.read_text()) if var_path.exists() else [f"ch{c}" for c in range(C)]

        # --- Загружаем scalers для денормализации ---
        scalers_path = Path(data_dir) / "scalers.npz"
        std = None
        if scalers_path.exists():
            scl = np.load(scalers_path)
            std = scl["std"].astype(np.float64)[:C]

        UNITS = {
            "t2m": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa",
            "tp": "m", "sp": "Pa", "tcwv": "kg/m²",
            "z_surf": "m²/s²", "lsm": "-",
            "t@850": "K", "u@850": "m/s", "v@850": "m/s",
            "z@850": "m²/s²", "q@850": "kg/kg",
            "t@500": "K", "u@500": "m/s", "v@500": "m/s",
            "z@500": "m²/s²", "q@500": "kg/kg",
        }

        # --- Per-horizon per-channel (ключевая таблица) ---
        if AR_STEPS > 1 and sm_pred_h and std is not None:
            # Ключевые переменные для компактной таблицы
            key_vars = ["t2m", "10u", "10v", "msl", "z@500", "t@850", "u@850", "v@850", "z@850"]
            key_idx = [i for i, v in enumerate(var_order[:C]) if v in key_vars]
            
            print(f"\nPer-horizon per-channel RMSE (physical units):")
            header = f"  {'var':>10s} {'unit':>6s}"
            for p in range(len(sm_pred_h)):
                header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
            print(header)
            
            for c in key_idx:
                name = var_order[c]
                unit = UNITS.get(name, "?")
                row = f"  {name:>10s} {unit:>6s}"
                for p in range(len(sm_pred_h)):
                    phys_rmse = sm_pred_h[p].rmse_per_channel[c] * std[c]
                    if "z@" in name or name == "z_surf":
                        row += f" {phys_rmse/9.81:8.1f}m"
                    elif name == "t2m" or name.startswith("t@"):
                        row += f" {phys_rmse:7.2f}°C"
                    else:
                        row += f" {phys_rmse:8.2f}"
                print(row)

        # --- Overall per-channel ---
        rmse_pc = sm_pred.rmse_per_channel
        acc_pc = sm_pred.acc_per_channel

        if std is not None:
            print(f"\nPer-channel metrics overall (physical units, avg over {AR_STEPS} horizons):")
            print(f"  {'#':>3s} {'var':>10s} {'ACC':>8s} {'RMSE_norm':>10s} {'RMSE_phys':>12s} {'unit':>8s}")
            for c, name in enumerate(var_order[:C]):
                unit = UNITS.get(name, "?")
                phys_rmse = rmse_pc[c] * std[c]
                extra = ""
                if "z@" in name or name == "z_surf":
                    extra = f"  (≈{phys_rmse/9.81:.1f} gpm)"
                elif name == "t2m" or name.startswith("t@"):
                    extra = f"  (≈{phys_rmse:.2f} °C)"
                print(f"  {c:3d} {name:>10s} {acc_pc[c]:8.4f} {rmse_pc[c]:10.4f} {phys_rmse:12.4f} {unit:>8s}{extra}")
        else:
            print(f"\nPer-channel ACC & RMSE (normalized):")
            for c, name in enumerate(var_order[:C]):
                print(f"  {c:2d}:{name:>10s}  ACC={acc_pc[c]:.4f}  RMSE={rmse_pc[c]:.4f}")

    if region_idxs is not None:
        sk_r = 1.0 - (sm_pred_r.rmse / (sm_base_r.rmse + 1e-12))
        if args.region:
            region_label = f"[{args.region[0]},{args.region[1]}]N x [{args.region[2]},{args.region[3]}]E"
        else:
            region_label = "is_regional mask"
        print(f"\n--- Region {region_label} ({len(region_idxs)} nodes) ---")
        print(f"RMSE={sm_pred_r.rmse:.6f} | base={sm_base_r.rmse:.6f} | skill={sk_r*100:.2f}%")
        print(f"ACC={sm_pred_r.acc:.4f} | base={sm_base_r.acc:.4f}")

        # Per-horizon region
        if AR_STEPS > 1 and sm_pred_rh:
            print(f"\n  Per-horizon (region):")
            for p in range(len(sm_pred_rh)):
                sp, sb = sm_pred_rh[p], sm_base_rh[p]
                sk = 1.0 - (sp.rmse / (sb.rmse + 1e-12))
                print(f"    +{(p+1)*6:02d}h: RMSE={sp.rmse:.6f} | base={sb.rmse:.6f} | skill={sk*100:.2f}% | ACC={sp.acc:.4f} (base {sb.acc:.4f})")

        # Per-channel region
        if args.per_channel:
            var_path = Path(data_dir) / "variables.json"
            var_order_r = json.loads(var_path.read_text()) if var_path.exists() else [f"ch{c}" for c in range(C)]
            scalers_path_r = Path(data_dir) / "scalers.npz"
            std_r = None
            if scalers_path_r.exists():
                scl_r = np.load(scalers_path_r)
                std_r = scl_r["std"].astype(np.float64)[:C]

            UNITS_R = {
                "t2m": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa",
                "tp": "m", "sp": "Pa", "tcwv": "kg/m²",
                "z_surf": "m²/s²", "lsm": "-",
                "t@850": "K", "u@850": "m/s", "v@850": "m/s",
                "z@850": "m²/s²", "q@850": "kg/kg",
                "t@500": "K", "u@500": "m/s", "v@500": "m/s",
                "z@500": "m²/s²", "q@500": "kg/kg",
            }

            # Per-horizon per-channel region table
            if AR_STEPS > 1 and sm_pred_rh and std_r is not None:
                key_vars_r = ["t2m", "10u", "10v", "msl", "t@850", "u@850", "v@850", "z@850", "z@500"]
                key_idx_r = [i for i, v in enumerate(var_order_r[:C]) if v in key_vars_r]

                print(f"\n  Per-horizon per-channel RMSE — REGION ({len(region_idxs)} nodes):")
                header = f"    {'var':>10s} {'unit':>6s}"
                for p in range(len(sm_pred_rh)):
                    header += f" {'+'+ str((p+1)*6) + 'h':>8s}"
                print(header)

                for c in key_idx_r:
                    name = var_order_r[c]
                    unit = UNITS_R.get(name, "?")
                    row = f"    {name:>10s} {unit:>6s}"
                    for p in range(len(sm_pred_rh)):
                        phys_rmse = sm_pred_rh[p].rmse_per_channel[c] * std_r[c]
                        if "z@" in name or name == "z_surf":
                            row += f" {phys_rmse/9.81:8.1f}m"
                        elif name == "t2m" or name.startswith("t@"):
                            row += f" {phys_rmse:7.2f}°C"
                        else:
                            row += f" {phys_rmse:8.2f}"
                    print(row)

            # Overall per-channel region
            rmse_pc_r = sm_pred_r.rmse_per_channel
            acc_pc_r = sm_pred_r.acc_per_channel
            if std_r is not None:
                print(f"\n  Per-channel region (physical units, avg over {AR_STEPS} horizons):")
                print(f"    {'#':>3s} {'var':>10s} {'ACC':>8s} {'RMSE_norm':>10s} {'RMSE_phys':>12s} {'unit':>8s}")
                for c, name in enumerate(var_order_r[:C]):
                    unit = UNITS_R.get(name, "?")
                    phys_rmse = rmse_pc_r[c] * std_r[c]
                    extra = ""
                    if "z@" in name or name == "z_surf":
                        extra = f"  (≈{phys_rmse/9.81:.1f} gpm)"
                    elif name == "t2m" or name.startswith("t@"):
                        extra = f"  (≈{phys_rmse:.2f} °C)"
                    print(f"    {c:3d} {name:>10s} {acc_pc_r[c]:8.4f} {rmse_pc_r[c]:10.4f} {phys_rmse:12.4f} {unit:>8s}{extra}")

    print("=" * 60)

if __name__ == "__main__":
    main()
