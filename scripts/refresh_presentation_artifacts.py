#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

CITY_BBOX = (55.5, 56.5, 92.0, 94.0)
ROI_BBOX = (50.0, 60.0, 83.0, 98.0)

DISPLAY = {
    "t2m": {"label": "Temperature 2m", "unit": "C", "to_phys": lambda x: x - 273.15, "cmap": "RdYlBu_r"},
    "10u": {"label": "U wind 10m", "unit": "m/s", "to_phys": lambda x: x, "cmap": "coolwarm"},
    "10v": {"label": "V wind 10m", "unit": "m/s", "to_phys": lambda x: x, "cmap": "coolwarm"},
    "sp": {"label": "Surface pressure", "unit": "hPa", "to_phys": lambda x: x / 100.0, "cmap": "viridis"},
    "msl": {"label": "MSL pressure", "unit": "hPa", "to_phys": lambda x: x / 100.0, "cmap": "viridis"},
    "t@850": {"label": "Temperature 850 hPa", "unit": "C", "to_phys": lambda x: x - 273.15, "cmap": "RdYlBu_r"},
    "t@500": {"label": "Temperature 500 hPa", "unit": "C", "to_phys": lambda x: x - 273.15, "cmap": "RdYlBu_r"},
}
DEFAULT_DISPLAY = {"label": "", "unit": "", "to_phys": lambda x: x, "cmap": "viridis"}

OBS_GROUPS = {
    "all": None,
    "surf": ["t2m", "10u", "10v", "sp", "msl"],
    "temp": ["t2m", "t@850", "t@500"],
    "dyn": ["10u", "10v", "u@850", "v@850", "u@500", "v@500", "z@850", "z@500", "sp", "msl"],
}


@dataclass
class BundleData:
    predictions: np.ndarray
    truth: np.ndarray
    var_names: list[str]
    latitudes: np.ndarray
    longitudes: np.ndarray
    city_mask: np.ndarray
    roi_mask: np.ndarray


@dataclass
class PreflightReport:
    failures: list[str]
    warnings: list[str]
    checks: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def run_logged(name: str, cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print("$", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.write(proc.stdout)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {name}. See {log_path}")


def run_logged_if_needed(
    name: str,
    cmd: list[str],
    log_path: Path,
    outputs: list[Path] | None = None,
    cwd: Path | None = None,
    reuse_existing: bool = False,
) -> None:
    expected_outputs = outputs or [log_path]
    if reuse_existing and log_path.exists() and all(path.exists() for path in expected_outputs):
        print(f"\n=== {name} ===")
        print(f"[skip] Reusing existing outputs: {', '.join(str(path) for path in expected_outputs)}")
        return
    run_logged(name=name, cmd=cmd, log_path=log_path, cwd=cwd)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def format_path(path: Path | None) -> str:
    if path is None:
        return "skipped"
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def default_artifact_store(out_dir: Path) -> Path:
    data_root = Path("/data")
    if data_root.exists():
        return data_root / "graphcast-lite" / "presentation_cache" / out_dir.name / "artifacts"
    return out_dir / "artifacts"


def ensure_file_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target_path)


def materialize_artifact_path(link_dir: Path, store_dir: Path, file_name: str) -> Path:
    link_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / file_name
    target_path = store_dir / file_name
    if link_dir.resolve() != store_dir.resolve():
        ensure_file_symlink(link_path, target_path)
    return target_path


def check_python_modules(module_names: list[str], failures: list[str], checks: list[str]) -> None:
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            failures.append(f"Missing Python module: {module_name}")
        else:
            checks.append(f"OK module: {module_name}")


def check_required_path(path: Path, label: str, failures: list[str], checks: list[str]) -> None:
    if path.exists():
        checks.append(f"OK {label}: {path}")
    else:
        failures.append(f"Missing {label}: {path}")


def check_dataset_dir(path: Path, label: str, failures: list[str], checks: list[str]) -> None:
    if not path.exists():
        failures.append(f"Missing {label}: {path}")
        return
    if not path.is_dir():
        failures.append(f"{label} is not a directory: {path}")
        return

    checks.append(f"OK {label}: {path}")
    required_files = ["coords.npz", "scalers.npz", "variables.json"]
    for file_name in required_files:
        check_required_path(path / file_name, f"{label}/{file_name}", failures, checks)

    has_data_source = (path / "data.npy").exists() or (path / "dataset_info.json").exists()
    if has_data_source:
        checks.append(f"OK {label}: found data.npy or dataset_info.json")
    else:
        failures.append(f"Missing {label} payload: expected data.npy or dataset_info.json in {path}")


def check_experiment_dir(path: Path, label: str, failures: list[str], checks: list[str]) -> None:
    if not path.exists():
        failures.append(f"Missing {label}: {path}")
        return
    if not path.is_dir():
        failures.append(f"{label} is not a directory: {path}")
        return

    checks.append(f"OK {label}: {path}")
    check_required_path(path / "config.json", f"{label}/config.json", failures, checks)
    check_required_path(path / "best_model.pth", f"{label}/best_model.pth", failures, checks)


def run_preflight(
    freeze_exp: Path,
    nofreeze_exp: Path,
    main_data: Path,
    jan_data: Path,
    wrf_json: Path,
) -> PreflightReport:
    failures: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    check_python_modules(["numpy", "torch", "matplotlib", "scipy", "torch_geometric"], failures, checks)
    check_required_path(REPO_ROOT / "scripts/predict.py", "predict script", failures, checks)
    check_required_path(REPO_ROOT / "scripts/plot_region_multires.py", "plot script", failures, checks)
    check_required_path(REPO_ROOT / "scripts/compare_wrf.py", "WRF compare script", failures, checks)
    check_experiment_dir(freeze_exp, "freeze experiment", failures, checks)
    check_experiment_dir(nofreeze_exp, "nofreeze experiment", failures, checks)
    check_dataset_dir(main_data, "main dataset", failures, checks)

    if jan_data.exists():
        check_dataset_dir(jan_data, "Jan2023 dataset", failures, checks)
    else:
        warnings.append(f"Jan2023 dataset missing, WRF step will be skipped: {jan_data}")

    if wrf_json.exists():
        checks.append(f"OK WRF JSON: {wrf_json}")
    else:
        warnings.append(f"WRF JSON missing, WRF step will be skipped: {wrf_json}")

    return PreflightReport(failures=failures, warnings=warnings, checks=checks)


def write_preflight_report(path: Path, report: PreflightReport) -> None:
    lines = ["# Preflight Report", ""]
    lines.append("Status: OK" if report.ok else "Status: FAILED")
    lines.append("")
    lines.append("Checks:")
    lines.extend(f"- {line}" for line in report.checks)
    lines.append("")
    lines.append("Warnings:")
    if report.warnings:
        lines.extend(f"- {line}" for line in report.warnings)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Failures:")
    if report.failures:
        lines.extend(f"- {line}" for line in report.failures)
    else:
        lines.append("- none")
    write_summary(path, lines)


def print_preflight_report(report: PreflightReport) -> None:
    print("\n=== preflight ===")
    for line in report.checks:
        print(f"[OK] {line}")
    for line in report.warnings:
        print(f"[WARN] {line}")
    for line in report.failures:
        print(f"[FAIL] {line}")
    if report.ok:
        print("Preflight passed.")
    else:
        print("Preflight failed.")


def bbox_mask(latitudes: np.ndarray, longitudes: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = bbox
    return (
        (latitudes >= lat_min)
        & (latitudes <= lat_max)
        & (longitudes >= lon_min)
        & (longitudes <= lon_max)
    )


def load_bundle(bundle_path: Path, data_dir: Path) -> BundleData:
    payload = torch.load(bundle_path, map_location="cpu")
    predictions = payload["predictions"].numpy()
    truth = payload["ground_truth"].numpy()
    n_features = int(payload["n_features"])
    ar_steps = int(payload["ar_steps"])

    with (data_dir / "variables.json").open("r", encoding="utf-8") as handle:
        var_names = json.load(handle)
    var_names = var_names[:n_features]

    scalers = np.load(data_dir / "scalers.npz")
    if "mean" in scalers:
        mean = scalers["mean"].astype(np.float32)[:n_features]
        std = scalers["std"].astype(np.float32)[:n_features]
    else:
        mean = scalers["y_mean"].astype(np.float32)[:n_features]
        std = scalers["y_scale"].astype(np.float32)[:n_features]

    coords = np.load(data_dir / "coords.npz")
    latitudes = coords["latitude"].astype(np.float32)
    longitudes = coords["longitude"].astype(np.float32)

    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], ar_steps, n_features)
    truth = truth.reshape(truth.shape[0], truth.shape[1], ar_steps, n_features)
    predictions = predictions * std[None, None, None, :] + mean[None, None, None, :]
    truth = truth * std[None, None, None, :] + mean[None, None, None, :]

    city_mask_full = bbox_mask(latitudes, longitudes, CITY_BBOX)
    roi_mask_full = bbox_mask(latitudes, longitudes, ROI_BBOX)
    keep_mask = city_mask_full | roi_mask_full
    if keep_mask.any():
        predictions = predictions[:, keep_mask]
        truth = truth[:, keep_mask]
        latitudes = latitudes[keep_mask]
        longitudes = longitudes[keep_mask]

    return BundleData(
        predictions=predictions.astype(np.float32),
        truth=truth.astype(np.float32),
        var_names=var_names,
        latitudes=latitudes,
        longitudes=longitudes,
        city_mask=bbox_mask(latitudes, longitudes, CITY_BBOX),
        roi_mask=bbox_mask(latitudes, longitudes, ROI_BBOX),
    )


def build_observation_channels(var_names: list[str], group_name: str) -> np.ndarray:
    wanted = OBS_GROUPS[group_name]
    if wanted is None:
        return np.arange(len(var_names), dtype=np.int64)
    idxs = [idx for idx, name in enumerate(var_names) if name in wanted]
    return np.array(idxs, dtype=np.int64)


def build_sparse_observations(bundle: BundleData, station_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    roi_indices = np.where(bundle.roi_mask)[0]
    if len(roi_indices) == 0:
        raise ValueError("ROI mask is empty")
    station_count = max(1, int(round(len(roi_indices) * station_fraction)))
    rng = np.random.default_rng(seed)
    station_positions = np.sort(rng.choice(len(roi_indices), size=station_count, replace=False))
    station_indices = roi_indices[station_positions]
    return roi_indices, station_indices


def apply_offline_nudging(bundle: BundleData, station_indices: np.ndarray, channel_indices: np.ndarray, alpha: float) -> np.ndarray:
    updated = bundle.predictions.copy()
    if len(channel_indices) == 0:
        return updated
    for channel_idx in channel_indices:
        updated[:, station_indices, :, channel_idx] = (
            (1.0 - alpha) * updated[:, station_indices, :, channel_idx]
            + alpha * bundle.truth[:, station_indices, :, channel_idx]
        )
    return updated


def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6_371_000.0
    lat1_r = np.deg2rad(lat1)[:, None]
    lon1_r = np.deg2rad(lon1)[:, None]
    lat2_r = np.deg2rad(lat2)[None, :]
    lon2_r = np.deg2rad(lon2)[None, :]
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return (2.0 * radius * np.arcsin(np.sqrt(a))).astype(np.float32)


def build_oi_gain(bundle: BundleData, roi_indices: np.ndarray, station_indices: np.ndarray, sigma_b: float, sigma_o: float, corr_len_m: float) -> np.ndarray:
    roi_lats = bundle.latitudes[roi_indices]
    roi_lons = bundle.longitudes[roi_indices]
    station_lats = bundle.latitudes[station_indices]
    station_lons = bundle.longitudes[station_indices]

    full_to_station = haversine_distance_m(roi_lats, roi_lons, station_lats, station_lons)
    station_to_station = haversine_distance_m(station_lats, station_lons, station_lats, station_lons)

    cov_xo = (sigma_b ** 2) * np.exp(-((full_to_station / corr_len_m) ** 2))
    cov_oo = (sigma_b ** 2) * np.exp(-((station_to_station / corr_len_m) ** 2))
    cov_oo += (sigma_o ** 2) * np.eye(cov_oo.shape[0], dtype=np.float32)
    return cov_xo @ np.linalg.inv(cov_oo)


def apply_roi_oi(
    bundle: BundleData,
    roi_indices: np.ndarray,
    station_indices: np.ndarray,
    channel_indices: np.ndarray,
    kalman_gain: np.ndarray,
) -> np.ndarray:
    updated = bundle.predictions.copy()
    if len(channel_indices) == 0:
        return updated

    roi_positions = np.arange(len(roi_indices))
    station_positions = np.searchsorted(roi_indices, station_indices)

    for sample_idx in range(updated.shape[0]):
        for step_idx in range(updated.shape[2]):
            for channel_idx in channel_indices:
                background_roi = updated[sample_idx, roi_indices, step_idx, channel_idx]
                innovation = bundle.truth[sample_idx, station_indices, step_idx, channel_idx] - background_roi[station_positions]
                analysis_roi = background_roi + kalman_gain @ innovation
                updated[sample_idx, roi_indices, step_idx, channel_idx] = analysis_roi.astype(np.float32)
    return updated


def save_region_npz(path: Path, bundle: BundleData, predictions: np.ndarray, mask: np.ndarray, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tag=np.array(tag),
        predictions=predictions[:, mask],
        truth=bundle.truth[:, mask],
        latitudes=bundle.latitudes[mask],
        longitudes=bundle.longitudes[mask],
        variables=np.array(bundle.var_names),
    )


def display_values(values: np.ndarray, var_name: str) -> np.ndarray:
    display = DISPLAY.get(var_name, DEFAULT_DISPLAY)
    return display["to_phys"](values)


def scatter_map(ax, lons: np.ndarray, lats: np.ndarray, values: np.ndarray, title: str, cmap: str, vmin=None, vmax=None, center_zero=False) -> None:
    if center_zero:
        lim = max(abs(float(values.min())), abs(float(values.max())), 1e-6)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        scatter = ax.scatter(lons, lats, c=values, s=18, cmap=cmap, edgecolors="none", norm=norm)
    else:
        scatter = ax.scatter(lons, lats, c=values, s=18, cmap=cmap, edgecolors="none", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def plot_truth_and_method_errors(
    out_path: Path,
    bundle: BundleData,
    method_predictions: dict[str, np.ndarray],
    var_name: str,
    sample_idx: int,
    step_idx: int,
) -> None:
    display = DISPLAY.get(var_name, DEFAULT_DISPLAY)
    city_idx = np.where(bundle.city_mask)[0]
    channel_idx = bundle.var_names.index(var_name)

    lats = bundle.latitudes[city_idx]
    lons = bundle.longitudes[city_idx]
    truth = display_values(bundle.truth[sample_idx, city_idx, step_idx, channel_idx], var_name)

    fig, axes = plt.subplots(1, len(method_predictions) + 1, figsize=(4.6 * (len(method_predictions) + 1), 4.8))
    pred_examples = [display_values(pred[sample_idx, city_idx, step_idx, channel_idx], var_name) for pred in method_predictions.values()]
    vmin = min(float(truth.min()), *(float(x.min()) for x in pred_examples))
    vmax = max(float(truth.max()), *(float(x.max()) for x in pred_examples))

    scatter_map(axes[0], lons, lats, truth, f"Truth\n{display['label']} [{display['unit']}]", display["cmap"], vmin=vmin, vmax=vmax)
    for ax, (name, pred) in zip(axes[1:], method_predictions.items()):
        pred_disp = display_values(pred[sample_idx, city_idx, step_idx, channel_idx], var_name)
        error = pred_disp - truth
        rmse = float(np.sqrt(np.mean(error ** 2)))
        scatter_map(ax, lons, lats, error, f"{name}\nError RMSE={rmse:.2f}", "bwr", center_zero=True)

    fig.suptitle(f"{var_name} +{(step_idx + 1) * 6:02d}h, sample {sample_idx}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_region_rmse_summary(out_path: Path, bundle: BundleData, method_predictions: dict[str, np.ndarray], var_name: str) -> None:
    city_idx = np.where(bundle.city_mask)[0]
    channel_idx = bundle.var_names.index(var_name)
    steps = bundle.truth.shape[2]
    labels = [f"+{(step + 1) * 6}h" for step in range(steps)]
    x = np.arange(steps)
    width = 0.8 / max(1, len(method_predictions))

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset_idx, (name, pred) in enumerate(method_predictions.items()):
        errs = []
        for step_idx in range(steps):
            truth = display_values(bundle.truth[:, city_idx, step_idx, channel_idx], var_name)
            pred_vals = display_values(pred[:, city_idx, step_idx, channel_idx], var_name)
            errs.append(float(np.sqrt(np.mean((pred_vals - truth) ** 2))))
        offset = (offset_idx - len(method_predictions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, errs, width=width, label=name)
        for bar, err in zip(bars, errs):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{err:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"RMSE [{DISPLAY.get(var_name, DEFAULT_DISPLAY)['unit']}]")
    ax.set_title(f"City-region RMSE by horizon: {var_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_rmse_curves(out_path: Path, bundle: BundleData, method_predictions: dict[str, np.ndarray], var_name: str, title: str) -> None:
    city_idx = np.where(bundle.city_mask)[0]
    channel_idx = bundle.var_names.index(var_name)
    steps = bundle.truth.shape[2]
    hours = np.array([(step + 1) * 6 for step in range(steps)], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, pred in method_predictions.items():
        errs = []
        for step_idx in range(steps):
            truth = display_values(bundle.truth[:, city_idx, step_idx, channel_idx], var_name)
            pred_vals = display_values(pred[:, city_idx, step_idx, channel_idx], var_name)
            errs.append(float(np.sqrt(np.mean((pred_vals - truth) ** 2))))
        ax.plot(hours, errs, marker="o", linewidth=2.0, label=name)

    ax.set_xlabel("Forecast horizon [hours]")
    ax.set_ylabel(f"RMSE [{DISPLAY.get(var_name, DEFAULT_DISPLAY)['unit']}]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_live_outputs(live_dir: Path, out_dir: Path, slides_dir: Path) -> Path | None:
    if not live_dir.exists():
        return None

    live_stage_dir = out_dir / "live"
    live_stage_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for file_name in ["summary.txt", "openmeteo_comparison.md", "openmeteo_comparison.csv"]:
        src = live_dir / file_name
        if src.exists():
            shutil.copy2(src, live_stage_dir / file_name)
            copied_files.append(file_name)

    live_plot = live_dir / "t2m_city_forecast.png"
    if live_plot.exists():
        shutil.copy2(live_plot, slides_dir / "live_gdas_t2m_3day.png")
        copied_files.append("slides/live_gdas_t2m_3day.png")

    write_summary(
        live_stage_dir / "README.md",
        [
            "# Live GDAS 3-Day Outputs",
            "",
            f"Source directory: {live_dir}",
            "Staged files:",
            *[f"- {name}" for name in copied_files],
            "",
            "Notes:",
            "- Input analyses are GDAS fields ingested into the trained GraphCast-lite multires model.",
            "- Open-Meteo files are kept only as a reference forecast comparison for the same valid times.",
        ],
    )
    return live_stage_dir


def build_predict_cmd(
    experiment_dir: Path,
    data_dir: Path,
    save_path: Path | None,
    region_bbox: tuple[float, float, float, float],
    ar_steps: int,
    max_samples: int,
    split: str = "test_only",
    no_save: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/predict.py",
        str(experiment_dir),
        "--data-dir",
        str(data_dir),
        "--region",
        str(region_bbox[0]),
        str(region_bbox[1]),
        str(region_bbox[2]),
        str(region_bbox[3]),
        "--ar-steps",
        str(ar_steps),
        "--per-channel",
        "--no-residual",
        "--max-samples",
        str(max_samples),
        "--split",
        split,
    ]
    if no_save:
        cmd.append("--no-save")
    else:
        cmd.extend(["--save", str(save_path)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh evaluation artifacts and slide assets for GraphCast-lite")
    parser.add_argument("--freeze-exp", default="experiments/multires_nores_freeze6")
    parser.add_argument("--nofreeze-exp", default="experiments/multires_nores_nofreeze")
    parser.add_argument("--main-data", default="data/datasets/multires_krsk_19f")
    parser.add_argument("--global-exp", default="experiments/wb2_64x32_ar_15f_4obs_4pred")
    parser.add_argument("--global-data", default="data/datasets/wb2_64x32_zq_15f_4obs_4pred")
    parser.add_argument("--global-region", nargs=4, type=float, default=CITY_BBOX)
    parser.add_argument("--global-ar-steps", type=int, default=12)
    parser.add_argument("--global-max-samples", type=int, default=200)
    parser.add_argument("--live-dir", default="results/live_gdas_run_3day")
    parser.add_argument("--include-live", action="store_true",
                        help="Stage already computed local live GDAS outputs into the presentation folder")
    parser.add_argument("--jan-data", default="data/datasets/multires_krsk_19f_jan2023_interp")
    parser.add_argument("--wrf-json", default="aaaa/wrf_krasnoyarsk/wrf_d03_jan2023.json")
    parser.add_argument("--out-dir", default="results/presentation_refresh")
    parser.add_argument("--artifact-store-dir", default=None,
                        help="Directory for heavy .pt bundles. If omitted and /data exists, uses /data/graphcast-lite/presentation_cache/<out-dir>/artifacts")
    parser.add_argument("--metric-samples", type=int, default=200)
    parser.add_argument("--artifact-samples", type=int, default=8)
    parser.add_argument("--ar-steps", type=int, default=4)
    parser.add_argument("--station-fraction", type=float, default=0.10)
    parser.add_argument("--nudging-alpha", type=float, default=0.5)
    parser.add_argument("--oi-sigma-b", type=float, default=0.8)
    parser.add_argument("--oi-sigma-o", type=float, default=0.5)
    parser.add_argument("--oi-corr-len", type=float, default=300000.0)
    parser.add_argument("--wrf-sample", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse existing logs/artifacts if the expected outputs are already present")
    args = parser.parse_args()

    freeze_exp = resolve_path(args.freeze_exp)
    nofreeze_exp = resolve_path(args.nofreeze_exp)
    main_data = resolve_path(args.main_data)
    global_exp = resolve_path(args.global_exp)
    global_data = resolve_path(args.global_data)
    live_dir = resolve_path(args.live_dir)
    jan_data = resolve_path(args.jan_data)
    wrf_json = resolve_path(args.wrf_json)
    out_dir = resolve_path(args.out_dir)
    logs_dir = out_dir / "logs"
    slides_dir = out_dir / "slides"
    artifact_dir = out_dir / "artifacts"
    da_dir = out_dir / "da"
    global_dir = out_dir / "global_3day"

    artifact_store_dir = resolve_path(args.artifact_store_dir) if args.artifact_store_dir else default_artifact_store(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    da_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(freeze_exp, nofreeze_exp, main_data, jan_data, wrf_json)
    check_experiment_dir(global_exp, "global coarse experiment", preflight.failures, preflight.checks)
    check_dataset_dir(global_data, "global coarse dataset", preflight.failures, preflight.checks)
    if args.include_live:
        if live_dir.exists():
            preflight.checks.append(f"OK live 3-day directory: {live_dir}")
        else:
            preflight.warnings.append(f"Live 3-day directory missing, live staging will be skipped: {live_dir}")
    preflight.checks.append(f"Heavy artifact storage: {artifact_store_dir}")
    write_preflight_report(logs_dir / "preflight.log", preflight)
    print_preflight_report(preflight)
    if not preflight.ok:
        raise SystemExit(f"Preflight failed. Fix required inputs and rerun. See {logs_dir / 'preflight.log'}")

    freeze_metrics_log = logs_dir / "freeze6_metrics.log"
    nofreeze_metrics_log = logs_dir / "nofreeze_metrics.log"
    global_metrics_log = global_dir / "global_metrics_3day.log"
    freeze_bundle = materialize_artifact_path(artifact_dir, artifact_store_dir, "freeze6_artifacts.pt")
    nofreeze_bundle = materialize_artifact_path(artifact_dir, artifact_store_dir, "nofreeze_artifacts.pt")
    jan_bundle = materialize_artifact_path(artifact_dir, artifact_store_dir, "freeze6_jan2023_wrf.pt")

    run_logged_if_needed(
        "freeze6 metrics",
        build_predict_cmd(freeze_exp, main_data, None, CITY_BBOX, args.ar_steps, args.metric_samples, no_save=True),
        freeze_metrics_log,
        outputs=[freeze_metrics_log],
        reuse_existing=args.reuse_existing,
    )
    run_logged_if_needed(
        "nofreeze metrics",
        build_predict_cmd(nofreeze_exp, main_data, None, CITY_BBOX, args.ar_steps, args.metric_samples, no_save=True),
        nofreeze_metrics_log,
        outputs=[nofreeze_metrics_log],
        reuse_existing=args.reuse_existing,
    )

    run_logged_if_needed(
        "global coarse 3-day metrics",
        build_predict_cmd(
            global_exp,
            global_data,
            None,
            tuple(args.global_region),
            args.global_ar_steps,
            args.global_max_samples,
            no_save=True,
        ),
        global_metrics_log,
        outputs=[global_metrics_log],
        reuse_existing=args.reuse_existing,
    )

    run_logged_if_needed(
        "freeze6 artifact bundle",
        build_predict_cmd(freeze_exp, main_data, freeze_bundle, CITY_BBOX, args.ar_steps, args.artifact_samples),
        logs_dir / "freeze6_artifacts.log",
        outputs=[freeze_bundle, logs_dir / "freeze6_artifacts.log"],
        reuse_existing=args.reuse_existing,
    )
    run_logged_if_needed(
        "nofreeze artifact bundle",
        build_predict_cmd(nofreeze_exp, main_data, nofreeze_bundle, CITY_BBOX, args.ar_steps, args.artifact_samples),
        logs_dir / "nofreeze_artifacts.log",
        outputs=[nofreeze_bundle, logs_dir / "nofreeze_artifacts.log"],
        reuse_existing=args.reuse_existing,
    )

    run_logged_if_needed(
        "freeze6 plots",
        [
            sys.executable,
            "scripts/plot_region_multires.py",
            str(freeze_bundle),
            "--data-dir",
            str(main_data),
            "--region",
            str(CITY_BBOX[0]),
            str(CITY_BBOX[1]),
            str(CITY_BBOX[2]),
            str(CITY_BBOX[3]),
            "--out-dir",
            str(out_dir / "plots_freeze6"),
            "--vars",
            "t2m",
            "10u",
            "10v",
            "sp",
            "--marker-size",
            "18",
        ],
        logs_dir / "freeze6_plots.log",
        outputs=[logs_dir / "freeze6_plots.log", out_dir / "plots_freeze6"],
        reuse_existing=args.reuse_existing,
    )
    run_logged_if_needed(
        "nofreeze plots",
        [
            sys.executable,
            "scripts/plot_region_multires.py",
            str(nofreeze_bundle),
            "--data-dir",
            str(main_data),
            "--region",
            str(CITY_BBOX[0]),
            str(CITY_BBOX[1]),
            str(CITY_BBOX[2]),
            str(CITY_BBOX[3]),
            "--out-dir",
            str(out_dir / "plots_nofreeze"),
            "--vars",
            "t2m",
            "10u",
            "10v",
            "sp",
            "--marker-size",
            "18",
        ],
        logs_dir / "nofreeze_plots.log",
        outputs=[logs_dir / "nofreeze_plots.log", out_dir / "plots_nofreeze"],
        reuse_existing=args.reuse_existing,
    )

    wrf_summary = None
    if jan_data.exists() and wrf_json.exists():
        run_logged_if_needed(
            "freeze6 Jan2023 WRF bundle",
            build_predict_cmd(freeze_exp, jan_data, jan_bundle, CITY_BBOX, args.ar_steps, max(args.artifact_samples, args.wrf_sample + 1), split="all"),
            logs_dir / "freeze6_jan2023.log",
            outputs=[jan_bundle, logs_dir / "freeze6_jan2023.log"],
            reuse_existing=args.reuse_existing,
        )
        run_logged_if_needed(
            "WRF comparison",
            [
                sys.executable,
                "scripts/compare_wrf.py",
                "--predictions",
                str(jan_bundle),
                "--data-dir",
                str(jan_data),
                "--wrf-path",
                str(wrf_json),
                "--experiment-dir",
                str(freeze_exp),
                "--ar-steps",
                str(args.ar_steps),
                "--wrf-sample",
                str(args.wrf_sample),
            ],
            logs_dir / "wrf_compare.log",
            outputs=[logs_dir / "wrf_compare.log"],
            reuse_existing=args.reuse_existing,
        )
        wrf_summary = logs_dir / "wrf_compare.log"
    else:
        skip_reason = []
        if not jan_data.exists():
            skip_reason.append(f"missing {jan_data}")
        if not wrf_json.exists():
            skip_reason.append(f"missing {wrf_json}")
        write_summary(logs_dir / "wrf_compare.log", ["WRF step skipped:", *skip_reason])
        wrf_summary = logs_dir / "wrf_compare.log"

    freeze_data = load_bundle(freeze_bundle, main_data)
    nofreeze_data = load_bundle(nofreeze_bundle, main_data)

    roi_indices, station_indices = build_sparse_observations(freeze_data, args.station_fraction, args.seed)
    write_summary(
        da_dir / "stations.txt",
        [
            f"ROI nodes: {len(roi_indices)}",
            f"Selected stations: {len(station_indices)}",
            f"Station fraction: {args.station_fraction:.3f}",
        ],
    )

    kalman_gain = build_oi_gain(
        freeze_data,
        roi_indices,
        station_indices,
        sigma_b=args.oi_sigma_b,
        sigma_o=args.oi_sigma_o,
        corr_len_m=args.oi_corr_len,
    )

    keep_in_memory = {"nudge_all", "oi_all", "oi_surf", "oi_temp", "oi_dyn"}
    da_outputs: dict[str, np.ndarray] = {"freeze6": freeze_data.predictions, "nofreeze": nofreeze_data.predictions}
    for group_name in OBS_GROUPS:
        channel_indices = build_observation_channels(freeze_data.var_names, group_name)
        nudged = apply_offline_nudging(freeze_data, station_indices, channel_indices, args.nudging_alpha)
        oi_pred = apply_roi_oi(freeze_data, roi_indices, station_indices, channel_indices, kalman_gain)
        save_region_npz(da_dir / f"nudge_{group_name}.npz", freeze_data, nudged, freeze_data.city_mask, f"nudge_{group_name}")
        save_region_npz(da_dir / f"oi_{group_name}.npz", freeze_data, oi_pred, freeze_data.city_mask, f"oi_{group_name}")
        if f"nudge_{group_name}" in keep_in_memory:
            da_outputs[f"nudge_{group_name}"] = nudged
        if f"oi_{group_name}" in keep_in_memory:
            da_outputs[f"oi_{group_name}"] = oi_pred

    sample_idx = min(freeze_data.predictions.shape[0] - 1, 3)
    plot_truth_and_method_errors(
        slides_dir / "t2m_method_errors_24h.png",
        freeze_data,
        {
            "freeze6": da_outputs["freeze6"],
            "nofreeze": da_outputs["nofreeze"],
            "nudge_all": da_outputs["nudge_all"],
            "oi_all": da_outputs["oi_all"],
        },
        "t2m",
        sample_idx,
        min(args.ar_steps - 1, 3),
    )
    plot_truth_and_method_errors(
        slides_dir / "t2m_da_groups_24h.png",
        freeze_data,
        {
            "freeze6": da_outputs["freeze6"],
            "oi_all": da_outputs["oi_all"],
            "oi_surf": da_outputs["oi_surf"],
            "oi_temp": da_outputs["oi_temp"],
            "oi_dyn": da_outputs["oi_dyn"],
        },
        "t2m",
        sample_idx,
        min(args.ar_steps - 1, 3),
    )
    plot_region_rmse_summary(
        slides_dir / "t2m_region_rmse_summary.png",
        freeze_data,
        {
            "freeze6": da_outputs["freeze6"],
            "nofreeze": da_outputs["nofreeze"],
            "nudge_all": da_outputs["nudge_all"],
            "oi_all": da_outputs["oi_all"],
        },
        "t2m",
    )
    plot_horizon_rmse_curves(
        slides_dir / f"t2m_rmse_curve_{args.ar_steps * 6:03d}h.png",
        freeze_data,
        {
            "freeze6": da_outputs["freeze6"],
            "nofreeze": da_outputs["nofreeze"],
            "nudge_all": da_outputs["nudge_all"],
            "oi_all": da_outputs["oi_all"],
        },
        "t2m",
        title=f"City-region t2m RMSE vs horizon (0..{args.ar_steps * 6}h)",
    )
    plot_horizon_rmse_curves(
        slides_dir / f"t2m_da_groups_curve_{args.ar_steps * 6:03d}h.png",
        freeze_data,
        {
            "freeze6": da_outputs["freeze6"],
            "oi_all": da_outputs["oi_all"],
            "oi_surf": da_outputs["oi_surf"],
            "oi_temp": da_outputs["oi_temp"],
            "oi_dyn": da_outputs["oi_dyn"],
        },
        "t2m",
        title=f"City-region t2m RMSE by DA variable group (0..{args.ar_steps * 6}h)",
    )

    staged_live_dir = stage_live_outputs(live_dir, out_dir, slides_dir) if args.include_live else None

    summary_lines = [
        "# Presentation Refresh Outputs",
        "",
        f"Main metrics log (freeze6): {format_path(freeze_metrics_log)}",
        f"Main metrics log (nofreeze): {format_path(nofreeze_metrics_log)}",
        f"Global coarse 3-day metrics: {format_path(global_metrics_log)}",
        f"Freeze6 artifact bundle: {format_path(freeze_bundle)}",
        f"Nofreeze artifact bundle: {format_path(nofreeze_bundle)}",
        f"Heavy artifact storage: {format_path(artifact_store_dir)}",
        f"WRF summary: {format_path(wrf_summary)}",
        f"Freeze6 plots: {format_path(out_dir / 'plots_freeze6')}",
        f"Nofreeze plots: {format_path(out_dir / 'plots_nofreeze')}",
        f"DA outputs: {format_path(da_dir)}",
        f"Slide assets: {format_path(slides_dir)}",
        f"Live 3-day staged outputs: {format_path(staged_live_dir)}",
        "",
        "Model notes:",
        "- freeze6 and nofreeze are both evaluated for presentation tables; freeze6 remains the stronger multires baseline.",
        f"- Long-horizon DA plots cover 0..{args.ar_steps * 6}h using the same fresh freeze6 forecasts.",
        f"- Global coarse baseline is evaluated separately for 3 days (0..{args.global_ar_steps * 6}h).",
        "",
        "DA notes:",
        f"- Observations sampled on ROI only ({len(roi_indices)} ROI nodes, {len(station_indices)} stations)",
        f"- Nudging alpha={args.nudging_alpha}",
        f"- OI sigma_b={args.oi_sigma_b}, sigma_o={args.oi_sigma_o}, L={args.oi_corr_len}",
        "- DA is applied to fresh freeze6 forecasts produced in this run, restricted to the regional ROI to keep OI tractable.",
    ]
    write_summary(out_dir / "README.md", summary_lines)
    print(f"\nAll outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()