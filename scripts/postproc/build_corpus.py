#!/usr/bin/env python3
"""Build per-station GNN-forecast corpus joined with ISD-Lite observations.

Input  : GNN checkpoint (Russia 33f) + global/regional base & extra ERA5 datasets +
         multires_russia_33f scalers/coords (copied from training VM) +
         ISD-Lite station files.
Output : Parquet with one row per (station, init_time, lead_h) — features for a
         residual MLP postprocessor (variant A2 of the postproc RFC).

Strategy
--------
* Build the same flat multires node ordering used at training time by reading
  ``coords.npz`` from the training dataset directory (no need to keep the 54 GB
  ``data.npy`` around — input is reconstructed on the fly per timestep).
* For each requested init_time:
    1. Build a (1, N, 33) normalized input: 19 base channels from
       global+regional base ``data.npy`` + 10 plev channels bilinear-sampled
       from global/regional ``data_extra.npy`` + 4 sin/cos forcings computed
       analytically.
    2. Roll the model forward AR steps (each step = +6 h). At every requested
       lead, sample 13 GNN variables at each of the top-N stations.
    3. Between AR steps, carry forward static channels [7=z_surf, 8=lsm] and
       forcing channels [29..32] (these are exogenous; the model predicts
       them but the truth is known).
* After all inits: build a DataFrame, join with ISD-Lite obs on
  (station_usaf, valid_time), compute derived/static/temporal features,
  drop rows without observations, write Parquet partitioned by year.

Usage on v3 (after `git pull` and ERA5 base decompression):

    python scripts/postproc/build_corpus.py \\
        --experiment-dir experiments/multires_russia_33f_v3_noroi \\
        --multires-dir   /data/datasets/multires_russia_33f \\
        --global-base    /data/datasets/wb2_512x256_19f_ar \\
        --regional-base  /data/datasets/region_russia_645x165_19f_2010-2021_025deg \\
        --global-extra   /data/datasets/global_512x256_extra_2010-2021_07deg \\
        --regional-extra /data/datasets/region_russia_645x165_extra_2010-2021_025deg \\
        --stations-json  data/russia_mos_stations.json \\
        --isd-dir        /data/datasets/isd_lite_russia \\
        --top-stations 50 \\
        --years 2018 2020 \\
        --init-hours 0 12 \\
        --leads-h 6 12 18 24 \\
        --out-parquet data/postproc/corpus_v1.parquet \\
        --device cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig
from src.main import load_model_from_experiment_config
from src.postprocessing.corpus_math import (
    bilinear_sample,
    compute_forcing,
    dewpoint_depression_K,
    solar_elevation,
    wind_components,
)
from src.postprocessing.geometry import (
    COHERENCE_MIN_REGIONAL,
    COHERENCE_MIN_WHOLE,
    field_coherence,
    neighbour_indices,
    snap_miss,
)
from src.utils import load_from_json_file

# ── ROI must match the one used to build multires_russia_33f ──────────────────
ROI = (41.0, 82.0, 19.0, 180.0)  # lat_min, lat_max, lon_min, lon_max

# Static + forcing channels (33ch layout, see multires_russia_33f/variables.json)
STATIC_CHANNELS = (7, 8)                  # z_surf, lsm
FORCING_CHANNELS = (29, 30, 31, 32)       # sin_hour, cos_hour, sin_doy, cos_doy

# Variables we sample at stations (13 vars used by the postproc model)
SAMPLE_VARS = [
    "t2m", "10u", "10v", "msl", "sp",
    "t@850", "u@850", "v@850",
    "t@500", "q@850", "z@500",
    "u@1000", "v@1000",
]
RENAME_FOR_PARQUET = {
    "t2m": "gnn_t2m",  "10u": "gnn_u10", "10v": "gnn_v10",
    "msl": "gnn_msl",  "sp":  "gnn_sp",
    "t@850": "gnn_t850", "u@850": "gnn_u850", "v@850": "gnn_v850",
    "t@500": "gnn_t500", "q@850": "gnn_q850", "z@500": "gnn_z500",
    "u@1000": "gnn_u1000", "v@1000": "gnn_v1000",
}


# ── small helpers ─────────────────────────────────────────────────────────────
def load_isd_station(usaf: str, wban: str, isd_dir: Path,
                     years: range) -> pd.DataFrame:
    """Return DataFrame with valid_time, obs_t2m_C, obs_ws, obs_wd."""
    rows: list[dict] = []
    for year in years:
        fpath = isd_dir / f"{usaf}-{wban}-{year}.gz"
        if not fpath.exists():
            continue
        try:
            with gzip.open(fpath, "rt") as f:
                text = f.read()
        except Exception:
            continue
        for line in text.strip().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                y, m, d, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                temp_raw = int(parts[4])
            except ValueError:
                continue
            if temp_raw == -9999:
                continue
            rec = {
                "valid_time": pd.Timestamp(year=y, month=m, day=d, hour=h),
                "obs_t2m_C": temp_raw / 10.0,
            }
            if len(parts) >= 9:
                try:
                    wd_raw = int(parts[7])
                    ws_raw = int(parts[8])
                    if ws_raw != -9999:
                        rec["obs_ws"] = ws_raw / 10.0
                    if wd_raw != -9999:
                        rec["obs_wd"] = float(wd_raw)
                except ValueError:
                    pass
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["valid_time", "obs_t2m_C", "obs_ws", "obs_wd"])
    return pd.DataFrame(rows)


# ── multires input builder ────────────────────────────────────────────────────
class MultiresInputBuilder:
    """Builds a normalized (N, 33) frame for a given timestep on the fly."""

    def __init__(self, multires_dir: Path,
                 global_base: Path | None, regional_base: Path | None,
                 global_extra: Path, regional_extra: Path,
                 merged_base: Path | None = None):
        self.merged_base = merged_base
        # 1. multires coords (defines node order). If absent, reconstruct from
        #    global+regional grids + ROI (same logic as build_multires_dataset.py).
        coords_path = multires_dir / "coords.npz"
        # Порядок узлов одинаков у 33-канального датасета и у слитого 19-канального
        # источника — 33-канальный собирается из него простым дописыванием каналов
        # и копирует coords.npz как есть. Поэтому если в 33f координат нет
        # (случалось после пересборки датасета), берём их у слитого источника.
        if not coords_path.exists() and merged_base is not None:
            alt = merged_base / "coords.npz"
            if alt.exists():
                print(f"[builder] в {multires_dir} нет coords.npz, беру из {alt}", flush=True)
                coords_path = alt
        if coords_path.exists():
            coords = np.load(coords_path)
            self.node_lats = coords["latitude"].astype(np.float32)
            self.node_lons = coords["longitude"].astype(np.float32)
            if "is_regional" in coords.files:
                self.is_regional = coords["is_regional"].astype(bool)
            else:
                # legacy: first n_global_kept are global, rest are regional
                info = json.loads((multires_dir / "dataset_info.json").read_text())
                n_gk = info["n_global_kept"]
                self.is_regional = np.zeros(len(self.node_lats), dtype=bool)
                self.is_regional[n_gk:] = True
            print(f"[builder] loaded coords from {coords_path}", flush=True)
        else:
            print(f"[builder] no coords at {coords_path}; reconstructing from "
                  f"global+regional grids + ROI {ROI}", flush=True)
            if global_base is None or regional_base is None:
                raise SystemExit(
                    "координат нет ни в multires-каталоге, ни в слитом источнике, "
                    "а восстановить их не из чего: для этого нужны --global-base и "
                    "--regional-base. Проверь, что в датасете есть coords.npz.")
            gc_tmp = np.load(global_base / "coords.npz")
            rc_tmp = np.load(regional_base / "coords.npz")
            g_lats_n = gc_tmp["latitude"].astype(np.float64)
            g_lons_n = gc_tmp["longitude"].astype(np.float64)
            r_lats_n = rc_tmp["latitude"].astype(np.float64)
            r_lons_n = rc_tmp["longitude"].astype(np.float64)
            lat_min, lat_max, lon_min, lon_max = ROI
            g_lon_mesh, g_lat_mesh = np.meshgrid(g_lons_n, g_lats_n)  # (nlat, nlon)
            in_roi = ((g_lat_mesh >= lat_min) & (g_lat_mesh <= lat_max)
                      & (g_lon_mesh >= lon_min) & (g_lon_mesh <= lon_max))
            keep = ~in_roi
            g_flat_lats = g_lat_mesh[keep]
            g_flat_lons = g_lon_mesh[keep]
            r_lon_mesh, r_lat_mesh = np.meshgrid(r_lons_n, r_lats_n)
            r_flat_lats = r_lat_mesh.reshape(-1)
            r_flat_lons = r_lon_mesh.reshape(-1)
            self.node_lats = np.concatenate([g_flat_lats, r_flat_lats]).astype(np.float32)
            self.node_lons = np.concatenate([g_flat_lons, r_flat_lons]).astype(np.float32)
            self.is_regional = np.zeros(len(self.node_lats), dtype=bool)
            self.is_regional[len(g_flat_lats):] = True
            # save it so subsequent runs can load fast
            np.savez(coords_path,
                     latitude=self.node_lats, longitude=self.node_lons,
                     is_regional=self.is_regional)
            print(f"[builder] saved reconstructed coords to {coords_path} "
                  f"(global_kept={len(g_flat_lats)}, regional={len(r_flat_lats)})",
                  flush=True)

        self.N = len(self.node_lats)
        assert self.is_regional.shape == (self.N,)
        self.n_global_kept = int((~self.is_regional).sum())
        self.n_regional = int(self.is_regional.sum())

        # 2. 33ch scalers
        sc_path = multires_dir / "scalers.npz"
        if not sc_path.exists():
            raise SystemExit(
                f"нет {sc_path} — без нормировок корпус собрать нельзя. "
                f"Проверь, полностью ли собран {multires_dir}.")
        sc = np.load(sc_path)
        self.s_mean = sc["mean"].astype(np.float32)
        self.s_std = sc["std"].astype(np.float32)
        assert self.s_mean.shape == (33,) and self.s_std.shape == (33,)

        # 3-4. Базовые 19 каналов.
        #
        # Красноярская линия: 19-канальная часть уже слита в один плоский
        # массив (multires_krsk_19f_merge) в порядке узлов, поэтому кадр
        # читается напрямую. Общероссийская собиралась из отдельных
        # глобальной и региональной сеток — для неё прежний путь.
        if merged_base is not None:
            m_info = json.loads((merged_base / "dataset_info.json").read_text())
            assert m_info.get("flat", False), f"{merged_base} не плоский датасет"
            self.mb_T = m_info["n_time"]
            mb_N = m_info["n_nodes"]
            mb_F = m_info.get("n_feat_base", m_info["n_feat"])
            assert mb_N == self.N, (
                f"узлов в слитом источнике {mb_N}, в координатах multires {self.N}")
            self.mb_data = np.memmap(merged_base / "data.npy", dtype=np.float16,
                                     mode="r", shape=(self.mb_T, mb_N, mb_F))
            self.mb_time_start = datetime.fromisoformat(
                str(m_info.get("time_start", "2010-01-01"))[:19].replace(" ", "T"))
            print(f"[builder] слитый источник: {self.mb_T} сроков x {mb_N} узлов "
                  f"x {mb_F} каналов, начало {self.mb_time_start:%Y-%m-%d}", flush=True)
        else:
            # 3. base global (T, n_lon, n_lat, 19) memmap
            g_info = json.loads((global_base / "dataset_info.json").read_text())
            self.gb_T = g_info["n_time"]
            self.gb_nlon = g_info["n_lon"]
            self.gb_nlat = g_info["n_lat"]
            assert g_info["n_feat"] == 19, f"global base must be 19f, got {g_info['n_feat']}"
            self.gb_time_start = datetime.fromisoformat(g_info["time_start"])
            self.gb_data = np.memmap(
                global_base / "data.npy", dtype=np.float16, mode="r",
                shape=(self.gb_T, self.gb_nlon, self.gb_nlat, 19),
            )
            gc = np.load(global_base / "coords.npz")
            # Native orientation of base global (order in data.npy)
            self.gb_lats_native = gc["latitude"].astype(np.float64)
            self.gb_lons_native = gc["longitude"].astype(np.float64)

            # 4. base regional (T, n_lon, n_lat, 19) memmap
            r_info = json.loads((regional_base / "dataset_info.json").read_text())
            self.rb_T = r_info["n_time"]
            self.rb_nlon = r_info["n_lon"]
            self.rb_nlat = r_info["n_lat"]
            assert r_info["n_feat"] == 19, f"regional base must be 19f, got {r_info['n_feat']}"
            self.rb_time_start = datetime.fromisoformat(r_info["time_start"])
            self.rb_data = np.memmap(
                regional_base / "data.npy", dtype=np.float16, mode="r",
                shape=(self.rb_T, self.rb_nlon, self.rb_nlat, 19),
            )
            assert self.rb_nlat * self.rb_nlon == self.n_regional, (
                f"regional flat size {self.rb_nlat * self.rb_nlon} != "
                f"coords.npz n_regional {self.n_regional}"
            )

        # 5. global extra (T, n_lon, n_lat, 10) memmap + scalers + coords
        ge_info_path = global_extra / "dataset_info_extra.json"
        if not ge_info_path.exists():
            ge_info_path = global_extra / "dataset_info.json"
        ge_info = json.loads(ge_info_path.read_text())
        self.ge_data = np.memmap(
            global_extra / "data_extra.npy", dtype=np.float16, mode="r",
            shape=(ge_info["n_time"], ge_info["n_lon"], ge_info["n_lat"], 10),
        )
        self.ge_time_start = datetime.fromisoformat(ge_info["time_start"])
        gec = np.load(global_extra / "coords.npz")
        self.ge_lats = gec["latitude"].astype(np.float64)
        self.ge_lons = gec["longitude"].astype(np.float64)
        self.ge_lat_flipped = False
        if self.ge_lats[0] > self.ge_lats[-1]:
            self.ge_lats = self.ge_lats[::-1]
            self.ge_lat_flipped = True

        # 6. regional extra (T, n_lon, n_lat, 10) memmap + coords
        re_info_path = regional_extra / "dataset_info_extra.json"
        if not re_info_path.exists():
            re_info_path = regional_extra / "dataset_info.json"
        re_info = json.loads(re_info_path.read_text())
        self.re_data = np.memmap(
            regional_extra / "data_extra.npy", dtype=np.float16, mode="r",
            shape=(re_info["n_time"], re_info["n_lon"], re_info["n_lat"], 10),
        )
        self.re_time_start = datetime.fromisoformat(re_info["time_start"])
        rec = np.load(regional_extra / "coords.npz")
        self.re_lats = rec["latitude"].astype(np.float64)
        self.re_lons = rec["longitude"].astype(np.float64)
        self.re_lat_flipped = False
        if self.re_lats[0] > self.re_lats[-1]:
            self.re_lats = self.re_lats[::-1]
            self.re_lat_flipped = True

        # Indices of regional/global nodes (for routing extras)
        self.reg_node_idx = np.where(self.is_regional)[0]
        self.glob_node_idx = np.where(~self.is_regional)[0]
        self.reg_node_lon = self.node_lons[self.reg_node_idx].astype(np.float64)
        self.reg_node_lat = self.node_lats[self.reg_node_idx].astype(np.float64)
        self.glob_node_lon = self.node_lons[self.glob_node_idx].astype(np.float64)
        self.glob_node_lat = self.node_lats[self.glob_node_idx].astype(np.float64)

        # Таблицы соответствия узлов исходным сеткам нужны только тогда,
        # когда базовые каналы собираются из двух сеток. При слитом источнике
        # кадр уже в порядке узлов, и сеток этих на машине может не быть вовсе.
        if merged_base is None:
            # Per-node index tables into the native global / regional base grids.
            # Robust to whatever node order was used at multires build time —
            # snaps each node to its exact source cell by coordinate match.
            self._gb_lon_idx = np.array([
                int(np.argmin(np.abs(self.gb_lons_native - lon)))
                for lon in self.glob_node_lon
            ], dtype=np.int64)
            self._gb_lat_idx = np.array([
                int(np.argmin(np.abs(self.gb_lats_native - lat)))
                for lat in self.glob_node_lat
            ], dtype=np.int64)
            self._check_snap(self.gb_lons_native, self.glob_node_lon, "долготе",
                             "глобальной")
            self._check_snap(self.gb_lats_native, self.glob_node_lat, "широте",
                             "глобальной")
            rb_coords = np.load(regional_base / "coords.npz")
            self.rb_lats_native = rb_coords["latitude"].astype(np.float64)
            self.rb_lons_native = rb_coords["longitude"].astype(np.float64)
            self._rb_lon_idx = np.array([
                int(np.argmin(np.abs(self.rb_lons_native - lon)))
                for lon in self.reg_node_lon
            ], dtype=np.int64)
            self._rb_lat_idx = np.array([
                int(np.argmin(np.abs(self.rb_lats_native - lat)))
                for lat in self.reg_node_lat
            ], dtype=np.int64)
            # Sanity: max snap error must be within one cell.
            gb_dlon = float(np.abs(self.gb_lons_native[1] - self.gb_lons_native[0]))
            gb_dlat = float(np.abs(self.gb_lats_native[1] - self.gb_lats_native[0]))
            e_glon = float(np.max(np.abs(
                self.gb_lons_native[self._gb_lon_idx] - self.glob_node_lon
            )))
            e_glat = float(np.max(np.abs(
                self.gb_lats_native[self._gb_lat_idx] - self.glob_node_lat
            )))
            assert e_glon < gb_dlon and e_glat < gb_dlat, (
                f"global node snap error: lon={e_glon} (cell={gb_dlon}), "
                f"lat={e_glat} (cell={gb_dlat})"
            )
            if self.reg_node_idx.size > 0:
                rb_dlon = float(np.abs(self.rb_lons_native[1] - self.rb_lons_native[0]))
                rb_dlat = float(np.abs(self.rb_lats_native[1] - self.rb_lats_native[0]))
                e_rlon = float(np.max(np.abs(
                    self.rb_lons_native[self._rb_lon_idx] - self.reg_node_lon
                )))
                e_rlat = float(np.max(np.abs(
                    self.rb_lats_native[self._rb_lat_idx] - self.reg_node_lat
                )))
                assert e_rlon < rb_dlon and e_rlat < rb_dlat, (
                    f"regional node snap error: lon={e_rlon} (cell={rb_dlon}), "
                    f"lat={e_rlat} (cell={rb_dlat})"
                )

        # base_T есть только у раздельных сеток; у слитого источника — mb_T.
        _bt = self.mb_T if merged_base is not None else self.gb_T
        print(
            f"[builder] N={self.N}  global_kept={self.n_global_kept}  "
            f"regional={self.n_regional}  base_T={_bt}",
            flush=True,
        )

    # — per-timestep loaders —
    @staticmethod
    def _check_snap(native: np.ndarray, wanted: np.ndarray, axis: str,
                    grid: str, tol_deg: float = 1e-3) -> None:
        """Убедиться, что узлы легли на ячейки сетки, а не рядом с ними."""
        worst = snap_miss(native, wanted)
        if worst > tol_deg:
            raise SystemExit(
                f"[узлы] по {axis} узлы {grid} сетки не совпали с ячейками: "
                f"наибольший промах {worst:.4f}° при допуске {tol_deg}°. "
                f"Кадр собрался бы из соседних ячеек, и заметить это было бы "
                f"нечем. Проверь, из той ли сетки взяты координаты.")

    def _time_idx(self, ds_time_start: datetime, dt: datetime, T_max: int) -> int:
        delta = dt - ds_time_start
        if delta.total_seconds() % (6 * 3600) != 0:
            raise ValueError(f"{dt} is not aligned to 6-h grid of {ds_time_start}")
        idx = int(delta.total_seconds() // (6 * 3600))
        if idx < 0 or idx >= T_max:
            raise IndexError(f"time {dt} out of range [{ds_time_start}, +{T_max} steps)")
        return idx

    def base_frame(self, dt: datetime) -> np.ndarray:
        """Return (N, 19) physical-units base channels, in multires node order."""
        if self.merged_base is not None:
            # Слитый источник уже в порядке узлов — ни выборки, ни склейки.
            mi = self._time_idx(self.mb_time_start, dt, self.mb_T)
            return np.asarray(self.mb_data[mi], dtype=np.float32)
        gi = self._time_idx(self.gb_time_start, dt, self.gb_T)
        ri = self._time_idx(self.rb_time_start, dt, self.rb_T)
        # global: gb_data[gi] is (n_lon, n_lat, 19)
        g_frame = np.asarray(self.gb_data[gi], dtype=np.float32)
        g_vals = g_frame[self._gb_lon_idx, self._gb_lat_idx, :]  # (n_global_kept, 19)
        # regional
        r_frame = np.asarray(self.rb_data[ri], dtype=np.float32)
        r_vals = r_frame[self._rb_lon_idx, self._rb_lat_idx, :]  # (n_regional, 19)
        out = np.empty((self.N, 19), dtype=np.float32)
        out[self.glob_node_idx] = g_vals
        out[self.reg_node_idx] = r_vals
        return out

    def plev_frame(self, dt: datetime) -> np.ndarray:
        """Return (N, 10) physical-units plev channels (bilinear from extras)."""
        gei = self._time_idx(self.ge_time_start, dt, self.ge_data.shape[0])
        rei = self._time_idx(self.re_time_start, dt, self.re_data.shape[0])
        g_chunk = np.array(self.ge_data[gei], dtype=np.float32)  # (n_lon, n_lat, 10)
        if self.ge_lat_flipped:
            g_chunk = g_chunk[:, ::-1, :]
        r_chunk = np.array(self.re_data[rei], dtype=np.float32)
        if self.re_lat_flipped:
            r_chunk = r_chunk[:, ::-1, :]
        out = np.empty((self.N, 10), dtype=np.float32)
        for ch in range(10):
            if self.glob_node_idx.size > 0:
                out[self.glob_node_idx, ch] = bilinear_sample(
                    g_chunk[:, :, ch], self.ge_lons, self.ge_lats,
                    self.glob_node_lon, self.glob_node_lat,
                )
            if self.reg_node_idx.size > 0:
                out[self.reg_node_idx, ch] = bilinear_sample(
                    r_chunk[:, :, ch], self.re_lons, self.re_lats,
                    self.reg_node_lon, self.reg_node_lat,
                )
        return out

    def check_node_order(self, frame: np.ndarray, ch: int) -> None:
        """Убедиться, что значения легли на свои узлы.

        Координаты узлов здесь восстановлены построением, а не прочитаны из
        слитого набора, поэтому стоит проверить, что порядок совпал. Опираемся
        на гладкость поля; вставку проверяем отдельно — она занимает 1,9 %
        узлов, и на общей связи её порча незаметна. Сам расчёт — в
        src/postprocessing/geometry.py, там же он и покрыт тестами.

        Проверка идёт по нормированному кадру: нормировка поканальная и
        линейная, а связь к линейному преобразованию нечувствительна.
        """
        whole = field_coherence(self.node_lats, self.node_lons, frame[:, ch])
        reg = (field_coherence(self.node_lats, self.node_lons, frame[:, ch],
                               mask=self.is_regional)
               if self.n_regional else float("nan"))
        print(f"[порядок узлов] связь с соседями: по всей сетке {whole:+.3f}, "
              f"во вставке {reg:+.3f}", flush=True)
        bad = (whole < COHERENCE_MIN_WHOLE
               or (self.n_regional and not np.isnan(reg)
                   and reg < COHERENCE_MIN_REGIONAL))
        if bad:
            raise SystemExit(
                "[порядок узлов] поле не выглядит гладким: значения легли не на "
                "те узлы. Скорее всего, разошёлся порядок между координатами и "
                "данными. Останавливаюсь, чтобы не считать корпус впустую.")

    def build_input(self, dt: datetime) -> np.ndarray:
        """Return (N, 33) NORMALIZED frame ready for the model."""
        base = self.base_frame(dt)            # (N, 19)
        plev = self.plev_frame(dt)            # (N, 10)
        forc = compute_forcing(dt)            # (4,)
        frame = np.empty((self.N, 33), dtype=np.float32)
        frame[:, :19] = base
        frame[:, 19:29] = plev
        frame[:, 29:33] = forc[None, :]
        # normalize
        return (frame - self.s_mean) / self.s_std

    def forcing_normalized(self, dt: datetime) -> np.ndarray:
        """Return normalized (4,) forcing for a given time (for AR carry-forward)."""
        forc = compute_forcing(dt)
        return (forc - self.s_mean[29:33]) / self.s_std[29:33]


# ── main pipeline ─────────────────────────────────────────────────────────────
def _coerce_int_arg(values, name):
    out: list[int] = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            raise SystemExit(f"--{name}: bad int '{v}'")
    return out



def save_table(df, out_path: Path) -> Path:
    """Пишет таблицу, откатываясь на pickle, если parquet недоступен.

    27.08.2026 два с лишним часа развёртки пропали на последней строке: в
    окружении виртуалки не оказалось pyarrow. Формат тут второстепенен —
    важно не потерять посчитанное.
    """
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except ImportError as e:
        alt = out_path.with_suffix(".pkl.gz")
        print(f"[save] parquet недоступен ({e.__class__.__name__}), пишу {alt}",
              flush=True)
        df.to_pickle(alt, compression="gzip")
        return alt


def load_partial(part_path: Path) -> list:
    """Собрать строки из черновика, дописанного кусками во время развёртки."""
    rows = []
    with open(part_path, "rb") as f:
        while True:
            try:
                rows.extend(pickle.load(f))
            except EOFError:
                break
            except pickle.UnpicklingError:
                # Обрыв посреди куска: машину могли выключить прямо на сбросе.
                # Целые куски уже прочитаны, их и берём — терять всё из-за
                # хвоста в пару килобайт незачем.
                print("[part] последний кусок оборван, беру прочитанное",
                      flush=True)
                break
    return rows


def check_writable(out_path: Path) -> None:
    """Проверяет запись ДО тяжёлого счёта, а не после."""
    probe = out_path.parent / (".write_probe" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    got = save_table(pd.DataFrame({"a": [1.0], "b": ["x"]}), probe)
    try:
        got.unlink()
    except FileNotFoundError:
        pass
    fmt = "parquet" if got.suffix == out_path.suffix else "pickle (parquet недоступен)"
    print(f"[save] проверка записи пройдена, формат: {fmt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--neighbours", type=int, default=0,
                    help="сколько ближайших узлов брать вокруг станции для "
                         "признаков окрестности (0 — не брать). Даёт модели "
                         "местный градиент поля и изрезанность рельефа")
    ap.add_argument("--from-partial", action="store_true",
                    help="не считать развёртку заново, а взять строки из "
                         "черновика <out>.partial.pkl и досшить с наблюдениями")
    ap.add_argument("--multires-dir", required=True,
                    help="Path with coords.npz + scalers.npz (33ch). data.npy not needed.")
    ap.add_argument("--merged-base", default=None,
                    help="плоский multires 19f (T, N, 19); заменяет --global-base/--regional-base")
    ap.add_argument("--global-base", default=None)
    ap.add_argument("--regional-base", default=None)
    ap.add_argument("--global-extra", required=True)
    ap.add_argument("--regional-extra", required=True)
    ap.add_argument("--stations-json", required=True)
    ap.add_argument("--isd-dir", required=True)
    ap.add_argument("--top-stations", type=int, default=50)
    ap.add_argument("--years", type=int, nargs=2, default=[2018, 2020],
                    help="Inclusive [start, end] year range for inits.")
    ap.add_argument("--init-hours", type=int, nargs="+", default=[0, 12])
    ap.add_argument("--leads-h", type=int, nargs="+",
                    default=[6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                    help="Forecast leads (hours, must be multiples of 6).")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-inits", type=int, default=0,
                    help="Cap on number of init times (0 = all). Useful for smoke tests.")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    leads = sorted(set(args.leads_h))
    for L in leads:
        if L % 6 != 0 or L <= 0:
            raise SystemExit(f"--leads-h must be positive multiples of 6, got {L}")
    max_ar = max(leads) // 6
    print(f"[cfg] leads={leads}  max_ar={max_ar}  init_hours={args.init_hours}  "
          f"years={args.years}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"[cfg] device={device}", flush=True)

    # 1. multires builder
    check_writable(Path(args.out_parquet))
    if not args.merged_base and not (args.global_base and args.regional_base):
        raise SystemExit("нужен либо --merged-base, либо пара --global-base/--regional-base")
    builder = MultiresInputBuilder(
        multires_dir=Path(args.multires_dir),
        global_base=Path(args.global_base) if args.global_base else None,
        regional_base=Path(args.regional_base) if args.regional_base else None,
        global_extra=Path(args.global_extra),
        regional_extra=Path(args.regional_extra),
        merged_base=Path(args.merged_base) if args.merged_base else None,
    )

    # 2. variables.json (33ch)
    var_names = json.loads((Path(args.multires_dir) / "variables.json").read_text())
    assert len(var_names) == 33
    sample_idx = [var_names.index(v) for v in SAMPLE_VARS]
    print(f"[cfg] sampling vars: {SAMPLE_VARS}  idx={sample_idx}", flush=True)

    # 3. load model
    exp_dir = Path(args.experiment_dir)
    exp_cfg = ExperimentConfig(**load_from_json_file(str(exp_dir / "config.json")))
    OBS = exp_cfg.data.obs_window_used
    C = exp_cfg.data.num_features_used
    assert C == 33, f"experiment must use 33 features, got {C}"
    assert OBS in (1, 2), f"this builder supports OBS in (1, 2), got {OBS}"
    use_residual = bool(getattr(exp_cfg, "use_residual", True))
    print(f"[cfg] OBS={OBS}  use_residual={use_residual}", flush=True)

    coords_pair = (builder.node_lats, builder.node_lons)
    region_bounds = None
    lat_span = float(builder.node_lats.max() - builder.node_lats.min())
    lon_span = float(builder.node_lons.max() - builder.node_lons.min())
    if lat_span < 90 and lon_span < 180:
        region_bounds = (float(builder.node_lats.min()), float(builder.node_lats.max()),
                         float(builder.node_lons.min()), float(builder.node_lons.max()))

    class _FakeMeta:
        num_longitudes = 0
        num_latitudes = 0
        flat_grid = True

    model = load_model_from_experiment_config(
        experiment_config=exp_cfg, device=device, dataset_metadata=_FakeMeta(),
        coordinates=coords_pair, region_bounds=region_bounds, flat_grid=True,
    )
    sd = torch.load(exp_dir / "best_model.pth", map_location="cpu", weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:
        print(f"[warn] missing keys: {len(miss)} (first 3: {miss[:3]})")
    if unexp:
        print(f"[warn] unexpected keys: {len(unexp)} (first 3: {unexp[:3]})")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}", flush=True)

    # 4. station selection (top-N by obs_per_day)
    stations_all: dict = json.loads(Path(args.stations_json).read_text())
    ranked = sorted(
        stations_all.items(),
        key=lambda kv: kv[1].get("obs_per_day", 0.0),
        reverse=True,
    )
    selected = ranked[: args.top_stations]
    print(f"[stations] using top-{len(selected)} (best obs/day = "
          f"{selected[0][1].get('obs_per_day'):.2f}, worst = "
          f"{selected[-1][1].get('obs_per_day'):.2f})", flush=True)

    # map each station → nearest multires node
    station_meta = []  # list of dict
    for usaf, info in selected:
        lat = info["lat"]
        lon = info["lon"] % 360.0  # multires uses 0..360
        d2 = (builder.node_lats - lat) ** 2 + (builder.node_lons - lon) ** 2
        gidx = int(np.argmin(d2))
        dist_km = float(np.sqrt(d2[gidx]) * 111.0)
        # Окрестность. Ближайший узел выбираем прежним правилом, чтобы корпус
        # остался сравнимым со старым; а вот НАБОР соседей берём по честному
        # расстоянию, со сжатием долготы на косинус широты — на 55° градус
        # долготы вдвое короче градуса широты, и без поправки в соседи попадал
        # бы вытянутый по долготе ряд вместо круга.
        nb_idx = neighbour_indices(builder.node_lats, builder.node_lons,
                                   lat, lon, args.neighbours)
        station_meta.append({
            "nb_idx": nb_idx,
            "usaf": usaf,
            "wban": info.get("wban", "99999"),
            "name": info.get("name", ""),
            "lat": float(info["lat"]),
            "lon": float(info["lon"]),
            "elev": float(info.get("elev", 0.0)),
            "grid_idx": gidx,
            "dist_km": dist_km,
        })
    avg_d = np.mean([s["dist_km"] for s in station_meta])
    print(f"[stations] mean station→node distance = {avg_d:.1f} km", flush=True)

    # 5. preload ISD-Lite for all stations (cover the lead horizon too)
    isd_dir = Path(args.isd_dir)
    year_range = range(args.years[0], args.years[1] + 1)
    print(f"[isd] loading observations for {len(station_meta)} stations…", flush=True)
    isd_lookup: dict[str, pd.DataFrame] = {}
    for s in station_meta:
        df = load_isd_station(s["usaf"], s["wban"], isd_dir, year_range)
        if df.empty:
            print(f"  [isd] WARN: no data for {s['usaf']}-{s['wban']}")
        isd_lookup[s["usaf"]] = df

    # 6. build the init_times list (only inits where dataset covers init + max_ar steps)
    init_times = []
    for year in year_range:
        for h in args.init_hours:
            t = datetime(year, 1, 1, h)
            end = datetime(year, 12, 31, h)
            while t <= end:
                init_times.append(t)
                t += timedelta(days=1)
    init_times.sort()
    if args.max_inits > 0:
        init_times = init_times[: args.max_inits]
    print(f"[inits] total = {len(init_times)}", flush=True)

    # 7. AR rollout + sampling
    s_mean = builder.s_mean
    s_std = builder.s_std
    rows = []
    t_start = time.time()
    n_skipped = 0
    _order_checked = False

    # Промежуточный сброс. Развёртка идёт больше двух часов, и всё это время
    # посчитанное живёт только в памяти: 27.08.2026 сбой на последней строке
    # стоил полного пересчёта. Дописываем накопленное кусками, чтобы после
    # любого обрыва можно было собрать корпус из того, что уже посчитано.
    part_path = Path(args.out_parquet).with_suffix(".partial.pkl")
    part_f, part_done = None, 0
    if args.from_partial:
        # Досбор после обрыва: развёртку не повторяем, берём посчитанное с диска
        # и идём сразу к сшивке с наблюдениями. Черновик не трогаем — вдруг
        # понадобится ещё раз.
        rows.extend(load_partial(part_path))
        init_times = []
        print(f"[part] восстановлено строк: {len(rows):,} из {part_path}", flush=True)
        if not rows:
            raise SystemExit(f"в {part_path} нет строк — досбирать нечего")
    else:
        try:
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_f = open(part_path, "wb")
        except OSError as e:
            print(f"[part] сброс недоступен ({e}), считаю без него", flush=True)

    def flush_partial():
        """Дозаписать то, что появилось с прошлого раза. Никогда не роняет счёт."""
        nonlocal part_done
        if part_f is None or len(rows) == part_done:
            return
        try:
            pickle.dump(rows[part_done:], part_f, protocol=4)
            part_f.flush()
            part_done = len(rows)
        except Exception as e:  # сброс — страховка, а не цель счёта
            print(f"[part] сброс не удался: {e}", flush=True)

    with torch.no_grad():
        for i_init, init_dt in enumerate(init_times):
            try:
                frames_norm = []
                # OBS=2: include init_dt - 6h as the first frame, then init_dt
                for k in range(OBS - 1, -1, -1):
                    fdt = init_dt - timedelta(hours=6 * k)
                    frames_norm.append(builder.build_input(fdt))  # (N, 33)
                if i_init == 0 and not _order_checked:
                    builder.check_node_order(frames_norm[-1], var_names.index("t2m"))
                    _order_checked = True
            except (IndexError, ValueError) as e:
                n_skipped += 1
                if n_skipped <= 3:
                    print(f"  [skip] {init_dt}: {e}")
                continue

            # cur_state: (1, N, OBS, C)
            stacked = np.stack(frames_norm, axis=1)  # (N, OBS, 33)
            cur_state = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
            # X_t = last input frame (used as static carry-forward source)
            X_t = cur_state[:, :, -1, :].clone()  # (1, N, 33)

            for step in range(1, max_ar + 1):
                inp = cur_state.reshape(1, cur_state.shape[1], OBS * C)
                delta = model(inp, attention_threshold=0.0)
                if delta.dim() == 2:
                    delta = delta.unsqueeze(0)
                if use_residual:
                    pred_norm = cur_state[:, :, -1, :] + delta
                else:
                    pred_norm = delta

                lead_h = step * 6
                valid_dt = init_dt + timedelta(hours=lead_h)

                if lead_h in leads:
                    pred_phys = pred_norm.squeeze(0).cpu().numpy() * s_std + s_mean  # (N, 33)
                    # static channels are carried forward verbatim from X_t (idx 7, 8);
                    # denormalise once → physical units (z_surf m^2/s^2; lsm 0..1)
                    static_phys = (
                        X_t.squeeze(0).cpu().numpy()
                        * s_std[None, :] + s_mean[None, :]
                    )  # (N, 33)
                    for s in station_meta:
                        gidx = s["grid_idx"]
                        rec = {
                            "station_usaf": s["usaf"],
                            "init_time_utc": init_dt,
                            "lead_h": lead_h,
                            "valid_time_utc": valid_dt,
                            "source": "ERA5_init",
                            "station_lat": s["lat"],
                            "station_lon": s["lon"],
                            "station_elev": s["elev"],
                            "grid_idx": gidx,
                            "node_lat": float(builder.node_lats[gidx]),
                            "node_lon": float(builder.node_lons[gidx]),
                            "dist_node_km": s["dist_km"],
                            "era5_z_surf": float(static_phys[gidx, 7]),
                            "era5_lsm": float(static_phys[gidx, 8]),
                        }
                        for v in SAMPLE_VARS:
                            rec[RENAME_FOR_PARQUET[v]] = float(
                                pred_phys[gidx, var_names.index(v)]
                            )
                        # Признаки окрестности: местный разброс поля и рельефа.
                        # Разбор по станциям показал, что поправка почти целиком
                        # чинит несоответствие площадки ячейке сетки, а сама
                        # изрезанность рельефа вокруг станции в корпусе до сих
                        # пор не была представлена ничем.
                        nb = s["nb_idx"]
                        if nb:
                            for v, nm in (("t2m", "t2m"), ("10u", "u10"), ("10v", "v10")):
                                col = pred_phys[nb, var_names.index(v)]
                                rec[f"nb_{nm}_mean"] = float(col.mean())
                                rec[f"nb_{nm}_std"] = float(col.std())
                                rec[f"nb_{nm}_dev"] = float(
                                    pred_phys[gidx, var_names.index(v)] - col.mean())
                            zs = static_phys[nb, 7] / 9.80665
                            rec["nb_z_surf_std"] = float(zs.std())
                            rec["nb_z_surf_range"] = float(zs.max() - zs.min())
                        # add z@1000 + q@1000 + t@1000 for derived features later
                        for extra in ("z@1000", "t@1000", "q@1000"):
                            rec[f"gnn_{extra.replace('@', '').lower()}"] = float(
                                pred_phys[gidx, var_names.index(extra)]
                            )
                        rows.append(rec)

                # prepare next input: replace static + forcing channels in pred
                if step < max_ar:
                    forc_norm = builder.forcing_normalized(valid_dt)  # (4,)
                    pred_norm[:, :, list(STATIC_CHANNELS)] = X_t[:, :, list(STATIC_CHANNELS)]
                    for kk, ch in enumerate(FORCING_CHANNELS):
                        pred_norm[:, :, ch] = float(forc_norm[kk])
                    # roll cur_state: drop oldest frame, append pred
                    if OBS == 1:
                        cur_state = pred_norm.unsqueeze(2)
                    else:
                        cur_state = torch.cat(
                            [cur_state[:, :, 1:, :], pred_norm.unsqueeze(2)], dim=2
                        )

            if (i_init + 1) % 500 == 0:
                flush_partial()
                print(f"[part] сброшено строк: {part_done:,}", flush=True)

            if (i_init + 1) % args.log_every == 0 or i_init == 0:
                elapsed = time.time() - t_start
                rate = (i_init + 1) / max(elapsed, 1e-3)
                eta = (len(init_times) - i_init - 1) / rate
                print(
                    f"  [{i_init+1}/{len(init_times)}] "
                    f"init={init_dt:%Y-%m-%d %H:%MZ}  rows={len(rows)}  "
                    f"rate={rate:.2f}/s  eta={eta/60:.1f}m",
                    flush=True,
                )

    flush_partial()
    if part_f is not None:
        part_f.close()
    print(f"[inference] done. rows={len(rows)}  skipped_inits={n_skipped}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows produced — nothing to write.")

    # 8. join with ISD obs
    obs_parts = []
    for s in station_meta:
        d = isd_lookup[s["usaf"]].copy()
        if d.empty:
            continue
        d["station_usaf"] = s["usaf"]
        obs_parts.append(d)
    if not obs_parts:
        raise SystemExit("No ISD observations loaded — cannot join.")
    obs = pd.concat(obs_parts, ignore_index=True)
    obs = obs.rename(columns={"valid_time": "valid_time_utc"})

    df = df.merge(obs, on=["station_usaf", "valid_time_utc"], how="left")
    df = df.dropna(subset=["obs_t2m_C"]).reset_index(drop=True)
    print(f"[join] rows after ISD join (with obs): {len(df)}", flush=True)

    # 9. derived/static/temporal features
    df["obs_t2m_K"] = df["obs_t2m_C"] + 273.15
    # Штиль: скорость 0, направления нет и в ISD-Lite оно помечено пропуском.
    # Прямая формула давала NaN, и штили целиком выпадали из оценки ветра —
    # а это как раз случаи с наибольшей относительной ошибкой модели.
    df["obs_u10"], df["obs_v10"] = wind_components(df["obs_ws"], df["obs_wd"])
    df["lapse_t850_1000"] = df["gnn_t850"] - df["gnn_t1000"]
    # gnn_sp is in hPa (see corpus_v1 stats: min 887, max 1063); formula needs Pa
    df["dewpoint_depression"] = [
        dewpoint_depression_K(t, q, p * 100.0)
        for t, q, p in zip(df["gnn_t2m"], df["gnn_q1000"], df["gnn_sp"])
    ]
    df["solar_zen"] = [
        90.0 - solar_elevation(la, lo, dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt)
        for la, lo, dt in zip(df["station_lat"], df["station_lon"], df["valid_time_utc"])
    ]
    # temporal
    h = df["valid_time_utc"].dt.hour + df["valid_time_utc"].dt.minute / 60.0
    doy = df["valid_time_utc"].dt.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * h / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * h / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["lead_norm"] = df["lead_h"] / 120.0
    # static placeholders (no urban/coast db on v3 yet); lsm + z_surf come from
    # static channels captured per-row during AR rollout (real ERA5 values).
    df["urban_flag"] = 0.0
    df["dist_to_coast"] = 0.0
    # z_surf in metres (channel was geopotential m^2/s^2 → /9.80665).
    df["z_surf"] = df["era5_z_surf"] / 9.80665
    df["lsm"] = df["era5_lsm"].clip(0.0, 1.0)
    # partition keys
    df["year"] = df["init_time_utc"].dt.year
    df["init_hour"] = df["init_time_utc"].dt.hour

    # 10. write Parquet
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = save_table(df, out_path)
    part_path.unlink(missing_ok=True)  # корпус на диске, черновик больше не нужен
    print(f"[done] wrote {out_path} — {len(df)} rows  "
          f"({df['station_usaf'].nunique()} stations)", flush=True)


if __name__ == "__main__":
    main()
