#!/usr/bin/env python3
"""Рисует ли модель в поле ложную мелкую рябь: проверка сглаживанием прогноза.

Зачем (29.09.2026). В разложении по масштабам (paper_error_scales.py) ошибка
давления мельче 100 км у нас 0,24 гПа, а собственная изменчивость реанализа
на этих масштабах 0,15 гПа: ошибка больше самого сигнала. Если модель на этих
масштабах рисует шум, сглаживание прогноза уменьшит ошибку. Если там
настоящий сигнал, который модель ловит не хуже GraphCast, ошибка вырастет.

Сглаживается прогноз (ошибка + истина), истина не трогается. Тот же фильтр
для сравнения применяется к GraphCast: у модели без ряби выигрыша быть не
должно. Числа на тех же 802 сроках тестового окна, что табл. 6; выбирать σ
для модели по ним нельзя, это диагностика.

  python3 scripts/paper_smoothing_check.py \\
      --ours docs/paper/runs/acc_lat_clim/t_mesh_ref_roi_errors.npz \\
      --gc /data/wb2/gc_krsk.npz --era5 /data/wb2/era5_krsk.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_error_scales import STEP, VARS, load, wms  # noqa: E402
from paper_error_scales import gauss_smooth as gauss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gc", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.125, 0.25, 0.375, 0.5],
                    help="σ фильтра в градусах")
    a = ap.parse_args()

    o, gc, e5 = load(a.ours), load(a.gc), load(a.era5)
    lat_g, lon_g = e5["lat"], e5["lon"]
    ypos = {round(float(v), 3): i for i, v in enumerate(lat_g)}
    xpos = {round(float(v), 3): i for i, v in enumerate(lon_g)}
    iy = np.array([ypos[round(float(v), 3)] for v in o["lat"]])
    ix = np.array([xpos[round(float(v), 3)] for v in o["lon"]])
    gflip = not np.allclose(gc["lat"], lat_g)
    wlat = np.cos(np.deg2rad(lat_g))
    wlat = (wlat / wlat.mean())[:, None]

    t_o = o["t0"].astype("datetime64[h]")
    gpos = {t: i for i, t in enumerate(gc["time"].astype("datetime64[h]"))}
    epos = {t: i for i, t in enumerate(e5["time"].astype("datetime64[h]"))}
    leads = [int(x) for x in o["lead_h"]]
    keep = [i for i, t in enumerate(t_o) if t in gpos and all(
        (t + np.timedelta64(L, "h")) in epos for L in leads)]
    print(f"общих сроков {len(keep)}; σ в градусах, шаг сетки {STEP}°\n")

    for ci, name in enumerate(str(c) for c in o["channels"]):
        wb, k, unit = VARS[name]
        shape = (len(keep), len(leads), len(lat_g), len(lon_g))
        E_o, E_g, T = (np.zeros(shape, np.float32) for _ in range(3))
        E_o[:, :, iy, ix] = o["err"][keep, :, :, ci].astype(np.float32)
        for n, i in enumerate(keep):
            P = gc[wb][gpos[t_o[i]]].astype(np.float32) * k
            if gflip:
                P = P[:, ::-1]
            for li, L in enumerate(leads):
                tr = e5[wb][epos[t_o[i] + np.timedelta64(L, "h")]].astype(np.float32)
                tr = (tr[0] if tr.ndim == 3 else tr) * k
                E_g[n, li], T[n, li] = P[li] - tr, tr

        r_o, r_g = np.sqrt(wms(E_o, wlat)), np.sqrt(wms(E_g, wlat))
        print(f"=== {name}, {unit}: без сглаживания наша {r_o:.3f}, GraphCast {r_g:.3f}")
        for s in a.sigmas:
            so = np.sqrt(wms(gauss(E_o + T, s / STEP) - T, wlat))
            sg = np.sqrt(wms(gauss(E_g + T, s / STEP) - T, wlat))
            print(f"  σ = {s:5.3f}°: наша {so:.3f} ({(so / r_o - 1) * 100:+5.1f} %), "
                  f"GraphCast {sg:.3f} ({(sg / r_g - 1) * 100:+5.1f} %)")
        print()


if __name__ == "__main__":
    main()
