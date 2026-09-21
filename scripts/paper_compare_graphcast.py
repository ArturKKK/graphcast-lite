#!/usr/bin/env python3
"""Сравнение с GraphCast по одинаковым узлам.

Требование рецензента: наши числа и числа GraphCast должны относиться к одним и
тем же точкам. Публикуемые метрики GraphCast посчитаны по всему шару, а у нас —
по вставке над Красноярским краем; напрямую они несопоставимы уже потому, что
60 % шара занимает океан, где прогнозировать легче.

Здесь этого нет: прогнозы GraphCast, реанализ ERA5 и климатология взяты из
WeatherBench 2 в разрешении 0,25° и обрезаны по нашему окну 41 × 61
(scripts/wb2_subset.py). Метрики считаются по тем же 2501 узлам и с тем же
широтным весом, что в статье.

Проверка на самих себе. Инерционный прогноз пересчитывается из скачанных
данных и сверяется с числами статьи (4,99 / 6,88 / 5,88 / 4,42 °C по приземной
температуре). Если сходится, значит узлы, сроки и единицы совпали; расхождение
означает ошибку сопоставления, а не разницу моделей.

  python3 scripts/paper_compare_graphcast.py \\
      --gc /data/wb2/gc_krsk.npz --era5 /data/wb2/era5_krsk.npz \\
      --clim /data/wb2/clim_krsk_wb2.npz \\
      --ours docs/paper/runs/acc_lat_clim/w_m33_chw_roi_samples.npz
"""
import argparse
from pathlib import Path

import numpy as np

# Соответствие имён: наше -> WeatherBench 2.
VARS = {
    "t2m": ("2m_temperature", "°C"),
    "10u": ("10m_u_component_of_wind", "м/с"),
    "10v": ("10m_v_component_of_wind", "м/с"),
    "msl": ("mean_sea_level_pressure", "гПа"),
}
SCALE = {"msl": 0.01}          # Па -> гПа
LEADS = [6, 12, 18, 24]
DATASET_START = np.datetime64("2010-01-01T00")
OBS_WINDOW = 2                 # два входных среза


def lat_weights(lat):
    """cos(широта), нормированные на единичное среднее — как в predict.py."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.mean()


def wrmse(a, b, wlat):
    """Широтно взвешенная ошибка по осям (…, широта, долгота)."""
    d2 = (a - b) ** 2
    w = wlat[:, None]
    return float(np.sqrt((d2 * w).sum() / (np.ones_like(d2) * w).sum()))


def wacc(f, t, c, wlat):
    """Аномальная корреляция по WeatherBench 2: отношение сумм, не среднее."""
    fa, aa = f - c, t - c
    w = wlat[:, None]
    num = (w * fa * aa).sum()
    den = np.sqrt((w * fa ** 2).sum() * (w * aa ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--clim", default=None)
    ap.add_argument("--ours", default=None)
    a = ap.parse_args()

    gc = np.load(a.gc, allow_pickle=True)
    e5 = np.load(a.era5, allow_pickle=True)
    cl = np.load(a.clim, allow_pickle=True) if a.clim else None

    # Сетки обоих наборов должны совпадать узел в узел, иначе сравнение
    # превращается в сравнение интерполяций.
    if not np.allclose(np.sort(gc["lat"]), np.sort(e5["lat"])):
        raise SystemExit("широты GraphCast и ERA5 не совпали")
    if not np.allclose(gc["lon"], e5["lon"]):
        raise SystemExit("долготы не совпали")

    gl, el = gc["lat"], e5["lat"]
    flip = not np.allclose(gl, el)          # порядок широт может быть обратным
    wlat = lat_weights(el)

    t_gc = gc["time"].astype("datetime64[h]")
    leads = list(gc["lead_h"])
    t_e5 = e5["time"].astype("datetime64[h]")
    e5_pos = {t: i for i, t in enumerate(t_e5)}

    print(f"GraphCast: {len(t_gc)} инициализаций, горизонты {leads} ч")
    print(f"ERA5:      {len(t_e5)} сроков, {t_e5[0]} … {t_e5[-1]}")

    # --- сроки, где есть и прогноз, и истина на всех горизонтах ---
    ok = []
    for i, t0 in enumerate(t_gc):
        if all((t0 + np.timedelta64(L, "h")) in e5_pos for L in leads) and t0 in e5_pos:
            ok.append(i)
    print(f"пригодных инициализаций: {len(ok)}\n")
    if not ok:
        raise SystemExit("пересечения по времени нет — проверьте окна")

    def field(src, name, idx, lead_i=None):
        x = src[name][idx] if lead_i is None else src[name][idx, lead_i]
        if flip and src is gc:
            x = x[::-1]
        return x.astype(np.float64)

    def clim_at(name, valid):
        """Климатология на конкретный срок: таблица час × день года."""
        h = int(valid.astype("datetime64[h]").astype(int) % 24)
        hi = int(np.where(cl["hour"] == h)[0][0])
        doy = int((valid.astype("datetime64[D]") -
                   valid.astype("datetime64[Y]")).astype(int)) + 1
        di = int(np.where(cl["dayofyear"] == doy)[0][0])
        return cl[name][hi, di].astype(np.float64)

    rows = []
    for our, (wb, unit) in VARS.items():
        if wb not in gc or wb not in e5:
            print(f"  нет {wb} — пропускаю"); continue
        k = SCALE.get(our, 1.0)
        for li, L in enumerate(leads):
            P, T, C, B = [], [], [], []
            for i in ok:
                v = t_gc[i] + np.timedelta64(L, "h")
                P.append(field(gc, wb, i, li) * k)
                T.append(field(e5, wb, e5_pos[v]) * k)
                B.append(field(e5, wb, e5_pos[t_gc[i]]) * k)   # инерция: поле на t0
                if cl is not None and wb in cl:
                    C.append(clim_at(wb, v) * k)
            P, T, B = np.array(P), np.array(T), np.array(B)
            r_gc = wrmse(P, T, wlat)
            r_pe = wrmse(B, T, wlat)
            acc = ""
            if C:
                C = np.array(C)
                acc = f"{wacc(P, T, C, wlat):.4f}"
            rows.append((our, unit, L, r_gc, r_pe, acc))
            print(f"  {our:4s} +{L:>2} ч: GraphCast {r_gc:7.3f} {unit:4s} | "
                  f"инерция {r_pe:7.3f} | ACC {acc}")

    print("\n--- проверка сопоставления ---")
    print("Инерционный прогноз по приземной температуре, посчитанный здесь:")
    pe = [f"{r[4]:.2f}" for r in rows if r[0] == "t2m"]
    print(f"   {' / '.join(pe)} °C")
    print("В статье (наша выборка, равные веса): 4,99 / 6,88 / 5,88 / 4,42 °C")
    print("Близость подтверждает, что узлы, сроки и единицы совпали.")

    if a.ours:
        o = np.load(a.ours, allow_pickle=True)
        key = "wmse_pred_region" if "wmse_pred_region" in o else "mse_pred_region"
        names = [str(v) for v in o["variables"]]
        t0s = DATASET_START + (o["t_offset"].astype("int64") + OBS_WINDOW - 1) * np.timedelta64(6, "h")
        want = set(t_gc[ok].astype("datetime64[h]"))
        sel = np.array([i for i, t in enumerate(t0s.astype("datetime64[h]")) if t in want])
        print(f"\n--- наша модель на тех же {len(sel)} инициализациях ---")
        if len(sel) == 0:
            print("   пересечения нет: проверьте DATASET_START и окно наблюдений")
        else:
            for our in VARS:
                if our not in names:
                    continue
                ch = names.index(our); std = o["std"][ch]
                k = SCALE.get(our, 1.0)
                vals = [np.sqrt(o[key][sel, li, ch].mean()) * std * k for li in range(len(leads))]
                print(f"  {our:4s}: " + " | ".join(f"+{L} ч {v:.3f}" for L, v in zip(leads, vals)))


if __name__ == "__main__":
    main()
