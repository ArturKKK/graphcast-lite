#!/usr/bin/env python3
"""Разложение ошибки по масштабам: наша модель против GraphCast.

Зачем. GraphCast точнее нас на 33–79 % (табл. 6 статьи). От того, на каких
масштабах сидит разница, зависит, что улучшать:
  * крупный масштаб (≥ 1–2°) — динамика: грубая глобальная сетка 0,703°,
    четыре уровня, мало параметров процессора;
  * мелкий масштаб (< 1°) — перенос в узлы 0,25°: кодировщик и декодировщик
    у нас GCNConv без признаков рёбер, и детали вставки могут теряться.

Как. Поле ошибки (прогноз − ERA5) на решётке вставки 41 × 61 делим фильтром
Гаусса на гладкую часть и остаток. Фильтр нормированный (свёртка маски),
чтобы у краёв окна не было провала. Сумма квадратов частей почти равна
квадрату целого: части почти ортогональны, перекрёстный член печатается.
Дополнительно: среднее по области смещение на каждом сроке (самый крупный
масштаб) и систематическая часть (среднее по срокам в каждом узле) — её
убирает постобработка, и её надо отличать от ошибок модели.

Все числа с широтным весом, на общих инициализациях 00/12 UTC.

  python3 scripts/paper_error_scales.py \\
      --ours docs/paper/runs/acc_lat_clim/d_chw_diag_errors.npz \\
      --gc /data/wb2/gc_krsk.npz --era5 /data/wb2/era5_krsk.npz
"""
import argparse

import numpy as np

VARS = {"t2m": ("2m_temperature", 1.0, "°C"),
        "10u": ("10m_u_component_of_wind", 1.0, "м/с"),
        "10v": ("10m_v_component_of_wind", 1.0, "м/с"),
        "msl": ("mean_sea_level_pressure", 0.01, "гПа")}
STEP = 0.25


def load(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def gauss_smooth(f, sigma_pts):
    """Нормированный гауссов фильтр по двум последним осям.

    Нормировка на свёртку единичной маски: у края окна веса не теряются,
    и гладкая часть не проседает к нулю.
    """
    if sigma_pts <= 0:
        return f.copy()
    r = int(np.ceil(3 * sigma_pts))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)

    def conv(a, axis):
        a = np.moveaxis(a, axis, -1)
        pad = np.zeros(a.shape[:-1] + (a.shape[-1] + 2 * r,), a.dtype)
        pad[..., r:-r] = a
        out = sum(k[i] * pad[..., i:i + a.shape[-1]] for i in range(len(k)))
        return np.moveaxis(out, -1, axis)

    num = conv(conv(f, -1), -2)
    den = conv(conv(np.ones(f.shape[-2:]), -1), -2)
    return num / den


def wms(a, w):
    """Взвешенный средний квадрат по двум последним осям и всем прочим."""
    return float((a ** 2 * w).sum() / (np.ones_like(a) * w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gc", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.5, 1.0, 2.0],
                    help="σ фильтра в градусах")
    ap.add_argument("--save", default=None, help="npz с картами для рисунков")
    a = ap.parse_args()

    o, gc, e5 = load(a.ours), load(a.gc), load(a.era5)

    # --- решётка вставки: узлы в порядке нашего графа -> (iy, ix) ---
    lat_g, lon_g = e5["lat"], e5["lon"]           # порядок WB2
    ypos = {round(float(v), 3): i for i, v in enumerate(lat_g)}
    xpos = {round(float(v), 3): i for i, v in enumerate(lon_g)}
    iy = np.array([ypos[round(float(v), 3)] for v in o["lat"]])
    ix = np.array([xpos[round(float(v), 3)] for v in o["lon"]])
    assert len(set(zip(iy, ix))) == len(iy) == len(lat_g) * len(lon_g), "узлы не легли на решётку"
    gflip = not np.allclose(gc["lat"], lat_g)
    wlat = np.cos(np.deg2rad(lat_g))
    wlat = (wlat / wlat.mean())[:, None]

    # --- общие сроки ---
    t_o = o["t0"].astype("datetime64[h]")
    t_g = gc["time"].astype("datetime64[h]")
    t_e = e5["time"].astype("datetime64[h]")
    gpos = {t: i for i, t in enumerate(t_g)}
    epos = {t: i for i, t in enumerate(t_e)}
    leads = [int(x) for x in o["lead_h"]]
    assert leads == [int(x) for x in gc["lead_h"]], (leads, gc["lead_h"])
    keep = [i for i, t in enumerate(t_o) if t in gpos and all(
        (t + np.timedelta64(L, "h")) in epos for L in leads)]
    print(f"наших сроков {len(t_o)}, общих с GraphCast и ERA5 {len(keep)}\n")

    chans = [str(c) for c in o["channels"]]
    out = {}
    for ci, name in enumerate(chans):
        wb, k, unit = VARS[name]
        # наша ошибка на решётке: (срок, горизонт, y, x)
        E_o = np.zeros((len(keep), len(leads), len(lat_g), len(lon_g)), np.float32)
        E_o[:, :, iy, ix] = o["err"][keep, :, :, ci].astype(np.float32)
        E_g = np.zeros_like(E_o)
        T = np.zeros_like(E_o)
        for n, i in enumerate(keep):
            t0 = t_o[i]
            P = gc[wb][gpos[t0]].astype(np.float32) * k       # (lead, y, x)
            if gflip:
                P = P[:, ::-1]
            for li, L in enumerate(leads):
                tr = e5[wb][epos[t0 + np.timedelta64(L, "h")]].astype(np.float32)
                tr = tr[0] if tr.ndim == 3 else tr
                tr = tr * k
                E_g[n, li] = P[li] - tr
                T[n, li] = tr

        print(f"=== {name}, {unit} (RMSE по всем горизонтам; в скобках +6/+12/+18/+24 ч) ===")

        def row(label, fo, fg):
            per_o = [np.sqrt(wms(fo[:, l], wlat)) for l in range(len(leads))]
            per_g = [np.sqrt(wms(fg[:, l], wlat)) for l in range(len(leads))]
            ro, rg = np.sqrt(wms(fo, wlat)), np.sqrt(wms(fg, wlat))
            print(f"  {label:34s} наша {ro:6.3f}  GC {rg:6.3f}  отн. {ro / rg:5.2f}   "
                  f"({' '.join(f'{x:.2f}' for x in per_o)} | {' '.join(f'{x:.2f}' for x in per_g)})")
            return ro, rg

        res = {"full": row("полная ошибка", E_o, E_g)}

        # систематическая часть: среднее по срокам в узле (на горизонт и час)
        hrs = (t_o[keep].astype("int64") % 24)
        S_o, S_g = np.zeros_like(E_o), np.zeros_like(E_g)
        for h in np.unique(hrs):
            m = hrs == h
            S_o[m] = E_o[m].mean(0, keepdims=True)
            S_g[m] = E_g[m].mean(0, keepdims=True)
        res["sys"] = row("систематическая (среднее по срокам)", S_o, S_g)
        R_o, R_g = E_o - S_o, E_g - S_g
        res["rand"] = row("случайная (остаток)", R_o, R_g)

        # случайную часть делим по масштабам
        B_o = (R_o * wlat).sum((-2, -1), keepdims=True) / (np.ones_like(R_o[0, 0]) * wlat).sum()
        B_g = (R_g * wlat).sum((-2, -1), keepdims=True) / (np.ones_like(R_g[0, 0]) * wlat).sum()
        res["bias"] = row("  среднее по области на сроке", np.broadcast_to(B_o, R_o.shape),
                          np.broadcast_to(B_g, R_g.shape))
        for s in a.sigmas:
            sp = s / STEP
            L_o, L_g = gauss_smooth(R_o, sp), gauss_smooth(R_g, sp)
            lo = row(f"  крупнее σ={s:g}° (≈{111 * s * 2.355:.0f} км)", L_o, L_g)
            hi = row(f"  мельче  σ={s:g}°", R_o - L_o, R_g - L_g)
            cross = 2 * float(((L_o * (R_o - L_o)) * wlat).sum() / (np.ones_like(R_o) * wlat).sum())
            res[f"lo{s:g}"], res[f"hi{s:g}"] = lo, hi
            print(f"    перекрёстный член {cross / wms(R_o, wlat) * 100:+.1f} % от квадрата случайной части")

        # сколько мелкого масштаба в самой истине: теряет ли модель детали
        A = T - T.mean(0, keepdims=True)
        for s in a.sigmas[:2]:
            sp = s / STEP
            fine_t = A - gauss_smooth(A, sp)
            print(f"  дисперсия истины мельче σ={s:g}°: {np.sqrt(wms(fine_t, wlat)):.3f} {unit} "
                  f"(всего аномалия {np.sqrt(wms(A, wlat)):.3f})")
        # --- точное разложение: косинусное преобразование окна ---
        # Гауссов фильтр делит ошибку на неортогональные части (перекрёстный
        # член выше до 20 %). DCT-II с нормировкой ortho сохраняет энергию, и
        # доли полос складываются ровно в 100 %. Широтный вес вносим как
        # множитель sqrt(w) до преобразования.
        from scipy.fft import dctn
        ny, nx = len(lat_g), len(lon_g)
        Ly = ny * STEP * 111.2
        Lx = nx * STEP * 111.2 * np.cos(np.deg2rad(np.mean(lat_g)))
        ky, kx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        with np.errstate(divide="ignore"):
            lam = 2.0 / np.sqrt((ky / Ly) ** 2 + (kx / Lx) ** 2)   # км, мода (0,0) -> inf
        # Мода (0, 0) — среднее по окну, у неё λ = inf; отдельной полосой, иначе
        # полуоткрытый интервал её теряет и суммы не сходятся с полной ошибкой.
        bands = [(np.inf, np.inf, "среднее по области"), (1000, np.inf, "≥1000 км"),
                 (500, 1000, "500–1000 км"),
                 (220, 500, "220–500 км"), (100, 220, "100–220 км"), (0, 100, "<100 км")]
        sw = np.sqrt(wlat)
        spec = {}
        for tag, F in (("наша", E_o), ("GC", E_g), ("истина-аном.", T - T.mean(0, keepdims=True))):
            C = dctn(F * sw, type=2, axes=(-2, -1), norm="ortho")
            P = (C.astype(np.float64) ** 2).mean(axis=(0, 1)) / (ny * nx)
            spec[tag] = [float(P[np.isinf(lam)].sum()) if np.isinf(lo) else
                         float(P[(lam >= lo) & (lam < hi)].sum()) for lo, hi, _ in bands]
        tot_o, tot_g = sum(spec["наша"]), sum(spec["GC"])
        print(f"  спектр (DCT), RMSE² по полосам, проверка сумм: наша {np.sqrt(tot_o):.3f}, GC {np.sqrt(tot_g):.3f}")
        d = tot_o - tot_g
        for i, (_, _, lab) in enumerate(bands):
            so, sg, st = spec["наша"][i], spec["GC"][i], spec["истина-аном."][i]
            print(f"    {lab:20s} наша {np.sqrt(so):6.3f}  GC {np.sqrt(sg):6.3f}  отн. "
                  f"{np.sqrt(so / sg) if sg > 0 else float('nan'):5.2f}  "
                  f"доля разности {100 * (so - sg) / d:5.1f} %   (изменчивость истины {np.sqrt(st):.3f})")
        res["spec"] = spec
        out[name] = res
        if a.save:
            out[name + "_sysmap"] = (S_o.mean((0, 1)), S_g.mean((0, 1)))
        print()

    # --- сводка: куда уходит разница квадратов ---
    print("=== доля разности квадратов (наша² − GC²), приходящейся на часть ===")
    for name in chans:
        r = out[name]
        d = r["full"][0] ** 2 - r["full"][1] ** 2
        parts = {"систематическая": r["sys"], "случайная: среднее по области": r["bias"]}
        s1 = a.sigmas[1] if len(a.sigmas) > 1 else a.sigmas[0]
        parts[f"случайная крупнее σ={s1:g}° (без среднего)"] = (
            np.sqrt(max(r[f"lo{s1:g}"][0] ** 2 - r["bias"][0] ** 2, 0)),
            np.sqrt(max(r[f"lo{s1:g}"][1] ** 2 - r["bias"][1] ** 2, 0)))
        parts[f"случайная мельче σ={s1:g}°"] = r[f"hi{s1:g}"]
        txt = ", ".join(f"{k} {100 * (v[0] ** 2 - v[1] ** 2) / d:.0f} %" for k, v in parts.items())
        print(f"  {name}: {txt}")

    if a.save:
        np.savez_compressed(a.save, lat=lat_g, lon=lon_g, **{
            k: np.stack(v) for k, v in out.items() if k.endswith("_sysmap")})
        print(f"\nкарты систематической ошибки → {a.save}")


if __name__ == "__main__":
    main()
