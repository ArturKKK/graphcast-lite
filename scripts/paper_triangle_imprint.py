#!/usr/bin/env python3
"""Виден ли в ошибке отпечаток треугольников меша.

Гипотеза (docs/results/error_scales_2026-09-24.md). Декодировщик у нас стоит
на GCNConv по графу «3 вершины треугольника → узел сетки». Нормировка GCN
даёт всем трём вершинам один и тот же вес, от положения узла внутри
треугольника он не зависит. Тогда вклад процессора постоянен на треугольнике
и скачет на его сторонах.

Проверка. Берём пары соседних узлов вставки (по широте и по долготе, шаг
0,25°). Пару называем «внутри», если оба узла попали в один треугольник меша
уровня 6, и «через сторону», если в разные. Сравниваем средний квадрат
разности ошибок в парах. Если гипотеза верна, у нас «через сторону» заметно
больше, чем «внутри», а у GraphCast отношение около 1: его декодировщик знает
смещение узла относительно вершин. Сама истина тоже гладкая и отношение для
неё служит базой: у пар через сторону в среднем чуть больше расстояние до
центра треугольника, и это не должно давать заметного эффекта.

  python3 scripts/paper_triangle_imprint.py \\
      --ours docs/paper/runs/acc_lat_clim/d_chw_diag_errors.npz \\
      --gc /data/wb2/gc_krsk.npz --era5 /data/wb2/era5_krsk.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mesh.create_mesh import get_hierarchy_of_triangular_meshes_for_sphere  # noqa: E402
from src.mesh.grid_mesh_connectivity import in_mesh_triangle_indices  # noqa: E402

VARS = {"t2m": ("2m_temperature", 1.0), "10u": ("10m_u_component_of_wind", 1.0),
        "10v": ("10m_v_component_of_wind", 1.0), "msl": ("mean_sea_level_pressure", 0.01)}


def load(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def triangle_of(lat2d, lon2d, level):
    """Номер треугольника меша для каждого узла решётки (форма как у lat2d)."""
    mesh = get_hierarchy_of_triangular_meshes_for_sphere(splits=level)[-1]
    gi, mi = in_mesh_triangle_indices(grid_latitude=lat2d.ravel(),
                                      grid_longitude=lon2d.ravel(),
                                      mesh=mesh, flat=True)
    # на каждый узел ровно три вершины; треугольник = отсортированная тройка
    order = np.argsort(gi, kind="stable")
    tri = mi[order].reshape(-1, 3)
    tri.sort(axis=1)
    _, tid = np.unique(tri, axis=0, return_inverse=True)
    return tid.reshape(lat2d.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--gc", required=True)
    ap.add_argument("--era5", required=True)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--lead", type=int, default=3, help="индекс горизонта, 3 = +24 ч")
    a = ap.parse_args()

    o, gc, e5 = load(a.ours), load(a.gc), load(a.era5)
    lat, lon = e5["lat"], e5["lon"]
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    tid = triangle_of(LAT, LON, a.level)
    print(f"треугольников меша уровня {a.level} в окне: {len(np.unique(tid))}, "
          f"узлов на треугольник в среднем {tid.size / len(np.unique(tid)):.1f}")

    yp = {round(float(v), 3): i for i, v in enumerate(lat)}
    xp = {round(float(v), 3): i for i, v in enumerate(lon)}
    iy = np.array([yp[round(float(v), 3)] for v in o["lat"]])
    ix = np.array([xp[round(float(v), 3)] for v in o["lon"]])
    flip = not np.allclose(gc["lat"], lat)

    # пары соседей по двум осям и признак «через сторону»
    pairs = []
    for dy, dx in ((1, 0), (0, 1)):
        A = (slice(0, LAT.shape[0] - dy), slice(0, LAT.shape[1] - dx))
        B = (slice(dy, None), slice(dx, None))
        pairs.append((A, B, tid[A] != tid[B]))
    n_cross = sum(int(c.sum()) for *_, c in pairs)
    n_all = sum(c.size for *_, c in pairs)
    print(f"пар соседей {n_all}, через сторону треугольника {n_cross}\n")

    t0 = o["t0"].astype("datetime64[h]")
    te = e5["time"].astype("datetime64[h]"); ep = {t: i for i, t in enumerate(te)}
    tg = gc["time"].astype("datetime64[h]"); gp = {t: i for i, t in enumerate(tg)}
    L = int(o["lead_h"][a.lead])
    print(f"горизонт +{L} ч; средний квадрат разности соседей, отношение «через сторону / внутри»")
    for ci, name in enumerate(str(c) for c in o["channels"]):
        wb, k = VARS[name]
        acc = {"наша ошибка": [0.0, 0.0, 0, 0], "ошибка GC": [0.0, 0.0, 0, 0],
               "истина": [0.0, 0.0, 0, 0]}
        for i, t in enumerate(t0):
            v = t + np.timedelta64(L, "h")
            if t not in gp or v not in ep:
                continue
            tr = e5[wb][ep[v]][0].astype(np.float64) * k
            eo = np.zeros_like(tr); eo[iy, ix] = o["err"][i, a.lead, :, ci]
            g = gc[wb][gp[t], a.lead].astype(np.float64) * k
            g = g[::-1] if flip else g
            for key, F in (("наша ошибка", eo), ("ошибка GC", g - tr), ("истина", tr)):
                for A, B, cross in pairs:
                    d2 = (F[A] - F[B]) ** 2
                    acc[key][0] += d2[cross].sum(); acc[key][2] += int(cross.sum())
                    acc[key][1] += d2[~cross].sum(); acc[key][3] += int((~cross).sum())
        line = []
        for key, (sc, si, nc, ni) in acc.items():
            line.append(f"{key} {np.sqrt(sc / nc):.3f}/{np.sqrt(si / ni):.3f} = "
                        f"{(sc / nc) / (si / ni):.2f}")
        print(f"  {name}: " + " | ".join(line))


if __name__ == "__main__":
    main()
