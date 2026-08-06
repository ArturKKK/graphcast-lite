#!/usr/bin/env python3
"""Восстанавливает coords.npz для global_512x256_extra_*.

Зачем. Сборщик 33f-датасета берёт оси широты и долготы глобальной сетки из
coords.npz каталога global_extra, но кладёт их туда setup_vm, копируя из
основного глобального датасета wb2_512x256_19f_ar. Если основной датасет удалён
ради места, сборка падает с FileNotFoundError на coords.npz.

Как восстанавливаем. Глобальные узлы merge-сетки — это та же решётка 512x256 с
вырезом под область интереса. Вырез узкий (15 градусов по долготе, 10 по
широте), поэтому ни одна широтная и ни одна долготная линия не исчезает
целиком: у каждой остаются узлы вне выреза. Значит множество уникальных
значений широты и долготы среди глобальных узлов и есть искомые оси.

Результат сверяется с n_lon и n_lat из dataset_info и проверяется на
равномерность шага — если что-то не сходится, скрипт не пишет ничего.

Запуск (пути по умолчанию под VM):
    python3 scripts/fix_global_extra_coords.py
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-dir", default="/data/datasets/multires_krsk_19f_merge")
    ap.add_argument("--extra-dir", default="/data/datasets/global_512x256_extra_2010-2021_07deg")
    a = ap.parse_args()

    merge, extra = Path(a.merge_dir), Path(a.extra_dir)
    dst = extra / "coords.npz"
    if dst.exists():
        z = np.load(dst)
        print(f"coords.npz уже есть: lat {len(z['latitude'])}, lon {len(z['longitude'])} — ничего не делаю")
        return

    info = json.loads((extra / "dataset_info.json").read_text())
    n_lon, n_lat = int(info["n_lon"]), int(info["n_lat"])
    print(f"Ожидается сетка {n_lon} x {n_lat}")

    co = np.load(merge / "coords.npz")
    lat_all = co["latitude"].astype(np.float64)
    lon_all = co["longitude"].astype(np.float64)
    is_reg = co["is_regional"].astype(bool)
    lat_g, lon_g = lat_all[~is_reg], lon_all[~is_reg]
    print(f"Глобальных узлов в merge: {len(lat_g)}")

    lats = np.unique(np.round(lat_g, 6))
    lons = np.unique(np.round(lon_g, 6))
    print(f"Уникальных значений: широт {len(lats)}, долгот {len(lons)}")

    ok = True
    if len(lats) != n_lat:
        print(f"  [!] широт {len(lats)}, а ожидалось {n_lat}"); ok = False
    if len(lons) != n_lon:
        print(f"  [!] долгот {len(lons)}, а ожидалось {n_lon}"); ok = False
    for name, ax in (("широта", lats), ("долгота", lons)):
        d = np.diff(ax)
        if d.size and (d.max() - d.min()) > 1e-4:
            print(f"  [!] шаг по {name} неравномерен: от {d.min():.6f} до {d.max():.6f}"); ok = False
        elif d.size:
            print(f"  {name}: от {ax[0]:.4f} до {ax[-1]:.4f}, шаг {d.mean():.6f}")

    if not ok:
        print("\nПроверки не пройдены — файл НЕ записан. "
              "Восстанавливать оси придётся из исходного датасета.")
        raise SystemExit(1)

    np.savez(dst, latitude=lats.astype(np.float32), longitude=lons.astype(np.float32))
    print(f"\n→ записан {dst}")
    print("   Порядок обеих осей по возрастанию — как в исходном датасете "
          "(в логе интерполяции было lat=[-89.65..89.65], lon=[0.00..359.30]).")


if __name__ == "__main__":
    main()
