#!/usr/bin/env python3
"""Вырезка окна Красноярского края из наборов WeatherBench 2 в облаке.

Зачем. Рецензенты просят сравнивать по одинаковым узлам: у GraphCast сетка
глобальная, у нас — вставка 61 × 41. Наборы WB2 лежат в zarr на публичном
бакете в разрешении 0,25°, то есть ровно в нашем, и нужные узлы выбираются без
интерполяции.

Почему трафик большой. Чанк в этих наборах — поле на весь шар. Наше окно
занимает 0,25 % площади, но меньше одного чанка не скачать. На диск при этом
ложится только окно: мегабайты против десятков гигабайт трафика.

Про выборку по границам чанков. У климатологии чанк — это 3 часа × 3 дня
(20 МБ). Запрашивать каждый час и день по отдельности значит тянуть один и тот
же чанк девять раз; первая версия скрипта так и делала, и вместо 4,9 ГБ на
переменную получалось 29. Поэтому индексы группируются по чанкам и каждый
качается ровно один раз.

Уровни. В архиве прогнозов все 37 уровней лежат одним чанком (около 80 МБ на
срок и горизонт), так что выбрать из него один уровень стоит как выбрать все:
на наш период это сотни гигабайт. Поэтому для прогнозов берём приземные поля,
а уровни — из климатологии, где они разложены по отдельным чанкам.

  python3 scripts/wb2_subset.py --preset clim      --out /data/wb2/clim_krsk.npz
  python3 scripts/wb2_subset.py --preset graphcast --out /data/wb2/gc_krsk.npz
  python3 scripts/wb2_subset.py --preset hres      --out /data/wb2/hres_krsk.npz
"""
import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LAT = (50.0, 60.0)      # наша вставка
LON = (83.0, 98.0)
BASE = "https://storage.googleapis.com/weatherbench2/datasets"

# По умолчанию берём то, что статья отчитывает поканально.
PRESETS = {
    "clim": dict(
        url=f"{BASE}/era5-hourly-climatology/1990-2019_6h_1440x721.zarr",
        surface=["2m_temperature", "10m_u_component_of_wind",
                 "10m_v_component_of_wind", "mean_sea_level_pressure"],
        plev={"temperature": [850], "geopotential": [500]},
        latname="latitude", lonname="longitude", outer=("hour", "dayofyear"),
    ),
    "graphcast": dict(
        url=f"{BASE}/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr",
        surface=["2m_temperature", "10m_u_component_of_wind",
                 "10m_v_component_of_wind", "mean_sea_level_pressure"],
        plev={}, latname="lat", lonname="lon",
        outer=("time", "prediction_timedelta"),
    ),
    "hres": dict(
        url=f"{BASE}/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr",
        surface=["2m_temperature", "10m_u_component_of_wind",
                 "10m_v_component_of_wind", "mean_sea_level_pressure"],
        plev={}, latname="latitude", lonname="longitude",
        outer=("time", "prediction_timedelta"),
    ),
}


def open_group(url):
    import fsspec
    import zarr
    return zarr.open_group(fsspec.get_mapper(url), mode="r")


def window(g, latname, lonname):
    lat = np.asarray(g[latname], dtype=np.float64)
    lon = np.asarray(g[lonname], dtype=np.float64)
    iy = np.where((lat >= LAT[0] - 1e-6) & (lat <= LAT[1] + 1e-6))[0]
    ix = np.where((lon >= LON[0] - 1e-6) & (lon <= LON[1] + 1e-6))[0]
    if len(iy) != 41 or len(ix) != 61:
        raise SystemExit(f"окно вышло {len(iy)}×{len(ix)}, ожидалось 41×61 — "
                         "сетка набора не 0,25°?")
    return iy, ix, lat[iy], lon[ix]


def chunk_groups(wanted, chunk):
    """Индексы, сгруппированные по номеру чанка: {чанк: [индексы]}."""
    by = defaultdict(list)
    for i in wanted:
        by[i // chunk].append(i)
    return by


def fetch(arr, want_a, want_b, iy, ix, workers, label, level_idx=None):
    """Тянет массив по границам чанков и режет окно.

    Возвращает (len(want_a), len(want_b), 41, 61). Каждый чанк скачивается ровно
    один раз: индексы сгруппированы, внутри группы берётся непрерывный срез.
    """
    ca, cb = arr.chunks[0], arr.chunks[1]
    ga, gb = chunk_groups(want_a, ca), chunk_groups(want_b, cb)
    pos_a = {v: k for k, v in enumerate(want_a)}
    pos_b = {v: k for k, v in enumerate(want_b)}
    out = np.empty((len(want_a), len(want_b), len(iy), len(ix)), dtype=np.float32)

    tasks = [(A, B) for A in ga for B in gb]
    done, t0 = [0], time.time()
    sy = slice(int(iy[0]), int(iy[-1]) + 1)
    sx = slice(int(ix[0]), int(ix[-1]) + 1)

    def one(t):
        A, B = t
        ia, ib = ga[A], gb[B]
        sa = slice(ia[0], ia[-1] + 1)
        sb = slice(ib[0], ib[-1] + 1)
        sel = (sa, sb) if level_idx is None else (sa, sb, level_idx)
        block = np.asarray(arr[sel + (sy, sx)], dtype=np.float32)
        for u, i in enumerate(range(sa.start, sa.stop)):
            if i not in pos_a:
                continue
            for v, j in enumerate(range(sb.start, sb.stop)):
                if j not in pos_b:
                    continue
                out[pos_a[i], pos_b[j]] = block[u, v]
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(tasks):
            el = time.time() - t0
            left = (len(tasks) - done[0]) / max(done[0] / max(el, 1e-6), 1e-9)
            print(f"    {label}: {done[0]}/{len(tasks)} чанков, "
                  f"осталось ~{left/60:.1f} мин", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, tasks))
    return out, len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-lead-h", type=int, default=24)
    ap.add_argument("--time-stride", type=int, default=1,
                    help="каждый N-й срок инициализации: трафик делится на N")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="посчитать трафик и выйти, ничего не качая")
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    print(f"[wb2] {cfg['url']}")
    g = open_group(cfg["url"])
    iy, ix, lats, lons = window(g, cfg["latname"], cfg["lonname"])
    print(f"[wb2] окно {len(iy)}×{len(ix)}: широты {lats[0]}…{lats[-1]}, "
          f"долготы {lons[0]}…{lons[-1]}")

    res = {"lat": lats, "lon": lons, "source": cfg["url"]}
    na, nb = cfg["outer"]

    if a.preset == "clim":
        hour, doy = np.asarray(g["hour"]), np.asarray(g["dayofyear"])
        want_a, want_b = list(range(len(hour))), list(range(len(doy)))
        res["hour"], res["dayofyear"] = hour, doy
    else:
        t, td = np.asarray(g[na]), np.asarray(g[nb])
        td_h = ((td / np.timedelta64(1, "h")).astype(int)
                if td.dtype.kind == "m" else td.astype(int))
        want_b = [int(i) for i in np.where(td_h <= a.max_lead_h)[0]]
        want_a = list(range(0, len(t), a.time_stride))
        res["time"], res["lead_h"] = t[want_a], td_h[want_b]
        print(f"[wb2] сроков {len(want_a)} (шаг {a.time_stride}), "
              f"горизонты {td_h[want_b]} ч")

    names = [n for n in cfg["surface"] if not a.only or n in a.only]
    jobs = [(n, None, n) for n in names]
    for name, levels in cfg["plev"].items():
        if a.only and name not in a.only:
            continue
        lev = np.asarray(g["level"])
        for L in levels:
            jobs.append((name, int(np.where(lev == L)[0][0]), f"{name}@{L}"))

    # Прикидка трафика: чанков на переменную × вес чанка.
    a0 = g[names[0]]
    per = len(chunk_groups(want_a, a0.chunks[0])) * len(chunk_groups(want_b, a0.chunks[1]))
    mb = 20 if a.preset == "clim" else 2.3
    print(f"[wb2] переменных {len(jobs)}, чанков на каждую {per}, "
          f"трафик около {len(jobs) * per * mb / 1024:.1f} ГБ")
    if a.dry_run:
        return

    total_chunks = 0
    for name, li, key in jobs:
        if name not in g:
            print(f"  нет переменной {name} — пропускаю"); continue
        print(f"  {key}")
        res[key], n = fetch(g[name], want_a, want_b, iy, ix, a.workers, key, li)
        total_chunks += n

    np.savez_compressed(a.out, **res)
    print(f"[wb2] чанков скачано {total_chunks}; "
          f"сохранено → {a.out} ({os.path.getsize(a.out) // 1024 // 1024} МБ)")


if __name__ == "__main__":
    main()
