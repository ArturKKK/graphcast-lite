#!/usr/bin/env python3
"""Перекладка климатологии WeatherBench 2 в порядок узлов нашего графа.

Зачем. Наш ACC считался против собственной климатологии по девяти годам, а у
GraphCast и прочих публикуемых моделей — против климатологии WB2 за 1990–2019
со скользящим окном. Пока климатологии разные, числа рядом ставить нельзя.
Этот скрипт переводит таблицу WB2 (час × день года × широта × долгота) в
порядок наших региональных узлов и в наши единицы.

Единицы. Наш набор хранит давление в гПа, а геопотенциал — высотой в гпм, то
есть уже поделённым на g. WB2 отдаёт паскали и м²/с². Расхождение вдвое или в
сто раз здесь не бросалось бы в глаза: ACC инвариантен к сдвигу, но не к
масштабу, и неверный множитель тихо портит только его.

Нормировку НЕ применяем: predict.py стандартизует климатологию сам по
scalers.npz того набора, на котором идёт счёт. Так файл не привязан к
конкретной сборке данных.

  python3 scripts/wb2_clim_to_nodes.py \\
      --wb2 /data/wb2/clim_krsk_wb2.npz \\
      --coords live_runtime_bundle/coords.npz \\
      --out docs/paper/runs/clim_wb2_nodes.npz
"""
import argparse
import os

import numpy as np

G = 9.80665

# Наше имя -> (имя в WB2, множитель к нашим единицам)
MAP = {
    "t2m": ("2m_temperature", 1.0),
    "10u": ("10m_u_component_of_wind", 1.0),
    "10v": ("10m_v_component_of_wind", 1.0),
    "msl": ("mean_sea_level_pressure", 0.01),        # Па -> гПа
    "sp": ("surface_pressure", 0.01),
    "tcwv": ("total_column_water_vapour", 1.0),
    "tp": ("total_precipitation_6hr", 1.0),
    "t@850": ("temperature@850", 1.0),
    "t@500": ("temperature@500", 1.0),
    "t@250": ("temperature@250", 1.0),
    "t@1000": ("temperature@1000", 1.0),
    "z@850": ("geopotential@850", 1.0 / G),          # м²/с² -> гпм
    "z@500": ("geopotential@500", 1.0 / G),
    "z@250": ("geopotential@250", 1.0 / G),
    "z@1000": ("geopotential@1000", 1.0 / G),
}
for lev in (850, 500, 250, 1000):
    MAP[f"u@{lev}"] = (f"u_component_of_wind@{lev}", 1.0)
    MAP[f"v@{lev}"] = (f"v_component_of_wind@{lev}", 1.0)
    MAP[f"q@{lev}"] = (f"specific_humidity@{lev}", 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb2", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with np.load(a.wb2, allow_pickle=True) as z:
        wb = {k: z[k] for k in z.files}
    co = np.load(a.coords)
    lat_n, lon_n = co["latitude"], co["longitude"]
    reg = co["is_regional"].astype(bool)
    node_idx = np.where(reg)[0]
    print(f"узлов в графе {len(lat_n)}, региональных {len(node_idx)}")

    wlat, wlon = wb["lat"], wb["lon"]
    # Позиция каждого нашего узла в окне WB2. Сопоставляем по координате, а не
    # по порядку: порядок широт у наборов разный (у WB2 сверху вниз).
    ypos = {round(float(v), 4): i for i, v in enumerate(wlat)}
    xpos = {round(float(v), 4): i for i, v in enumerate(wlon)}
    iy = np.array([ypos[round(float(v), 4)] for v in lat_n[node_idx]])
    ix = np.array([xpos[round(float(v), 4)] for v in lon_n[node_idx]])
    print(f"окно WB2 {len(wlat)}×{len(wlon)}, все узлы сопоставлены")

    have = [c for c in MAP if MAP[c][0] in wb]
    missing = [c for c in MAP if MAP[c][0] not in wb]
    print(f"каналов с климатологией: {len(have)} — {have}")
    if missing:
        print(f"нет в выгрузке (ACC по ним считаться не будет): {missing}")

    hour, doy = wb["hour"], wb["dayofyear"]
    out = np.empty((len(have), len(hour), len(doy), len(node_idx)), dtype=np.float32)
    for k, name in enumerate(have):
        src, mul = MAP[name]
        out[k] = wb[src][:, :, iy, ix] * mul

    np.savez_compressed(a.out, clim=out, channels=np.array(have),
                        hour=hour, dayofyear=doy, node_index=node_idx,
                        source=str(wb.get("source", "")))
    mb = os.path.getsize(a.out) // 1024 // 1024
    print(f"сохранено → {a.out} ({mb} МБ), форма {out.shape}")
    print("проверка t2m по краю: "
          f"январь {out[have.index('t2m'), 2, 14].mean()-273.15:.1f} °C, "
          f"июль {out[have.index('t2m'), 2, 195].mean()-273.15:.1f} °C")


if __name__ == "__main__":
    main()
