#!/usr/bin/env python3
"""Считает описатели рельефа вокруг станций и приклеивает их к корпусу.

Признаки рельефа СТАТИЧНЫ: они не зависят ни от срока, ни от времени. Поэтому
корпус пересобирать не нужно — четыре часа развёртки не повторяются, таблица на
71 строку считается за минуту и приклеивается по номеру станции.

    python3 scripts/postproc/add_terrain.py --corpus C.parquet --out O.parquet \\
        --stations data/krsk_postproc_stations.json --dem-dir data/dem

Промежуточная таблица сохраняется рядом (--terrain-json): по ней видно, что
получилось у каждой станции, и её можно приклеить к другому корпусу мгновенно.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing.stations import load_stations
from src.postprocessing.terrain import load_mosaic, station_terrain


def build_table(stations, dem_dir, radii_m, margin_deg=0.35) -> pd.DataFrame:
    """Описатели для каждой станции. Пропущенные листы не роняют счёт."""
    rows, skipped = [], []
    for s in stations:
        usaf = str(s["usaf"])
        lat, lon = float(s["lat"]), float(s["lon"])   # долгота уже в [-180, 180]
        try:
            dem, center, cell = load_mosaic(dem_dir, lat, lon, margin_deg)
        except FileNotFoundError as e:
            skipped.append((usaf, str(e).split(" — ")[0]))
            continue
        rec = station_terrain(dem, center, lat, cell, radii_m)
        rec["station_usaf"] = usaf
        # Разность между высотой из матрицы и заявленной в описании станции.
        # Расхождение в сотню метров означает, что координаты станции неточны, а
        # тогда и все описатели вокруг неё считаны не в том месте.
        rec["terr_elev_mismatch"] = rec["terr_dem_elev"] - float(s.get("elev", np.nan))
        rows.append(rec)
    if skipped:
        print(f"[рельеф] без листов осталось станций: {len(skipped)}", flush=True)
        for usaf, why in skipped[:5]:
            print(f"    {usaf}: {why}", flush=True)
    return pd.DataFrame(rows)


def prepare_outputs(*paths) -> None:
    """Создать каталоги и убедиться, что запись пройдёт — ДО тяжёлого счёта.

    30.08.2026 описатели на 71 станцию считались две минуты и были выброшены,
    потому что каталога для побочной таблицы не существовало. Условие, которое
    проверяется за миллисекунду, не должно обнаруживаться в конце работы.
    """
    for pth in paths:
        if pth is None:
            continue
        pth = Path(pth)
        pth.parent.mkdir(parents=True, exist_ok=True)
        probe = pth.parent / (".write_probe_" + pth.name)
        try:
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            raise SystemExit(f"не могу писать в {pth.parent}: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--dem-dir", required=True)
    ap.add_argument("--terrain-json", default=None,
                    help="куда сохранить таблицу по станциям (по умолчанию рядом с --out)")
    ap.add_argument("--radii-km", type=float, nargs="+", default=[1.0, 5.0, 20.0],
                    help="радиусы окрестности: 1 км — сама площадка, 5 км — форма "
                         "долины, 20 км — положение в крупном рельефе")
    a = ap.parse_args()

    # Проверяем условия ДО счёта: описатели считаются минуты, и терять их
    # из-за отсутствующего каталога недопустимо.
    tj = Path(a.terrain_json) if a.terrain_json else \
        Path(a.out).with_name(Path(a.out).stem + "_terrain.json")
    prepare_outputs(a.out, tj)

    stations = load_stations(a.stations)
    table = build_table(stations, a.dem_dir, [r * 1000 for r in a.radii_km])
    if table.empty:
        raise SystemExit("ни одной станции не посчитано — проверь --dem-dir")

    terr_cols = [c for c in table.columns if c.startswith("terr_")]
    print(f"[рельеф] станций посчитано: {len(table)}, признаков: {len(terr_cols)}",
          flush=True)
    for c in ("terr_tpi_1km", "terr_tpi_20km", "terr_slope", "terr_horizon",
              "terr_elev_mismatch"):
        if c in table:
            v = table[c].dropna()
            if len(v):
                print(f"    {c:<22} от {v.min():8.1f} до {v.max():8.1f}, "
                      f"медиана {v.median():7.1f}", flush=True)

    tj.write_text(table.to_json(orient="records", force_ascii=False, indent=1))
    print(f"[рельеф] таблица по станциям: {tj}", flush=True)

    df = pd.read_parquet(a.corpus)
    before = len(df)
    df["station_usaf"] = df["station_usaf"].astype(str)
    df = df.merge(table, on="station_usaf", how="left")
    assert len(df) == before, "склейка размножила строки — проверь номера станций"
    missing = df[terr_cols[0]].isna().mean() * 100 if terr_cols else 100.0
    print(f"[рельеф] строк {len(df):,}, без рельефа {missing:.1f} %", flush=True)
    df.to_parquet(a.out, index=False)
    print(f"[рельеф] записано {a.out}", flush=True)


if __name__ == "__main__":
    main()
