#!/usr/bin/env python3
"""Пересчитывает составляющие ветра в готовом корпусе, не пересобирая его.

Зачем. При штиле ISD-Lite ставит скорость 0, а направление помечает пропуском —
направления у штиля нет. Составляющие считались как u = -V·sin(dd), и при
пропущенном направлении выходил NaN, хотя вектор очевидно нулевой. Из-за этого
штили выпадали не только из оценки ветра: датасет постпроцессора выбрасывает
строку, если пропущена ЛЮБАЯ из трёх целей, — значит вместе с ветром терялась и
температура этих сроков. А штиль это чаще всего антициклон с инверсией, то есть
как раз те случаи, где ошибка модели по температуре наибольшая.

Пересобирать корпус ради этого не нужно: скорость и направление в нём сохранены,
испорчены только производные столбцы. Пересчёт занимает секунды против двух с
половиной часов развёртки.

Запуск:
    python3 scripts/postproc/fix_calm_wind.py --corpus ПУТЬ [--out ПУТЬ]

Без --out файл переписывается на месте.
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

from src.postprocessing.corpus_math import wind_components


def fix_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Пересчитать obs_u10/obs_v10. Возвращает кадр и что именно изменилось."""
    for c in ("obs_ws", "obs_wd"):
        if c not in df.columns:
            raise SystemExit(
                f"в корпусе нет столбца {c} — пересчитать ветер не из чего, "
                f"придётся пересобирать корпус")
    before_nan = int(df["obs_u10"].isna().sum()) if "obs_u10" in df.columns else len(df)
    u, v = wind_components(df["obs_ws"], df["obs_wd"])
    df = df.copy()
    df["obs_u10"], df["obs_v10"] = u, v
    after_nan = int(np.isnan(u).sum())
    calm = int((df["obs_ws"].to_numpy() == 0.0).sum())
    return df, {
        "строк": len(df),
        "штилей": calm,
        "было пропусков в ветре": before_nan,
        "стало пропусков": after_nan,
        "возвращено строк": before_nan - after_nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=None, help="куда писать (по умолчанию на место)")
    a = ap.parse_args()
    src = Path(a.corpus)
    df = pd.read_parquet(src)
    df, stat = fix_frame(df)
    for k, v in stat.items():
        print(f"  {k}: {v:,}")
    if stat["возвращено строк"]:
        share = stat["возвращено строк"] / stat["строк"] * 100
        print(f"  доля возвращённого: {share:.1f} % корпуса")
    out = Path(a.out) if a.out else src
    df.to_parquet(out, index=False)
    print(f"[ветер] записано {out}", flush=True)


if __name__ == "__main__":
    main()
