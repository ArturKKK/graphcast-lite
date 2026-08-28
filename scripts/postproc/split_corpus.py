#!/usr/bin/env python3
"""Режет корпус постобработки по годам.

Раньше каждый раннер делал это своим куском кода внутри heredoc — три почти
одинаковых куска, и разойтись они могли незаметно. Теперь один скрипт, и он
покрыт тестами.

Годы задаются парами «имя=годы»:

    python3 scripts/postproc/split_corpus.py --in corpus.parquet --out-dir DIR \\
        --prefix krsk train=2016,2017,2018 val=2019 test=2020

Год берётся по СРОКУ ДЕЙСТВИЯ прогноза, а не по сроку выпуска: проверочная
выборка не должна содержать сроков, попадающих в обучающие годы. Выпуск 31
декабря со сроком +120 ч действует уже в следующем году, и по выпуску такая
строка ушла бы не в ту часть.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_by_years(df: pd.DataFrame, parts: dict[str, list[int]],
                   time_col: str = "valid_time_utc") -> dict[str, pd.DataFrame]:
    """Разложить строки по частям. Год берётся из time_col."""
    year = pd.to_datetime(df[time_col]).dt.year
    seen: dict[int, str] = {}
    for name, years in parts.items():
        for y in years:
            if y in seen:
                raise SystemExit(
                    f"год {y} указан и в «{seen[y]}», и в «{name}» — "
                    f"части пересеклись бы, а это утечка обучения в проверку")
            seen[y] = name
    return {name: df[year.isin(years)] for name, years in parts.items()}


def parse_parts(items: list[str]) -> dict[str, list[int]]:
    parts: dict[str, list[int]] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"ожидалось «имя=годы», получено «{it}»")
        name, years = it.split("=", 1)
        try:
            parts[name] = [int(y) for y in years.split(",") if y]
        except ValueError:
            raise SystemExit(f"в «{it}» годы не разобрались")
        if not parts[name]:
            raise SystemExit(f"в «{it}» не задан ни один год")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="начало имени файлов части")
    ap.add_argument("--time-col", default="valid_time_utc")
    ap.add_argument("parts", nargs="+", metavar="ИМЯ=ГОДЫ",
                    help="например train=2016,2017,2018 val=2019 test=2020")
    a = ap.parse_args()

    parts = parse_parts(a.parts)
    df = pd.read_parquet(a.inp)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in split_by_years(df, parts, a.time_col).items():
        if part.empty:
            raise SystemExit(f"часть «{name}» пуста — проверь годы {parts[name]}")
        p = out / f"{a.prefix}_{name}.parquet"
        part.to_parquet(p, index=False)
        print(f"  {name}: {len(part):,} строк -> {p}", flush=True)


if __name__ == "__main__":
    main()
