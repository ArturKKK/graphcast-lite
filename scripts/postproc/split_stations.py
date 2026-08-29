#!/usr/bin/env python3
"""Делит корпус по СТАНЦИЯМ: часть на обучение, часть придерживается.

Зачем. Постпроцессор с вложением станции нельзя применить к площадке, которой он
не видел: для неё попросту нет строки вложения. Значит вопрос «работает ли он на
новой станции» без такого деления даже не поставить, а вопрос этот главный для
оперативного применения: если поправку надо настраивать годами наблюдений на
каждой площадке, поставить её на новую нельзя.

Отбор придерживаемых станций делается ПО ЖРЕБИЮ с заданным зерном, а не «первые
N по списку»: список отсортирован по числу наблюдений, и первые N оказались бы
сплошь самыми полными станциями, а последние — самыми редкими. И то и другое
дало бы смещённую оценку.

    python3 scripts/postproc/split_stations.py --in corpus.parquet --out-dir DIR \\
        --prefix st --holdout 14 --seed 42

На выходе <prefix>_seen.parquet и <prefix>_unseen.parquet, а также список
придержанных станций в <prefix>_holdout.json — чтобы прогон был воспроизводим.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def choose_holdout(stations, n_holdout: int, seed: int) -> list:
    """Выбрать станции, которые придерживаем. Возвращает отсортированный список."""
    uniq = sorted(set(stations))
    if n_holdout <= 0:
        raise SystemExit("--holdout должно быть больше нуля")
    if n_holdout >= len(uniq):
        raise SystemExit(
            f"придержать просят {n_holdout} станций из {len(uniq)} — "
            f"на обучение не останется ничего")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(uniq, size=n_holdout, replace=False).tolist())


def choose_train(stations, holdout: list, keep: int | None, seed: int) -> list:
    """Какие станции оставить на обучение. keep=None — все, кроме придержанных.

    Нужно, чтобы менять РАЗМЕР обучающей выборки, не трогая проверочную. Иначе
    опыт «зависит ли перенос от числа станций» неразрешим: меняя число
    придержанных, меняешь заодно состав проверки, и числа перестают быть
    сравнимыми. 29.08.2026 на это и напоролись — при семи придержанных станциях
    одна трудная определяла весь результат.
    """
    rest = sorted(set(stations) - set(holdout))
    if keep is None or keep >= len(rest):
        return rest
    if keep < 1:
        raise SystemExit("--keep должно быть не меньше единицы")
    rng = np.random.default_rng(seed + 1000)   # другой поток, чем у придержанных
    return sorted(rng.choice(rest, size=keep, replace=False).tolist())


def split_by_stations(df: pd.DataFrame, holdout: list,
                      col: str = "station_usaf",
                      train: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить на обучающую и придержанную части.

    ``train`` позволяет взять лишь часть непридержанных станций: придержанная
    часть при этом не меняется вовсе, и проверка остаётся сравнимой.
    """
    held = df[col].isin(holdout)
    seen = df[~held]
    if train is not None:
        seen = seen[seen[col].isin(train)]
    return seen, df[held]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--holdout", type=int, required=True,
                    help="сколько станций придержать")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--station-col", default="station_usaf")
    ap.add_argument("--keep", type=int, default=None,
                    help="сколько станций оставить на обучение (по умолчанию все "
                         "непридержанные). Меняет РАЗМЕР обучения, не трогая "
                         "проверку — иначе сравнивать нечего")
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    holdout = choose_holdout(df[a.station_col], a.holdout, a.seed)
    train = choose_train(df[a.station_col], holdout, a.keep, a.seed)
    seen, unseen = split_by_stations(df, holdout, a.station_col, train)
    if seen.empty or unseen.empty:
        raise SystemExit("одна из частей пуста — проверь --holdout")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("seen", seen), ("unseen", unseen)):
        p = out / f"{a.prefix}_{name}.parquet"
        part.to_parquet(p, index=False)
        print(f"  {name}: {len(part):,} строк, "
              f"{part[a.station_col].nunique()} станций -> {p}", flush=True)
    (out / f"{a.prefix}_holdout.json").write_text(json.dumps(
        {"seed": a.seed, "holdout": holdout, "train": train,
         "n_train": len(train)}, ensure_ascii=False, indent=1))
    print(f"  на обучении станций: {len(train)}", flush=True)
    print(f"  придержаны станции: {', '.join(map(str, holdout))}", flush=True)


if __name__ == "__main__":
    main()
