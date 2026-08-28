"""Нарезка корпуса по годам.

Годы частей не должны пересекаться, а год берётся по сроку ДЕЙСТВИЯ прогноза.
Выпуск 31 декабря со сроком +120 ч действует уже в следующем году: разложи такие
строки по сроку выпуска — и в проверочной выборке окажутся сроки обучающих лет.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import load_module  # noqa: E402

sc = load_module("scripts/postproc/split_corpus.py", "split_corpus")


def make_df():
    rows = []
    for year in (2018, 2019, 2020):
        for d in range(10):
            init = pd.Timestamp(f"{year}-06-{d + 1:02d}")
            rows.append({"init_time_utc": init,
                         "valid_time_utc": init + pd.Timedelta(hours=24),
                         "v": year})
    # выпуск в конце года, срок действия — уже в следующем
    init = pd.Timestamp("2019-12-31 12:00")
    rows.append({"init_time_utc": init,
                 "valid_time_utc": init + pd.Timedelta(hours=120),
                 "v": 9999})
    return pd.DataFrame(rows)


def test_parts_are_split_by_valid_time():
    df = make_df()
    parts = sc.split_by_years(df, {"train": [2018], "test": [2020]})
    assert set(parts["train"]["v"]) == {2018}
    # строка с выпуском 31.12.2019 и сроком +120 ч действует в 2020 -> в проверку
    assert 9999 in set(parts["test"]["v"])


def test_split_by_init_time_puts_it_elsewhere():
    """Тот же корпус по сроку выпуска кладёт пограничную строку иначе."""
    df = make_df()
    parts = sc.split_by_years(df, {"train": [2019], "test": [2020]},
                              time_col="init_time_utc")
    assert 9999 in set(parts["train"]["v"])


def test_overlapping_years_are_refused():
    """Пересечение частей — это утечка, и оно должно быть отказом, а не молчанием."""
    with pytest.raises(SystemExit, match="пересеклись"):
        sc.split_by_years(make_df(), {"train": [2018, 2019], "val": [2019]})


def test_parts_are_parsed():
    got = sc.parse_parts(["train=2016,2017", "test=2020"])
    assert got == {"train": [2016, 2017], "test": [2020]}


@pytest.mark.parametrize("bad", ["train", "train=", "train=две-тысячи"])
def test_bad_part_spec_is_refused(bad):
    with pytest.raises(SystemExit):
        sc.parse_parts([bad])
