"""Деление корпуса по станциям.

Без него нельзя поставить главный для оперативного применения вопрос: работает
ли поправка на площадке, которой модель не видела. Ошибка в делении — например,
станция, попавшая в обе части, — сделала бы ответ на этот вопрос ложным и
незаметно завышенным.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import load_module  # noqa: E402

ss = load_module("scripts/postproc/split_stations.py", "split_stations")


def make_df(n_st=20, rows_each=5):
    return pd.DataFrame({
        "station_usaf": np.repeat([f"2{i:04d}" for i in range(n_st)], rows_each),
        "v": np.arange(n_st * rows_each),
    })


def test_holdout_has_the_requested_size():
    got = ss.choose_holdout(make_df()["station_usaf"], 6, seed=1)
    assert len(got) == 6 and len(set(got)) == 6


def test_holdout_is_reproducible_by_seed():
    """Один жребий — один и тот же набор: иначе прогон не повторить."""
    a = ss.choose_holdout(make_df()["station_usaf"], 5, seed=7)
    b = ss.choose_holdout(make_df()["station_usaf"], 5, seed=7)
    assert a == b
    assert ss.choose_holdout(make_df()["station_usaf"], 5, seed=8) != a


def test_holdout_is_not_just_the_first_stations():
    """Отбор по жребию, а не по порядку.

    Список станций отсортирован по числу наблюдений: первые N оказались бы
    сплошь самыми полными, последние — самыми редкими. И то и другое дало бы
    смещённую оценку переноса.
    """
    uniq = sorted(set(make_df(n_st=30)["station_usaf"]))
    got = ss.choose_holdout(make_df(n_st=30)["station_usaf"], 8, seed=3)
    assert got != uniq[:8] and got != uniq[-8:]


def test_parts_do_not_share_a_single_station():
    """Главный инвариант: станция не может попасть в обе части.

    Попади — и «неизвестная» станция окажется знакомой, а оценка переноса
    завышенной, причём заметить это по числам будет нельзя.
    """
    df = make_df()
    holdout = ss.choose_holdout(df["station_usaf"], 6, seed=2)
    seen, unseen = ss.split_by_stations(df, holdout)
    assert set(seen["station_usaf"]) & set(unseen["station_usaf"]) == set()
    assert set(unseen["station_usaf"]) == set(holdout)


def test_no_row_is_lost_or_duplicated():
    df = make_df()
    seen, unseen = ss.split_by_stations(df, ss.choose_holdout(df["station_usaf"], 4, 5))
    assert len(seen) + len(unseen) == len(df)
    assert sorted(pd.concat([seen, unseen])["v"]) == sorted(df["v"])


def test_asking_for_all_stations_is_refused():
    with pytest.raises(SystemExit, match="не останется ничего"):
        ss.choose_holdout(make_df(n_st=10)["station_usaf"], 10, seed=0)


def test_asking_for_none_is_refused():
    with pytest.raises(SystemExit, match="больше нуля"):
        ss.choose_holdout(make_df()["station_usaf"], 0, seed=0)
