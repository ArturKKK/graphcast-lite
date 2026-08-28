"""Базовые линии постобработки: арифметика таблиц и безопасность отбора признаков.

Таблица поправок — это то, с чем сравнивается всё остальное. Ошибка в стягивании
к родителю или в подстановке для пустой ячейки не уронит счёт, а тихо сдвинет
опорные числа, и все выводы про «сеть лучше таблицы на 8 %» окажутся ни о чём.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "baselines", ROOT / "scripts" / "postproc" / "baselines.py")
bl = importlib.util.module_from_spec(spec)
sys.modules["baselines"] = bl
spec.loader.exec_module(bl)


def test_shrinkage_matches_the_formula():
    """Стянутое среднее равно (n·среднее + k·родитель) / (n + k)."""
    train = pd.DataFrame({"key": ["a"] * 3 + ["b"], "_r": [1.0, 2.0, 3.0, 10.0]})
    parent = np.full(4, 100.0)
    t = bl.shrunk_table(train, "_r", ["key"], parent, k=2.0)
    assert t["a"] == pytest.approx((6.0 + 2 * 100.0) / (3 + 2))
    assert t["b"] == pytest.approx((10.0 + 2 * 100.0) / (1 + 2))


def test_shrinkage_pulls_thin_cells_to_the_parent():
    """Ячейка с одним наблюдением почти вся определяется родителем.

    Без этого таблица «станция×месяц×час» переобучается на единичных случаях и
    проигрывает более грубой — то есть выводы получились бы обратными.
    """
    train = pd.DataFrame({"key": ["thin"] + ["fat"] * 500,
                          "_r": [50.0] + [1.0] * 500})
    parent = np.zeros(501)
    t = bl.shrunk_table(train, "_r", ["key"], parent, k=10.0)
    assert abs(t["thin"]) < 5.0, "редкая ячейка не стянута к родителю"
    assert t["fat"] == pytest.approx(1.0, abs=0.03), "полная ячейка стянута зря"


def test_unseen_cell_falls_back_to_parent():
    """Ячейки, которой не было на обучении, берётся значение родителя."""
    table = pd.Series({"a": 5.0}, name="_r")
    df = pd.DataFrame({"key": ["a", "неведомая"]})
    got = bl.apply_table(df, table, ["key"], fallback=np.array([-1.0, -1.0]))
    assert got.tolist() == [5.0, -1.0]


def test_multi_key_tables_apply_correctly():
    """Составной ключ станция×месяц не путает ячейки."""
    table = pd.Series({("s1", 1): 1.0, ("s1", 2): 2.0, ("s2", 1): 3.0})
    df = pd.DataFrame({"station_usaf": ["s2", "s1", "s1"], "month": [1, 2, 1]})
    got = bl.apply_table(df, table, ["station_usaf", "month"], np.zeros(3))
    assert got.tolist() == [3.0, 2.0, 1.0]


def test_metrics_are_what_they_say():
    pred = np.array([1.0, 2.0, 3.0])
    obs = np.array([0.0, 0.0, 0.0])
    m = bl.metrics(pred, obs)
    assert m["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert m["mae"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(2.0)


# --- отбор признаков: сюда не должна попасть цель ---------------------------

@pytest.mark.parametrize("col", [
    "obs_t2m_K", "obs_t2m_C", "obs_u10", "obs_v10", "obs_ws", "obs_wd",
    "gnn_t2m", "obs_t2m", "obs_lag_age_h_extra",
])
def test_targets_never_selected_as_features(col):
    """Ни один столбец наблюдений не проходит под образец имени.

    Попади цель в признаки — регрессия «улучшится» до нуля ошибки, и это будет
    чистая утечка, неотличимая по метрикам от настоящего успеха.
    """
    assert bl.OBS_RE.match(col) is None, f"{col} попал под образец признаков"


@pytest.mark.parametrize("col", [
    "obs_t2m_lag0", "obs_u10_lag24", "err_v10_lag_mean",
    "obs_t2m_tend24", "obs_v10_anom",
])
def test_derived_features_are_selected(col):
    assert bl.OBS_RE.match(col) is not None, f"{col} не опознан как признак"


def test_obs_features_includes_age_and_nothing_else():
    df = pd.DataFrame(columns=["obs_t2m_lag0", "obs_lag_age_h", "obs_u10",
                               "gnn_t2m", "lead_h"])
    assert set(bl.obs_features(df)) == {"obs_t2m_lag0", "obs_lag_age_h"}


def test_time_features_are_periodic():
    df = pd.DataFrame({"valid_time_utc": pd.to_datetime(
        ["2020-01-01 00:00", "2020-01-01 12:00", "2020-07-01 00:00"])})
    out = bl.add_time_features(df)
    assert out["month"].tolist() == [1, 1, 7]
    assert out["hour"].tolist() == [0, 12, 0]
    assert out["sin_hour"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["cos_hour"][0] == pytest.approx(1.0, abs=1e-6)
    assert out["cos_hour"][1] == pytest.approx(-1.0, abs=1e-6)
    # январь и июль должны быть на разных концах годового круга
    assert out["cos_doy"][0] > 0.9 and out["cos_doy"][2] < -0.9
