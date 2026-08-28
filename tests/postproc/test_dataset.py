"""Датасет нейронного постпроцессора: набор признаков и обращение с пропусками.

Здесь за один день сломалось трижды, и каждый раз молча или почти молча:
  • список признаков дополнялся в датасете, а модель строилась по прежнему
    счёту — обучение падало на первом батче;
  • при оценке список брался из чекпойнта и дополнялся снова — падала оценка;
  • ветровые лаги заполнены на 65-87 %, и обычное среднее обращало столбец, а за
    ним и всю выборку, в NaN.
Тесты закрывают ровно эти три случая.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import load_module  # noqa: E402

_ds = load_module("src/postprocessing/neural/dataset.py", "postproc_dataset")
DEFAULT_FEATURES = _ds.DEFAULT_FEATURES
StationCorpusDataset = _ds.StationCorpusDataset
observation_features = _ds.observation_features

STATIONS = {"20001": 0, "20002": 1}


def make_parquet(path: Path, *, n=2000, obs_cols=(), missing=0.0, seed=0,
                 constant_cols=("urban_flag",)):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n).astype("float32")
                       for c in DEFAULT_FEATURES})
    # у корпуса свои имена, датасет переименовывает их сам
    df["station_lat"] = df.pop("lat")
    df["station_lon"] = df.pop("lon")
    df["station_elev"] = df.pop("elev")
    for c in constant_cols:
        df[c] = 0.0
    df["station_usaf"] = rng.choice(list(STATIONS), n)
    df["init_time_utc"] = pd.Timestamp("2020-01-01")
    df["valid_time_utc"] = pd.Timestamp("2020-01-02")
    df["lead_h"] = 24
    df["obs_t2m_K"] = 273.15 + rng.normal(size=n)
    df["obs_u10"] = rng.normal(size=n)
    df["obs_v10"] = rng.normal(size=n)
    for c in obs_cols:
        v = rng.normal(size=n)
        if missing:
            v[rng.random(n) < missing] = np.nan
        df[c] = v
    df.to_parquet(path, index=False)
    return df


def test_observation_features_never_include_targets():
    df = pd.DataFrame(columns=["obs_t2m_lag0", "err_u10_lag_mean", "obs_lag_age_h",
                               "obs_u10", "obs_v10", "obs_t2m_K", "gnn_t2m"])
    got = set(observation_features(df))
    assert got == {"obs_t2m_lag0", "err_u10_lag_mean", "obs_lag_age_h"}


def test_observation_features_are_added_on_training(tmp_path):
    p = tmp_path / "c.parquet"
    make_parquet(p, obs_cols=("obs_t2m_lag0", "err_v10_lag24", "obs_lag_age_h"))
    ds = StationCorpusDataset(p, station_to_idx=STATIONS)
    for c in ("obs_t2m_lag0", "err_v10_lag24", "obs_lag_age_h"):
        assert c in ds.feature_cols


def test_scalers_pin_the_feature_set(tmp_path):
    """С переданными нормировками список признаков не дополняется.

    Нормировки посчитаны на обучении и задают ширину входа. Дополнить их —
    значит подать модели вход не той ширины: 28.08.2026 так падала оценка
    абляции, в чекпойнте было 26 признаков, а датасет собирал 60.
    """
    p = tmp_path / "c.parquet"
    make_parquet(p, obs_cols=("obs_t2m_lag0", "err_v10_lag24"))
    train = StationCorpusDataset(p, station_to_idx=STATIONS)
    pinned = list(DEFAULT_FEATURES)
    ev = StationCorpusDataset(p, feature_cols=pinned,
                              scalers=train.export_scalers(),
                              station_to_idx=STATIONS)
    assert ev.feature_cols == pinned, "список из чекпойнта был дополнен"
    assert len(ev.feature_cols) < len(train.feature_cols)


def test_missing_values_become_the_column_mean(tmp_path):
    """Пропуск после нормировки — ровно нуль, то есть среднее признака.

    У ветровых лагов заполнено 65-87 %. Обычное среднее по столбцу с пропусками
    даёт NaN, он расходится по всей выборке, и обучение идёт вхолостую.
    """
    p = tmp_path / "c.parquet"
    make_parquet(p, obs_cols=("obs_u10_lag24",), missing=0.35)
    ds = StationCorpusDataset(p, station_to_idx=STATIONS)
    assert np.isfinite(ds.X_norm).all(), "в нормированных признаках остались NaN"
    i = ds.feature_cols.index("obs_u10_lag24")
    raw = ds.X[:, i]
    filled = ds.X_norm[np.isnan(raw), i]
    assert len(filled) > 100
    assert np.allclose(filled, 0.0), "пропуск заменён не средним значением"


def test_constant_column_survives_normalisation(tmp_path):
    """Столбец с нулевой дисперсией не превращается в бесконечность."""
    p = tmp_path / "c.parquet"
    make_parquet(p)
    ds = StationCorpusDataset(p, station_to_idx=STATIONS)
    i = ds.feature_cols.index("urban_flag")
    assert np.isfinite(ds.X_norm[:, i]).all()


def test_elevation_mismatch_feature_is_the_difference(tmp_path):
    """dz_station — это разность высоты станции и рельефа модели.

    Разбор по станциям 28.08.2026: выигрыш поправки связан с высотой станции на
    0,72, и двигает дело именно разность, а не каждая высота по отдельности.
    """
    p = tmp_path / "c.parquet"
    df = make_parquet(p)
    ds = StationCorpusDataset(p, station_to_idx=STATIONS)
    assert "dz_station" in ds.feature_cols
    i = ds.feature_cols.index("dz_station")
    expected = df["station_elev"].to_numpy() - df["z_surf"].to_numpy()
    assert np.allclose(ds.X[:, i], expected, atol=1e-4)


def test_rows_without_a_target_are_dropped(tmp_path):
    """Строка без любого из трёх наблюдений выбрасывается целиком.

    Это свойство, а не изъян: считать ошибку не с чем. Но из-за него выборка
    сети на 21 % меньше корпуса, и базовые линии для сравнения надо считать на
    той же выборке — иначе числа несопоставимы.
    """
    p = tmp_path / "c.parquet"
    df = make_parquet(p, n=1000)
    df.loc[:199, "obs_v10"] = np.nan
    df.to_parquet(p, index=False)
    ds = StationCorpusDataset(p, station_to_idx=STATIONS)
    assert len(ds) == 800


def test_feature_order_is_stable_between_runs(tmp_path):
    """Один и тот же корпус даёт один и тот же порядок признаков.

    Порядок входит в чекпойнт; поплыви он — веса легли бы на другие столбцы, и
    модель тихо считала бы чушь.
    """
    p = tmp_path / "c.parquet"
    make_parquet(p, obs_cols=("obs_t2m_lag0", "err_v10_lag24", "obs_u10_anom"))
    a = StationCorpusDataset(p, station_to_idx=STATIONS).feature_cols
    b = StationCorpusDataset(p, station_to_idx=STATIONS).feature_cols
    assert a == b
