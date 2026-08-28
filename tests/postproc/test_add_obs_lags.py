"""Признаки из наблюдений станции: главное — отсутствие подсказки из будущего.

Признак «что станция намерила перед выпуском прогноза» — самый сильный из всех,
что есть у постпроцессора: линейной регрессии он поднял выигрыш с 0,5 % до
12,1 %. Ровно поэтому он и самый опасный. Стоит поиску съехать на срок действия
вместо срока выпуска — и модель будет «предсказывать» по ответу, показывая на
проверке прекрасные числа, которых в работе не будет никогда.
"""
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "postproc" / "add_obs_lags.py"


def make_corpus(path: Path, *, n_stations=3, days=120, leads=(6, 24, 120)):
    """Корпус, где наблюдение несёт в себе собственное время.

    Значение равно числу часов от начала отсчёта, поэтому по любому попавшему в
    признак числу видно, из какого момента оно взято, — и утечку не приходится
    искать косвенно.
    """
    t0 = pd.Timestamp("2019-01-01")
    rows = []
    for st in range(n_stations):
        for d in range(days):
            for init_h in (0, 12):
                init = t0 + pd.Timedelta(days=d, hours=init_h)
                for lead in leads:
                    valid = init + pd.Timedelta(hours=lead)
                    hours = (valid - t0).total_seconds() / 3600.0
                    rows.append({
                        "station_usaf": f"2000{st}",
                        "init_time_utc": init,
                        "valid_time_utc": valid,
                        "lead_h": lead,
                        # наблюдение = час от начала отсчёта, в кельвинах
                        "obs_t2m_K": 273.15 + hours,
                        "gnn_t2m": 273.15 + hours,
                        "obs_u10": float(hours), "gnn_u10": float(hours),
                        "obs_v10": float(hours), "gnn_v10": float(hours),
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def run_script(inp: Path, out: Path, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--in", str(inp),
                        "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_lags_never_come_from_the_future(tmp_path):
    """Ни одно значение признака не взято позже момента выпуска."""
    inp, out = tmp_path / "c.parquet", tmp_path / "c_lag.parquet"
    make_corpus(inp)
    run_script(inp, out, "--clim-years", "2019")
    df = pd.read_parquet(out)

    t0 = pd.Timestamp("2019-01-01")
    init_hours = (df["init_time_utc"] - t0).dt.total_seconds() / 3600.0
    for lag in (0, 6, 12, 24):
        col = df[f"obs_t2m_lag{lag}"]
        got = col.dropna()
        # значение признака — это час, из которого взято наблюдение
        assert (got <= init_hours[got.index] + 1e-6).all(), (
            f"obs_t2m_lag{lag}: есть значения позже выпуска — утечка")
        # и это именно запрошенный сдвиг, а не что попало
        assert np.allclose(got, init_hours[got.index] - lag)


def test_error_lags_never_come_from_the_future(tmp_path):
    """То же для признаков недавней ошибки модели."""
    inp, out = tmp_path / "c.parquet", tmp_path / "c_lag.parquet"
    make_corpus(inp)
    run_script(inp, out, "--clim-years", "2019")
    df = pd.read_parquet(out)
    for lag in (0, 6, 12, 24):
        col = f"err_{'t2m'}_lag{lag}"
        assert col in df.columns
        # прогноз и наблюдение в этом корпусе совпадают, значит ошибка ровно нуль;
        # любое иное значение означало бы, что срослись разные моменты времени
        got = df[col].dropna()
        assert np.allclose(got, 0.0), f"{col}: ошибка не нулевая — срослись сроки"


def test_age_is_never_negative(tmp_path):
    """Возраст последнего наблюдения не бывает отрицательным."""
    inp, out = tmp_path / "c.parquet", tmp_path / "c_lag.parquet"
    make_corpus(inp)
    run_script(inp, out, "--clim-years", "2019")
    age = pd.read_parquet(out)["obs_lag_age_h"].dropna()
    assert (age >= 0).all()


def test_all_three_variables_get_features(tmp_path):
    """Признаки строятся по температуре и по обеим составляющим ветра.

    До 28.08.2026 они были только по температуре, и регрессия по ветру не знала,
    какой был ветер: наблюдения давали ей 14,0 % против 14,1 %, то есть ничего.
    """
    inp, out = tmp_path / "c.parquet", tmp_path / "c_lag.parquet"
    make_corpus(inp)
    run_script(inp, out, "--clim-years", "2019")
    cols = set(pd.read_parquet(out).columns)
    for var in ("t2m", "u10", "v10"):
        for suffix in ("lag0", "lag24", "tend24", "anom"):
            assert f"obs_{var}_{suffix}" in cols, f"нет obs_{var}_{suffix}"
        assert f"err_{var}_lag_mean" in cols, f"нет err_{var}_lag_mean"


def test_climatology_uses_only_given_years(tmp_path):
    """Норма считается по заданным годам, а не по всему корпусу.

    Иначе в признак отклонения от нормы попадает проверочный год, и это утечка,
    которую по метрикам не отличить от честного улучшения.
    """
    t0 = pd.Timestamp("2018-01-01")
    rows = []
    # Три станции: у сборщика есть порог в 1000 наблюдений, ниже которого
    # признаки переменной не строятся вовсе, и на одной станции за 400 дней
    # набирается только 400 — тест бы проверял пустоту.
    # Выпуски в 00 и 12 UTC со сроками 6 и 12 ч дают наблюдения в 00, 06, 12 и
    # 18 — то есть в том числе В МОМЕНТ выпуска. Это обязательно: лаг ищет
    # ближайшее наблюдение не позже выпуска, но не дальше полутора часов, и в
    # корпусе, где наблюдения есть только на сроках действия, все лаги пусты.
    # В настоящем корпусе так и есть — выпуски в 00 и 12, а станции отчитываются
    # ежечасно.
    for st in range(3):
        for d in range(400):                  # захватывает 2018 и 2019
            for init_h in (0, 12):
                init = t0 + pd.Timedelta(days=d, hours=init_h)
                for lead in (6, 12):
                    valid = init + pd.Timedelta(hours=lead)
                    # в 2018 наблюдения около 10, в 2019 — около 30
                    base = 10.0 if valid.year == 2018 else 30.0
                    rows.append({"station_usaf": f"2000{st}", "init_time_utc": init,
                                 "valid_time_utc": valid, "lead_h": lead,
                                 "obs_t2m_K": 273.15 + base, "gnn_t2m": 273.15 + base,
                                 "obs_u10": 0.0, "gnn_u10": 0.0,
                                 "obs_v10": 0.0, "gnn_v10": 0.0})
    inp, out = tmp_path / "c.parquet", tmp_path / "c_lag.parquet"
    pd.DataFrame(rows).to_parquet(inp, index=False)
    run_script(inp, out, "--clim-years", "2018")
    df = pd.read_parquet(out)
    y = df["valid_time_utc"].dt.year
    # норма посчитана по 2018 (≈10), значит в 2019 отклонение ≈ +20, а не ≈ 0
    anom_2019 = df.loc[y == 2019, "obs_t2m_anom"].dropna()
    assert len(anom_2019) > 50
    assert anom_2019.mean() == pytest.approx(20.0, abs=1.0), (
        "отклонение посчитано от нормы самого проверочного года — утечка")
