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


# --- прогон целиком ----------------------------------------------------------

def _corpus_with_known_bias(path, *, n_st=6, years=(2016, 2017, 2018, 2020)):
    """Корпус с известным смещением у каждой станции.

    Прогноз занижен ровно на постоянную величину, своя у станции. Такое смещение
    таблица «станция» обязана снять почти полностью — это самая простая проверка
    того, что весь прогон считает то, что заявлено.
    """
    rng = np.random.default_rng(3)
    rows = []
    for st in range(n_st):
        bias = 1.0 + st                       # 1..6 °C, разное у станций
        for year in years:
            for d in range(120):
                for lead in (6, 24):
                    vt = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=d, hours=lead)
                    obs = 273.15 + 10 * np.cos(2 * np.pi * vt.dayofyear / 365.25) \
                        + rng.normal(0, 1.0)
                    rows.append({
                        "station_usaf": f"2000{st}", "lead_h": lead,
                        "init_time_utc": vt - pd.Timedelta(hours=lead),
                        "valid_time_utc": vt,
                        "obs_t2m_K": obs, "gnn_t2m": obs - bias,
                        "obs_u10": rng.normal(0, 3), "gnn_u10": rng.normal(0, 3),
                        "obs_v10": rng.normal(0, 3), "gnn_v10": rng.normal(0, 3),
                        "lat": 55.0 + st, "lon": 90.0 + st, "elev": 100.0 * st,
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def _run(corpus, *extra):
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "postproc" / "baselines.py"),
         "--corpus", str(corpus), "--train-years", "2016", "2017", "2018",
         "--test-years", "2020", *extra],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _rmse_of(out, label):
    for line in out.splitlines():
        if line.strip().startswith(label):
            return float(line.split()[-4])
    raise AssertionError(f"в выводе нет строки «{label}»:\n{out}")


def test_station_table_removes_a_station_bias(tmp_path):
    """Постоянное смещение станции снимается таблицей «станция» почти нацело."""
    c = tmp_path / "c.parquet"
    _corpus_with_known_bias(c)
    out = _run(c)
    raw = _rmse_of(out, "сырой прогноз")
    fixed = _rmse_of(out, "станция ")
    assert raw > 3.0, "в корпусе заложено смещение 1..6 °C, ошибка должна быть большой"
    assert fixed < 1.2, "таблица «станция» не сняла постоянное смещение"


def test_richer_tables_do_not_lose_to_coarser_ones(tmp_path):
    """Дробление таблицы не должно ухудшать результат.

    Если ухудшает — значит стягивание к родителю сломано, и таблица
    переобучается на редких ячейках. Ровно это и было причиной, по которой
    стягивание вводилось.
    """
    c = tmp_path / "c.parquet"
    _corpus_with_known_bias(c)
    out = _run(c)
    st = _rmse_of(out, "станция ")
    smh = _rmse_of(out, "станция×месяц×час")
    assert smh <= st * 1.05, "подробная таблица заметно хуже грубой"


def test_per_year_error_is_reported(tmp_path):
    c = tmp_path / "c.parquet"
    _corpus_with_known_bias(c)
    out = _run(c)
    assert "сырая ошибка приземной температуры по годам" in out
    for year in (2016, 2020):
        assert f"    {year}:" in out


def test_complete_obs_shrinks_the_sample(tmp_path):
    """Флаг --complete-obs оставляет строки, где есть все три наблюдения."""
    c = tmp_path / "c.parquet"
    df = _corpus_with_known_bias(c)
    df.loc[df.index[:len(df) // 4], "obs_v10"] = np.nan
    df.to_parquet(c, index=False)
    out = _run(c, "--complete-obs")
    assert "только полные наблюдения" in out
    kept = int(out.split("->")[1].split("строк")[0].replace(",", "").strip())
    assert kept == pytest.approx(len(df) * 0.75, rel=0.01)


def test_empty_year_selection_is_refused(tmp_path):
    """Пустая выборка — отказ, а не таблица из NaN."""
    import subprocess
    import sys
    c = tmp_path / "c.parquet"
    _corpus_with_known_bias(c)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "postproc" / "baselines.py"),
         "--corpus", str(c), "--train-years", "1999", "--test-years", "2020"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "пустая выборка" in r.stdout + r.stderr
