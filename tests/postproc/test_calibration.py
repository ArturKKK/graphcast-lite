"""Проверка калибровки: соответствует ли заявленный разброс действительной ошибке.

Тесты на чистой математике, без модели. Проверяется, что величины означают
именно то, что заявлено: иначе по ним нельзя судить, можно ли верить σ, — а
именно ради этого суждения всё и считается.
"""
import math
import numpy as np
import pytest

from src.postprocessing import calibration as cal


# --- непрерывная оценка ------------------------------------------------------

def test_crps_of_a_point_forecast_equals_the_error():
    """При σ→0 оценка вырождается в модуль ошибки.

    Это связывает вероятностный прогноз с точечным: переменная σ обязана давать
    меньше, иначе она не нужна.
    """
    got = cal.crps_gaussian(np.array([0.0]), np.array([1e-9]), np.array([2.0]))
    assert got[0] == pytest.approx(2.0, abs=1e-4)


def test_crps_rewards_being_right():
    """Попал точнее — оценка меньше, при том же разбросе."""
    close = cal.crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.1]))
    far = cal.crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([3.0]))
    assert close[0] < far[0]


def test_crps_prefers_honest_uncertainty():
    """При большой ошибке честно широкий разброс лучше самоуверенно узкого.

    Это и делает оценку осмысленной: она наказывает и за промах, и за ложную
    уверенность.
    """
    obs = np.array([3.0])
    narrow = cal.crps_gaussian(np.array([0.0]), np.array([0.3]), obs)[0]
    honest = cal.crps_gaussian(np.array([0.0]), np.array([3.0]), obs)[0]
    assert honest < narrow


def test_crps_punishes_being_needlessly_vague():
    """А при малой ошибке широкий разброс, наоборот, хуже узкого."""
    obs = np.array([0.05])
    narrow = cal.crps_gaussian(np.array([0.0]), np.array([0.1]), obs)[0]
    wide = cal.crps_gaussian(np.array([0.0]), np.array([5.0]), obs)[0]
    assert narrow < wide


def test_crps_is_minimised_by_the_true_spread():
    """У правильного разброса оценка минимальна — на выборке из него же."""
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 2.0, 20000)
    mu = np.zeros_like(obs)
    scores = {s: cal.crps_gaussian(mu, np.full_like(obs, s), obs).mean()
              for s in (0.5, 1.0, 2.0, 4.0, 8.0)}
    assert min(scores, key=scores.get) == 2.0


# --- надёжность по корзинам --------------------------------------------------

def test_reliability_splits_into_equal_bins():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.1, 3.0, 1000)
    err = rng.normal(0, sigma)
    rows = cal.reliability(sigma, err, n_bins=10)
    assert len(rows) == 10
    assert sum(r["n"] for r in rows) == 1000
    assert max(r["n"] for r in rows) - min(r["n"] for r in rows) <= 1


def test_reliability_bins_are_ordered_by_declared_spread():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 3.0, 2000)
    rows = cal.reliability(sigma, rng.normal(0, sigma), n_bins=5)
    means = [r["sigma_mean"] for r in rows]
    assert means == sorted(means)


def test_well_calibrated_model_matches_in_every_bin():
    """Если σ честная, в каждой корзине заявленное совпадает с действительным.

    Средние величины могут сойтись случайно — при завышенной σ на лёгких случаях
    и заниженной на трудных. Разбивка по корзинам это и ловит.
    """
    rng = np.random.default_rng(3)
    sigma = rng.uniform(0.2, 4.0, 200000)
    err = rng.normal(0, sigma)
    for r in cal.reliability(sigma, err, n_bins=8):
        assert r["rmse"] / r["sigma_mean"] == pytest.approx(1.0, abs=0.08), r


def test_overconfident_model_is_exposed():
    """Модель, занижающая разброс вдвое, даёт отношение около двух."""
    rng = np.random.default_rng(4)
    sigma_true = rng.uniform(0.2, 4.0, 50000)
    err = rng.normal(0, sigma_true)
    for r in cal.reliability(sigma_true / 2.0, err, n_bins=5):
        assert r["rmse"] / r["sigma_mean"] > 1.7


def test_bins_catch_a_mismatch_that_averages_hide():
    """Средние сходятся, а по корзинам видно, что σ перепутана местами.

    Именно ради этого случая разбивка и делается: модель, объявляющая большую
    неуверенность там, где ошибается мало, и наоборот, в среднем выглядит
    безупречно.
    """
    rng = np.random.default_rng(5)
    n = 40000
    err = np.concatenate([rng.normal(0, 0.5, n), rng.normal(0, 4.0, n)])
    sigma_swapped = np.concatenate([np.full(n, 4.0), np.full(n, 0.5)])
    overall = np.sqrt((err ** 2).mean()) / np.sqrt((sigma_swapped ** 2).mean())
    assert overall == pytest.approx(1.0, abs=0.05), "в среднем расхождения нет"
    ratios = [r["rmse"] / r["sigma_mean"] for r in cal.reliability(sigma_swapped, err, 2)]
    assert max(ratios) / min(ratios) > 5, "разбивка не выявила подмену"
