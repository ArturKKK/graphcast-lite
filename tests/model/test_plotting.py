"""Общие части рисования полей.

Не требует matplotlib: проверяются расчёт пределов и порядок осей, а они на
numpy. Сами панели — тонкая обёртка над imshow, проверять там нечего.

Почему это стоит тестов. Ошибка в рисовании не падает и не выглядит ошибкой:
получается картинка, которая читается как карта погоды. Забытое
транспонирование поворачивает карту, несимметричные пределы сдвигают нулевую
линию у карты ошибки, разные пределы у правды и прогноза делают их несравнимыми.
Всё это выглядит убедительно и всё это неверно.
"""
import numpy as np
import pytest

from src.plotting import (DIV_CMAP, SEQ_CMAP, shared_limits, symmetric_limit,
                          to_map_orientation)


# --- порядок осей ------------------------------------------------------------

def test_orientation_swaps_lon_and_lat():
    """Поле (долгота, широта) превращается в (широта, долгота)."""
    f = np.zeros((7, 3))                      # 7 долгот, 3 широты
    assert to_map_orientation(f).shape == (3, 7)


def test_orientation_moves_the_right_values():
    f = np.arange(6).reshape(3, 2)            # (долгота=3, широта=2)
    got = to_map_orientation(f)
    for i in range(3):
        for j in range(2):
            assert got[j, i] == f[i, j], "значение уехало не туда"


def test_orientation_refuses_a_non_2d_field():
    """Трёхмерный вход — отказ, а не молчаливая перестановка не тех осей."""
    with pytest.raises(ValueError, match="двумерное"):
        to_map_orientation(np.zeros((2, 3, 4)))


# --- общие пределы -----------------------------------------------------------

def test_shared_limits_cover_all_fields():
    """Правда и прогноз получают одну шкалу — иначе они несравнимы на глаз."""
    a = np.array([[0.0, 5.0]])
    b = np.array([[-3.0, 2.0]])
    assert shared_limits(a, b) == (-3.0, 5.0)


def test_shared_limits_ignore_missing_values():
    a = np.array([1.0, np.nan, 9.0])
    assert shared_limits(a) == (1.0, 9.0)


def test_shared_limits_survive_an_all_nan_field():
    """Поле из одних пропусков не должно ронять рисование."""
    lo, hi = shared_limits(np.full(5, np.nan))
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_percentile_cuts_the_tails():
    """Одна выпавшая точка не должна задавать всю шкалу.

    Без отсечения хвостов единственный выброс делает поле одноцветным, и
    картинка перестаёт что-либо показывать.
    """
    x = np.concatenate([np.random.default_rng(0).normal(0, 1, 10000), [1000.0]])
    full = shared_limits(x)
    cut = shared_limits(x, percentile=99)
    assert full[1] > 900
    assert cut[1] < 10


# --- симметрия у карт ошибки -------------------------------------------------

def test_symmetric_limit_takes_the_largest_absolute_value():
    """Нуль обязан быть ровно в середине шкалы, иначе знак читается неверно."""
    assert symmetric_limit(np.array([-7.0, 2.0])) == pytest.approx(7.0)
    assert symmetric_limit(np.array([1.0, 4.0])) == pytest.approx(4.0)


def test_symmetric_limit_of_a_zero_field_is_not_zero():
    """Идеальный прогноз не должен давать вырожденную шкалу (0, 0)."""
    assert symmetric_limit(np.zeros(10)) > 0


def test_symmetric_limit_ignores_missing_values():
    assert symmetric_limit(np.array([np.nan, -3.0, 1.0])) == pytest.approx(3.0)


def test_symmetric_limit_across_several_fields():
    assert symmetric_limit(np.array([1.0]), np.array([-6.0])) == pytest.approx(6.0)


# --- палитры -----------------------------------------------------------------

def test_default_colormaps_are_not_jet():
    """«jet» неравномерна по восприятию: рисует полосы там, где поле гладкое.

    Она создаёт ложную структуру и плохо читается при дальтонизме. В разведочных
    скриптах так и было; в общем модуле умолчание другое.
    """
    assert SEQ_CMAP != "jet"
    assert DIV_CMAP != "jet"


def test_diverging_colormap_is_used_for_errors_not_the_sequential_one():
    assert SEQ_CMAP != DIV_CMAP
