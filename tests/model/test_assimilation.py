"""Усвоение: подтягивание к наблюдениям, маска сшивания границ, оптимальная интерполяция.

Требует torch.

Усвоение дало статье 12,3 % на сутках, и проверять его надо особенно тщательно
по одной причине: почти все его отказы МОЛЧАЛИВЫ. Не сошлись формы — прогноз
возвращается как есть; не подошла маска каналов — усваиваются все переменные
вместо выбранных. Со стороны и то и другое выглядит как «усвоение применили, а
выигрыша нет», и вывод о его бесполезности был бы ложным.
"""
import numpy as np
import pytest

from conftest import needs_torch

pytestmark = needs_torch


@pytest.fixture
def nud():
    from src.assimilation.nudging import (NudgingAssimilator,
                                          build_boundary_taper_mask,
                                          cosine_taper_2d)
    return dict(A=NudgingAssimilator, taper=cosine_taper_2d,
                mask=build_boundary_taper_mask)


# --- подтягивание к наблюдениям ---------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_nudging_follows_its_formula(nud, alpha):
    """x_new = x + alpha·(y − x): при alpha=0 ничего, при alpha=1 ровно наблюдение."""
    import torch
    a = nud["A"](alpha=alpha)
    f = torch.zeros(4, 3)
    o = torch.full((4, 3), 10.0)
    got = a.apply(f, o)
    assert torch.allclose(got, torch.full((4, 3), 10.0 * alpha), atol=1e-6)


def test_missing_observations_leave_the_forecast_alone(nud):
    """Где наблюдения нет, прогноз не трогается вовсе."""
    import torch
    a = nud["A"](alpha=0.5)
    f = torch.full((3, 2), 5.0)
    o = torch.full((3, 2), 9.0)
    o[1, :] = float("nan")
    got = a.apply(f, o)
    assert torch.allclose(got[1], torch.full((2,), 5.0))
    assert torch.allclose(got[0], torch.full((2,), 7.0))


def test_feature_mask_selects_which_variables_are_assimilated(nud):
    import torch
    mask = torch.tensor([True, False])
    a = nud["A"](alpha=1.0, feature_mask_flat=mask)
    f = torch.zeros(3, 2)
    o = torch.full((3, 2), 8.0)
    got = a.apply(f, o)
    assert torch.allclose(got[:, 0], torch.full((3,), 8.0)), "выбранный канал не усвоен"
    assert torch.allclose(got[:, 1], torch.zeros(3)), "невыбранный канал усвоен зря"


def test_shape_mismatch_warns_instead_of_silently_doing_nothing(nud, capsys):
    """Молчаливый отказ — худший из возможных: он неотличим от «не помогло»."""
    import torch
    a = nud["A"](alpha=0.5)
    f = torch.zeros(3, 2)
    got = a.apply(f, torch.zeros(3, 5))
    assert torch.allclose(got, f), "при несовпадении форм прогноз должен вернуться как есть"
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_wrong_feature_mask_length_warns(nud, capsys):
    import torch
    a = nud["A"](alpha=1.0, feature_mask_flat=torch.tensor([True, False, True]))
    a.apply(torch.zeros(2, 2), torch.ones(2, 2))
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_nan_free_output(nud):
    import torch
    a = nud["A"](alpha=0.3)
    o = torch.full((5, 3), 2.0)
    o[::2] = float("nan")
    assert torch.isfinite(a.apply(torch.ones(5, 3), o)).all()


# --- маска сшивания границ ---------------------------------------------------

def test_taper_shape_follows_the_node_order(nud):
    """Маска строится (широта, долгота) — в том же порядке, что выпрямлены узлы.

    До 28.08.2026 она строилась поперёк: на сетке 4x6 треть узлов получала чужой
    вес, а на 512x256 маска была бы бессмысленной целиком.
    """
    n_lat, n_lon, b = 4, 6, 2
    flat = nud["mask"](n_lat, n_lon, b, b).numpy()
    assert flat.shape == (n_lat * n_lon,)
    grid = flat.reshape(n_lat, n_lon)
    # у окна Ханна края нулевые по обеим осям
    assert np.allclose(grid[0], 0.0) and np.allclose(grid[-1], 0.0)
    assert np.allclose(grid[:, 0], 0.0) and np.allclose(grid[:, -1], 0.0)


def test_taper_is_one_in_the_middle(nud):
    flat = nud["mask"](9, 11, 2, 2).numpy().reshape(9, 11)
    assert flat[4, 5] == pytest.approx(1.0)


def test_zero_border_gives_all_ones(nud):
    assert nud["mask"](5, 7, 0, 0).numpy().min() == pytest.approx(1.0)


def test_border_wider_than_half_the_grid_is_clamped(nud):
    """Спад с двух сторон не перекрывается в середине.

    Иначе вторая запись затирает первую, и окно выходит несимметричным: это
    ловушка, потому что число никто не проверяет, а маска молча портится.
    """
    flat = nud["mask"](6, 6, 10, 10).numpy().reshape(6, 6)
    assert np.allclose(flat, flat[::-1, :], atol=1e-6), "окно несимметрично по широте"
    assert np.allclose(flat, flat[:, ::-1], atol=1e-6), "окно несимметрично по долготе"


# --- оптимальная интерполяция ------------------------------------------------

@pytest.fixture
def oi_cls():
    from src.assimilation.optimal_interpolation import OptimalInterpolation
    return OptimalInterpolation


def test_covariance_falls_off_with_distance(oi_cls):
    """Ковариация ошибки прогноза убывает с расстоянием и равна sigma^2 на нуле."""
    import torch
    lats = np.array([55.0, 55.5, 60.0])
    lons = np.array([93.0, 93.0, 93.0])
    oi = oi_cls(lats, lons, sigma_b=2.0, sigma_o=1.0, L=100000.0,
                device=torch.device("cpu"), flat_grid=True)
    B = oi.B.numpy()
    assert np.allclose(np.diag(B), 4.0, atol=1e-4), "на нуле должно быть sigma_b^2"
    assert B[0, 1] > B[0, 2] > 0, "ковариация не убывает с расстоянием"
    assert np.allclose(B, B.T, atol=1e-6), "матрица несимметрична"


def test_distance_is_the_great_circle(oi_cls):
    """Расстояние считается по дуге большого круга, а не по разности координат."""
    import torch
    oi = oi_cls(np.array([0.0, 0.0]), np.array([0.0, 90.0]), sigma_b=1.0,
                sigma_o=1.0, L=1e7, device=torch.device("cpu"), flat_grid=True)
    d = oi._dist_matrix(oi.grid_coords, oi.grid_coords)
    assert d[0, 1] == pytest.approx(6371000.0 * np.pi / 2, rel=1e-6)


def test_distance_wraps_across_the_meridian(oi_cls):
    """Точки по обе стороны нулевого меридиана близки, а не на полкруга.

    Долготы приходят в диапазоне 0..360, и наивная разность дала бы 359°.
    Гаверсинус периодичен и справляется, но проверить это стоит.
    """
    import torch
    oi = oi_cls(np.array([0.0, 0.0]), np.array([359.5, 0.5]), sigma_b=1.0,
                sigma_o=1.0, L=1e7, device=torch.device("cpu"), flat_grid=True)
    d = oi._dist_matrix(oi.grid_coords, oi.grid_coords)
    assert d[0, 1] == pytest.approx(6371000.0 * np.deg2rad(1.0), rel=1e-6)


def test_observation_operator_picks_the_nearest_node(oi_cls):
    """H привязывает наблюдение к ближайшему узлу, и ровно к одному."""
    import torch
    lats = np.array([50.0, 55.0, 60.0])
    lons = np.array([90.0, 90.0, 90.0])
    oi = oi_cls(lats, lons, sigma_b=1.0, sigma_o=1.0, L=1e5,
                device=torch.device("cpu"), flat_grid=True)
    H = oi._build_H(np.array([[54.9, 90.0]])).numpy()
    assert H.shape == (1, 3)
    assert H.sum() == pytest.approx(1.0)
    assert H[0, 1] == pytest.approx(1.0), "выбран не ближайший узел"
