"""Взвешенная целевая функция: веса широты, маски каналов и вес области.

Требует настоящего torch, поэтому здесь пропускается, а гоняется на виртуалке.

Почему это важно проверять. Вес области — самый крупный рычаг статьи: он дал
5,4 % против всех прочих правок в доли процента. Заявлено, что вес w у узлов
вставки при единице у остальных отдаёт области долю w·n / (w·n + (N−n)) целевой
функции. Если бы вместо взвешенного среднего получалась взвешенная СУММА без
деления на сумму весов, масштаб градиента менялся бы вместе с весом, и мы
сравнивали бы не веса области, а скорости обучения.
"""
import numpy as np
import pytest
from conftest import needs_torch

pytestmark = needs_torch


@pytest.fixture
def tr():
    import torch
    return torch


@pytest.fixture
def loss_mod():
    from src.train import (
        build_boundary_mask,
        combine_spatial_masks,
        get_lat_weights,
        weighted_mse_loss,
    )
    return dict(loss=weighted_mse_loss, lat=get_lat_weights,
                boundary=build_boundary_mask, combine=combine_spatial_masks)


# --- взвешенное среднее, а не сумма ------------------------------------------

def test_unweighted_loss_is_plain_mse(tr, loss_mod):
    p = tr.randn(2, 7, 3)
    t = tr.randn(2, 7, 3)
    got = loss_mod["loss"](p, t)
    assert got.item() == pytest.approx(((p - t) ** 2).mean().item(), rel=1e-6)


def test_uniform_weights_do_not_change_the_value(tr, loss_mod):
    """Умножение всех весов на константу не меняет ответ.

    Это и означает, что считается среднее, а не сумма: иначе вес области менял
    бы масштаб градиента, и опыты с разным весом были бы несравнимы.
    """
    p, t = tr.randn(2, 9, 4), tr.randn(2, 9, 4)
    plain = loss_mod["loss"](p, t).item()
    scaled = loss_mod["loss"](p, t, spatial_mask=tr.full((1, 9, 1), 7.0)).item()
    assert scaled == pytest.approx(plain, rel=1e-6)


def test_channel_mask_removes_a_channel_entirely(tr, loss_mod):
    """Нулевой вес канала выкидывает его из ответа целиком.

    Так выключаются статические каналы: рельеф и маска суши не предсказываются,
    и их невязка не должна ни влиять на ответ, ни давать градиент.
    """
    p, t = tr.zeros(1, 5, 3), tr.zeros(1, 5, 3)
    p[..., 2] = 100.0                      # огромная невязка в третьем канале
    mask = tr.tensor([1.0, 1.0, 0.0])
    assert loss_mod["loss"](p, t, channel_mask=mask).item() == pytest.approx(0.0)
    assert loss_mod["loss"](p, t).item() > 100.0


def test_channel_mask_gives_no_gradient_to_masked_channels(tr, loss_mod):
    p = tr.zeros(1, 5, 3, requires_grad=True)
    t = tr.ones(1, 5, 3)
    loss_mod["loss"](p, t, channel_mask=tr.tensor([1.0, 0.0, 1.0])).backward()
    assert tr.allclose(p.grad[..., 1], tr.zeros(1)), "в выключенный канал течёт градиент"
    assert not tr.allclose(p.grad[..., 0], tr.zeros(1))


# --- вес области -------------------------------------------------------------

def test_roi_weight_gives_the_declared_share(tr, loss_mod):
    """Доля области в целевой функции совпадает с объявленной формулой.

    Берём поле, где ошибка есть ТОЛЬКО во вставке, и смотрим, какую часть
    ответа она составляет. Это и есть доля области.
    """
    n_total, n_roi, w = 1000, 19, 10.0
    roi = tr.zeros(1, n_total, 1)
    roi[0, :n_roi, 0] = 1.0
    mask = 1.0 + (w - 1.0) * roi

    p, t = tr.zeros(1, n_total, 1), tr.zeros(1, n_total, 1)
    p[0, :n_roi, 0] = 1.0                  # единичная невязка только во вставке
    got = loss_mod["loss"](p, t, spatial_mask=mask).item()
    expected = w * n_roi / (w * n_roi + (n_total - n_roi))
    assert got == pytest.approx(expected, rel=1e-6)


def test_roi_weight_one_is_the_same_as_no_weighting(tr, loss_mod):
    n_total, n_roi = 200, 20
    roi = tr.zeros(1, n_total, 1)
    roi[0, :n_roi, 0] = 1.0
    mask = 1.0 + (1.0 - 1.0) * roi         # вес 1 — вырожденный случай
    p, t = tr.randn(1, n_total, 3), tr.randn(1, n_total, 3)
    assert loss_mod["loss"](p, t, spatial_mask=mask).item() == pytest.approx(
        loss_mod["loss"](p, t).item(), rel=1e-6)


def test_larger_roi_weight_moves_the_answer_toward_the_roi(tr, loss_mod):
    """Чем больше вес, тем сильнее ответ определяется ошибкой во вставке."""
    n_total, n_roi = 500, 25
    roi = tr.zeros(1, n_total, 1)
    roi[0, :n_roi, 0] = 1.0
    p, t = tr.zeros(1, n_total, 1), tr.zeros(1, n_total, 1)
    p[0, :n_roi, 0] = 1.0
    shares = [loss_mod["loss"](p, t, spatial_mask=1.0 + (w - 1.0) * roi).item()
              for w in (1.0, 10.0, 30.0, 100.0)]
    assert shares == sorted(shares), "рост веса не увеличивает долю области"


# --- веса широты -------------------------------------------------------------

def test_latitude_weights_average_to_one(tr, loss_mod):
    w = loss_mod["lat"](lat_dim=32, lon_dim=64, device=tr.device("cpu"))
    assert w.mean().item() == pytest.approx(1.0, rel=1e-5)


def test_latitude_weights_favour_the_equator(tr, loss_mod):
    """У полюсов вес меньше: там ячейки сетки мельче по площади."""
    w = loss_mod["lat"](lat_dim=9, lon_dim=4,
                        device=tr.device("cpu")).view(9, 4)[:, 0]
    assert w[4].item() > w[0].item() and w[4].item() > w[-1].item()


def test_latitude_weights_follow_the_given_axis(tr, loss_mod):
    """Для региональной сетки берётся её собственная широтная ось.

    linspace(-90, 90) дал бы полосе 50-60° широты от полюса до полюса, и веса
    оказались бы бессмысленными.
    """
    lats = np.linspace(50.0, 60.0, 5)
    w = loss_mod["lat"](lat_dim=5, lon_dim=1, device=tr.device("cpu"),
                        lats_axis=lats).view(-1).numpy()
    ref = np.cos(np.deg2rad(lats))
    assert np.allclose(w / w[0], ref / ref[0], atol=1e-5)


def test_flat_grid_weights_use_per_node_latitudes(tr, loss_mod):
    lats = np.array([0.0, 60.0, 60.0, 89.0], dtype=np.float32)
    w = loss_mod["lat"](lat_dim=0, lon_dim=0, device=tr.device("cpu"),
                        flat_lats=lats).view(-1).numpy()
    assert w[0] > w[1] > w[3]
    assert w[1] == pytest.approx(w[2])


# --- буферная маска ----------------------------------------------------------

def test_boundary_mask_zeroes_the_edges(tr, loss_mod):
    n_lon, n_lat, width = 10, 6, 2
    m = loss_mod["boundary"](n_lon, n_lat, width, tr.device("cpu")).view(n_lat, n_lon)
    assert m[:width].sum().item() == 0 and m[-width:].sum().item() == 0
    assert m[:, :width].sum().item() == 0 and m[:, -width:].sum().item() == 0
    assert m[width:-width, width:-width].min().item() == 1.0


def test_masks_combine_by_multiplication(tr, loss_mod):
    a = tr.tensor([[[1.0], [0.0], [1.0]]])
    b = tr.tensor([[[2.0], [2.0], [0.0]]])
    got = loss_mod["combine"](a, b, None)
    assert got.view(-1).tolist() == [2.0, 0.0, 0.0]


def test_combining_nothing_returns_nothing(tr, loss_mod):
    assert loss_mod["combine"](None, None) is None
