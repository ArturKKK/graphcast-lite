"""Свёрточный уточнитель и остаточная поправка по области.

Требует torch.

`WeatherUNet` работает в оперативном контуре, `ROIResidualModel` — в
мультимасштабной линии. Оба до 29.08.2026 не были покрыты вовсе.
"""
import numpy as np
import pytest
from conftest import needs_torch

pytestmark = needs_torch


# =====================  свёрточный уточнитель  ==============================

@pytest.fixture
def unet_cls():
    from src.unet.model import WeatherUNet
    return WeatherUNet


@pytest.mark.parametrize("h, w", [
    (64, 64),      # степень двойки
    (61, 41),      # НАША сетка: обе стороны нечётные
    (40, 24),
    (33, 17),      # нечётные и небольшие
])
def test_output_keeps_the_grid_size(unet_cls, h, w):
    """Выход того же размера, что вход, при любой стороне.

    Свёрточный уточнитель трижды уменьшает сетку вдвое и трижды возвращает.
    При нечётной стороне округление вниз теряет строку, и её приходится
    добивать отступом — иначе склейка с пропущенной связью не сойдётся по
    размеру. Наша сетка 61x41 нечётная по обеим сторонам, то есть этот путь у
    нас основной, а не запасной.
    """
    import torch
    m = unet_cls(in_channels=6, out_channels=3, base_filters=4).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 6, h, w))
    assert out.shape == (2, 3, h, w)


def test_channels_follow_the_arguments(unet_cls):
    import torch
    m = unet_cls(in_channels=38, out_channels=19, base_filters=4).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 38, 32, 32))
    assert out.shape[1] == 19


def test_gradient_reaches_the_first_layer(unet_cls):
    """Градиент доходит до входного блока: пропущенные связи не разорваны."""
    import torch
    m = unet_cls(in_channels=4, out_channels=2, base_filters=4)
    m(torch.randn(2, 4, 32, 32)).sum().backward()
    first = m.inc.net[0].weight
    assert first.grad is not None and first.grad.abs().sum() > 0


def test_too_small_a_grid_is_a_hard_failure(unet_cls):
    """Меньше восьми точек по стороне сеть не переваривает.

    Три уменьшения вдвое от стороны меньше восьми дают нулевой размер. Это
    граница применимости: падает громко, но знать о ней надо, потому что
    сообщение torch о ней невразумительное.
    """
    import torch
    m = unet_cls(in_channels=2, out_channels=1, base_filters=4).eval()
    # Именно RuntimeError: torch отказывается свернуть тензор нулевого размера.
    # Ловить голое Exception нельзя — так тест прошёл бы и на опечатке в вызове.
    with pytest.raises(RuntimeError), torch.no_grad():
        m(torch.randn(1, 2, 6, 6))


def test_eval_mode_is_deterministic_and_differs_from_train(unet_cls):
    """Нормировка по батчу молча меняет ответ, если забыть eval().

    В обучении она нормирует каждый батч его собственной статистикой, в
    проверке — накопленной. Забыть переключить режим при инференсе — классическая
    оперативная ошибка: прогноз меняется от того, сколько выборок в батче.
    """
    import torch
    m = unet_cls(in_channels=3, out_channels=2, base_filters=4)
    x = torch.randn(4, 3, 32, 32)
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.allclose(a, b), "в режиме проверки два прогона дали разное"
    m.train()
    with torch.no_grad():
        c = m(x)
    assert not torch.allclose(a, c, atol=1e-4), (
        "режимы обучения и проверки неразличимы — нормировка по батчу не работает")


def test_batch_size_does_not_change_the_forecast_in_eval(unet_cls):
    """В режиме проверки прогноз одной выборки не зависит от соседей по батчу."""
    import torch
    m = unet_cls(in_channels=3, out_channels=2, base_filters=4).eval()
    x = torch.randn(4, 3, 24, 24)
    with torch.no_grad():
        full = m(x)
        alone = m(x[:1])
    assert torch.allclose(full[:1], alone, atol=1e-5)


# =====================  остаточная поправка по области  =====================

@pytest.fixture
def roi_mod():
    from src.roi_residual import ROIResidualHead, build_roi_knn_graph
    return dict(graph=build_roi_knn_graph, head=ROIResidualHead)


def grid(n_lat=12, n_lon=16):
    lats = np.linspace(40.0, 70.0, n_lat)
    lons = np.linspace(70.0, 110.0, n_lon)
    LO, LA = np.meshgrid(lons, lats)
    return LA.reshape(-1), LO.reshape(-1)


def test_region_mask_selects_exactly_the_nodes_inside(roi_mod):
    lats, lons = grid()
    roi = (50.0, 60.0, 83.0, 98.0)
    mask, idx, ei, ef = roi_mod["graph"](lats, lons, roi, k=4)
    want = ((lats >= 50) & (lats <= 60) & (lons >= 83) & (lons <= 98))
    assert np.array_equal(mask, want)
    assert np.array_equal(idx, np.where(want)[0])


def test_each_region_node_gets_k_neighbours(roi_mod):
    lats, lons = grid()
    mask, idx, ei, ef = roi_mod["graph"](lats, lons, (50.0, 60.0, 83.0, 98.0), k=4)
    n = len(idx)
    assert ei.shape == (2, n * 4)
    assert ef.shape[0] == n * 4


def test_no_node_is_its_own_neighbour(roi_mod):
    """Петли исключены: ближайшим к узлу является он сам, и его надо отбросить."""
    lats, lons = grid()
    _, _, ei, _ = roi_mod["graph"](lats, lons, (50.0, 60.0, 83.0, 98.0), k=4)
    assert not (ei[0] == ei[1]).any(), "в графе области есть петли"


def test_edges_stay_inside_the_region(roi_mod):
    """Связи строятся только между узлами области — номера локальные."""
    lats, lons = grid()
    _, idx, ei, _ = roi_mod["graph"](lats, lons, (50.0, 60.0, 83.0, 98.0), k=4)
    assert ei.min() >= 0 and ei.max() < len(idx)


def test_neighbours_are_the_nearest_ones(roi_mod):
    """Соседями выбраны действительно ближайшие узлы области."""
    from scipy.spatial import cKDTree
    lats, lons = grid()
    roi = (50.0, 60.0, 83.0, 98.0)
    _, idx, ei, _ = roi_mod["graph"](lats, lons, roi, k=3)
    la, lo = np.radians(lats[idx]), np.radians(lons[idx])
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    want = cKDTree(xyz).query(xyz, k=4)[1][:, 1:]
    got = ei[0].numpy().reshape(len(idx), 3)
    assert np.array_equal(np.sort(got, 1), np.sort(want, 1))


def test_empty_region_is_refused(roi_mod):
    lats, lons = grid()
    with pytest.raises(ValueError, match="No grid points in ROI"):
        roi_mod["graph"](lats, lons, (0.0, 1.0, 0.0, 1.0), k=4)


def test_more_neighbours_than_nodes_is_survivable(roi_mod):
    """Соседей просят больше, чем узлов в области — берём сколько есть."""
    lats, lons = grid(n_lat=40, n_lon=40)
    _, idx, ei, _ = roi_mod["graph"](lats, lons, (50.0, 52.0, 90.0, 92.0), k=100)
    assert len(idx) >= 2
    assert ei.shape[1] == len(idx) * (len(idx) - 1)


def test_single_node_region_is_refused_clearly(roi_mod):
    """Область из одного узла — внятный отказ, а не ошибка numpy о пустом массиве.

    Графа соседей на одном узле не бывает: рёбер выходит ноль, и всё дальше
    разваливалось сообщением «zero-size array to reduction operation maximum»,
    которое уводит от настоящей причины.
    """
    lats, lons = grid(n_lat=5, n_lon=5)
    with pytest.raises(ValueError, match="узел сетки"):
        roi_mod["graph"](lats, lons, (39.9, 40.1, 69.9, 70.1), k=4)


def test_edge_features_survive_an_empty_edge_set():
    """Пустой набор рёбер даёт пустую таблицу признаков, а не отказ."""
    import numpy as np

    from src.create_graphs import _compute_mesh_edge_features
    got = _compute_mesh_edge_features(np.array([10.0, 20.0]), np.array([30.0, 40.0]),
                                      np.zeros((2, 0), dtype=int))
    assert tuple(got.shape) == (0, 4)


def test_residual_head_starts_from_almost_zero(roi_mod):
    """При создании поправка почти нулевая — модель стартует с глобального прогноза.

    Последний слой намеренно проинициализирован малыми весами и нулевым
    смещением. Замени инициализацию на обычную — и обучение начнётся с заметно
    испорченного прогноза, а первые эпохи уйдут на возврат к исходному.
    """
    import torch
    h = roi_mod["head"](input_dim=8, hidden_dim=16, output_dim=4).eval()
    with torch.no_grad():
        out = h(torch.randn(50, 5), torch.randn(50, 3))
    assert out.abs().max().item() < 0.1, f"поправка на старте не мала: {out.abs().max()}"


def test_residual_head_shapes(roi_mod):
    import torch
    h = roi_mod["head"](input_dim=7, hidden_dim=16, output_dim=3).eval()
    with torch.no_grad():
        out = h(torch.randn(11, 4), torch.randn(11, 3))
    assert out.shape == (11, 3)
