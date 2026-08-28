"""Построение графа: порядок узлов сетки и поиск соседей по радиусу.

Порядок узлов здесь обязан совпадать с тем, в каком их выпрямляет загрузчик.
Разойдись он — и каждый узел графа получит данные чужой точки: обучение пойдёт,
ошибка будет считаться, модель выучит географию наизнанку и никто не заметит.
Поэтому порядок сверяется напрямую, а не через комментарии в коде.
"""
import numpy as np
import scipy.spatial

from src.mesh.grid_mesh_connectivity import (
    _grid_lat_lon_to_coordinates,
    radius_query_indices,
)


def unit_positions(lats, lons):
    """Положения узлов, посчитанные независимо от проверяемого кода."""
    LO, LA = np.meshgrid(lons, lats)          # (n_lat, n_lon)
    phi = np.deg2rad(LO.reshape(-1))
    theta = np.deg2rad(90 - LA.reshape(-1))
    return np.stack([np.cos(phi) * np.sin(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(theta)], axis=-1)


# --- порядок узлов -----------------------------------------------------------

def test_node_order_is_latitude_major():
    """Узел k — это широта k // n_lon и долгота k % n_lon.

    Ровно этот порядок даёт загрузчик, выпрямляя сетку как (lat, lon). Здесь он
    проверяется на самих координатах, а не на договорённости.
    """
    lats = np.array([50.0, 55.0, 60.0])
    lons = np.array([80.0, 85.0, 90.0, 95.0])
    got = _grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    n_lon = len(lons)
    for k in range(len(lats) * len(lons)):
        want, _, _ = (unit_positions(np.array([lats[k // n_lon]]),
                                     np.array([lons[k % n_lon]])), None, None)
        assert np.allclose(got[k], want[0], atol=1e-12), f"узел {k} не на своём месте"


def test_regular_and_flat_modes_agree():
    """Плоский режим даёт то же, что обычный, если координаты спарить вручную."""
    lats = np.array([50.0, 55.0, 60.0])
    lons = np.array([80.0, 90.0])
    regular = _grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    LO, LA = np.meshgrid(lons, lats)
    flat = _grid_lat_lon_to_coordinates(LA.reshape(-1), LO.reshape(-1), flat=True)
    assert np.allclose(regular, flat, atol=1e-12)


def test_grid_positions_are_on_the_unit_sphere():
    lats = np.linspace(-89.0, 89.0, 17)
    lons = np.linspace(0.0, 350.0, 36)
    pos = _grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    assert np.allclose(np.linalg.norm(pos, axis=1), 1.0, atol=1e-12)


def test_the_pole_maps_to_the_axis():
    pos = _grid_lat_lon_to_coordinates(np.array([90.0]), np.array([0.0, 123.0]))
    pos = pos.reshape(-1, 3)
    assert np.allclose(pos[:, 2], 1.0, atol=1e-12)
    assert np.allclose(pos[:, :2], 0.0, atol=1e-12)


# --- поиск соседей по радиусу ------------------------------------------------

class FakeMesh:
    """Меш из заданных вершин — так проверяется сам поиск, без икосаэдра."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)


def test_every_returned_edge_is_within_the_radius():
    lats = np.linspace(-60.0, 60.0, 7)
    lons = np.linspace(0.0, 300.0, 7)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mesh = FakeMesh(v)
    r = 0.5
    gi, mi = radius_query_indices(grid_latitude=lats, grid_longitude=lons,
                                  mesh=mesh, radius=r)
    pos = _grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    d = np.linalg.norm(pos[gi] - v[mi], axis=1)
    assert np.all(d <= r + 1e-12), "вернулось ребро длиннее радиуса"


def test_no_pair_within_the_radius_is_missed():
    """Обратная сторона: всё, что в радиусе, обязано попасть в рёбра."""
    lats = np.linspace(-60.0, 60.0, 5)
    lons = np.linspace(0.0, 288.0, 5)
    rng = np.random.default_rng(1)
    v = rng.normal(size=(40, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mesh = FakeMesh(v)
    r = 0.6
    gi, mi = radius_query_indices(grid_latitude=lats, grid_longitude=lons,
                                  mesh=mesh, radius=r)
    got = set(zip(gi.tolist(), mi.tolist()))

    pos = _grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    d = scipy.spatial.distance.cdist(pos, v)
    want = set(map(tuple, np.argwhere(d <= r).tolist()))
    assert got == want


def test_indices_stay_in_range():
    lats = np.linspace(-45.0, 45.0, 4)
    lons = np.linspace(0.0, 270.0, 4)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(30, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    gi, mi = radius_query_indices(grid_latitude=lats, grid_longitude=lons,
                                  mesh=FakeMesh(v), radius=0.7)
    assert gi.min() >= 0 and gi.max() < len(lats) * len(lons)
    assert mi.min() >= 0 and mi.max() < len(v)


def test_too_small_a_radius_leaves_nodes_without_edges():
    """Граница применимости, записанная тестом.

    Узел сетки, у которого в радиусе не нашлось ни одной вершины меша, просто не
    получает рёбер: ошибки не будет, но его данные в модель не попадут вовсе.
    Радиус задаётся как доля наибольшего ребра меша, и при неудачном множителе
    так может выйти. Отсюда проверка на изолированные узлы в create_graphs.
    """
    lats = np.array([0.0, 30.0])
    lons = np.array([0.0, 180.0])
    mesh = FakeMesh([[1.0, 0.0, 0.0]])        # единственная вершина
    gi, _ = radius_query_indices(grid_latitude=lats, grid_longitude=lons,
                                 mesh=mesh, radius=0.1)
    covered = set(gi.tolist())
    assert len(covered) < len(lats) * len(lons), "ожидались изолированные узлы"
