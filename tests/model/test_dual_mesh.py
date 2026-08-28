"""Двойной меш: региональное сгущение и связи с глобальным мешом.

Требует torch.

Здесь проверяется прежде всего НЕЯВНОЕ ДОПУЩЕНИЕ, на котором держится
региональный меш: вершины грубого уровня считаются префиксом мелкого, и
`create_regional_mesh` отбирает «новые» вершины просто по номеру, начиная с
длины уровня 6. Допущение верно, потому что дробление дописывает потомков в
конец списка, — но нигде не записано. Измени порядок в построении меша, и
регион молча наберёт не те вершины: часть узлов сгущения окажется дублями
глобальных, часть настоящих потеряется, а ошибки не будет.
"""
import numpy as np
import pytest

from conftest import needs_torch

pytestmark = needs_torch

ROI = (50.0, 60.0, 83.0, 98.0)          # Красноярский край, как в статье


@pytest.fixture(scope="module")
def hierarchy():
    from src.mesh.create_mesh import get_hierarchy_of_triangular_meshes_for_sphere
    return get_hierarchy_of_triangular_meshes_for_sphere(splits=7)


@pytest.fixture(scope="module")
def dm():
    from src.dual_mesh import build_cross_edges, create_regional_mesh
    return dict(regional=create_regional_mesh, cross=build_cross_edges)


# --- допущение о префиксе ----------------------------------------------------

def test_coarse_vertices_are_a_prefix_of_the_fine_ones(hierarchy):
    """Вершины грубого уровня стоят в начале списка мелкого, в том же порядке.

    На этом стоит отбор «новых» вершин по номеру в create_regional_mesh.
    """
    for k in range(len(hierarchy) - 1):
        coarse, fine = hierarchy[k].vertices, hierarchy[k + 1].vertices
        assert len(fine) > len(coarse)
        assert np.allclose(fine[:len(coarse)], coarse, atol=1e-12), (
            f"уровень {k} не является префиксом уровня {k + 1}")


def test_each_split_roughly_quadruples_the_vertices(hierarchy):
    """Каждое дробление делит грань на четыре — число вершин растёт вчетверо."""
    for k in range(1, len(hierarchy) - 1):
        a, b = len(hierarchy[k].vertices), len(hierarchy[k + 1].vertices)
        assert 3.8 < (b - 2) / (a - 2) < 4.2, f"уровень {k}: {a} -> {b}"


def test_all_mesh_vertices_are_on_the_unit_sphere(hierarchy):
    for m in hierarchy:
        assert np.allclose(np.linalg.norm(m.vertices, axis=1), 1.0, atol=1e-9)


def test_faces_reference_existing_vertices(hierarchy):
    for m in hierarchy:
        assert m.faces.min() >= 0
        assert m.faces.max() < len(m.vertices)


# --- региональный меш --------------------------------------------------------

@pytest.fixture(scope="module")
def regional(dm):
    return dm["regional"](ROI, level=7, buffer_deg=2.0)


def test_regional_vertices_lie_inside_the_region(regional):
    mesh, lats, lons = regional
    lat_min, lat_max, lon_min, lon_max = ROI
    b = 2.0
    assert lats.min() >= lat_min - b - 1e-4 and lats.max() <= lat_max + b + 1e-4
    assert lons.min() >= lon_min - b - 1e-4 and lons.max() <= lon_max + b + 1e-4


def test_regional_mesh_is_not_empty_and_is_much_smaller_than_the_globe(regional, hierarchy):
    mesh, lats, _ = regional
    assert len(lats) > 100, "сгущение вышло пустым"
    assert len(lats) < 0.05 * len(hierarchy[7].vertices), "область не локальна"


def test_regional_vertices_are_new_ones_only(regional, hierarchy):
    """В сгущение не попадает ни одна вершина глобального меша.

    Иначе узел был бы представлен дважды — и в глобальном меше, и в
    региональном, — а сообщения по нему пошли бы двумя путями.
    """
    mesh, lats, lons = regional
    g = hierarchy[6].vertices
    from scipy.spatial import cKDTree
    d, _ = cKDTree(g).query(mesh.vertices, k=1)
    assert d.min() > 1e-9, "в сгущении есть вершина глобального меша"


def test_regional_faces_are_valid(regional):
    mesh, lats, _ = regional
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < len(lats)
    assert mesh.faces.shape[1] == 3


def test_empty_region_is_refused_loudly(dm):
    """Пустая область — понятный отказ, а не молчаливый пустой меш."""
    with pytest.raises(ValueError, match="No regional mesh vertices"):
        dm["regional"]((89.99, 89.995, 0.0, 0.001), level=7, buffer_deg=0.0)


def test_region_crossing_the_meridian_is_not_supported(dm):
    """Граница применимости, записанная тестом.

    Долготы приведены к [0, 360), а отбор идёт простым сравнением с границами.
    Область через нулевой меридиан (например 350°..10°) даёт пустое пересечение
    и отказ. Для Красноярского края (83°..98°) это не важно, но знать надо: для
    области у меридиана функцию придётся править, а не просто позвать.
    """
    with pytest.raises(ValueError):
        dm["regional"]((50.0, 60.0, 350.0, 10.0), level=7, buffer_deg=1.0)


# --- связи между мешами ------------------------------------------------------

@pytest.fixture(scope="module")
def cross(dm, regional, hierarchy):
    from src.utils import get_mesh_lat_long
    g_lat, g_lon = get_mesh_lat_long(hierarchy[6])
    _, r_lat, r_lon = regional
    return dm["cross"](g_lat, g_lon, r_lat, r_lon, k=3), (g_lat, g_lon), (r_lat, r_lon)


def test_every_regional_node_gets_k_global_neighbours(cross, regional):
    (ei, ef), _, _ = cross
    _, r_lat, _ = regional
    n_reg, k = len(r_lat), 3
    assert ei.shape[1] == 2 * n_reg * k, "число рёбер не сходится"
    assert ef.shape == (2 * n_reg * k, 4)


def test_cross_edges_are_bidirectional(cross, regional):
    """Каждая связь есть в обе стороны: и от глобального узла, и к нему."""
    (ei, _), _, _ = cross
    _, r_lat, _ = regional
    half = ei.shape[1] // 2
    fwd = ei[:, :half].numpy()
    back = ei[:, half:].numpy()
    assert np.array_equal(fwd[0], back[1]), "обратные рёбра не совпали"
    assert np.array_equal(fwd[1], back[0])


def test_chosen_global_nodes_are_the_nearest(cross):
    """Выбраны действительно ближайшие глобальные узлы, а не какие попало."""
    (ei, _), (g_lat, g_lon), (r_lat, r_lon) = cross
    from scipy.spatial import cKDTree

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                         np.sin(la)], axis=-1)

    want = cKDTree(xyz(g_lat, g_lon)).query(xyz(r_lat, r_lon), k=3)[1]
    half = ei.shape[1] // 2
    got = ei[0, :half].numpy().reshape(-1, 3)
    assert np.array_equal(np.sort(got, axis=1), np.sort(want, axis=1))


def test_cross_edge_features_are_finite(cross):
    (_, ef), _, _ = cross
    import torch
    assert torch.isfinite(ef).all()


def test_single_neighbour_still_works(dm, regional, hierarchy):
    """k=1 не должно ломаться на форме массива соседей.

    scipy при k=1 возвращает одномерный массив вместо двумерного — классическая
    ловушка, из-за которой такой путь падает только в частном случае.
    """
    from src.utils import get_mesh_lat_long
    g_lat, g_lon = get_mesh_lat_long(hierarchy[6])
    _, r_lat, r_lon = regional
    ei, ef = dm["cross"](g_lat, g_lon, r_lat[:50], r_lon[:50], k=1)
    assert ei.shape[1] == 2 * 50
    assert ef.shape[0] == 2 * 50
