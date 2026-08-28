"""Преобразования координат и повороты в местную систему приёмника.

На этом стоят все признаки рёбер графа: и сетка→меш, и меш→меш, и меш→сетка.
Ошибка здесь не роняет обучение и никак себя не проявляет — модель просто
учится по неверной геометрии и работает хуже, чем могла бы, а объяснить это
нечем. Проверять такое надо инвариантами, а не глазами.
"""
import numpy as np
import pytest

from src.utils import (
    cartesian_to_spherical,
    get_bipartite_relative_position_in_receiver_local_coordinates,
    get_rotation_matrices_to_local_coordinates,
    lat_lon_deg_to_spherical,
    spherical_to_cartesian,
    spherical_to_lat_lon,
)


def to_xyz(lat, lon):
    phi, theta = lat_lon_deg_to_spherical(np.asarray(lat, dtype=float),
                                          np.asarray(lon, dtype=float))
    return np.stack(spherical_to_cartesian(phi, theta), axis=-1), phi, theta


# --- основные тождества ------------------------------------------------------

def test_known_points_land_where_expected():
    """Северный полюс, экватор на нулевом меридиане и на 90° в.д."""
    xyz, _, _ = to_xyz([90.0, 0.0, 0.0], [0.0, 0.0, 90.0])
    assert np.allclose(xyz[0], [0, 0, 1], atol=1e-12)      # полюс
    assert np.allclose(xyz[1], [1, 0, 0], atol=1e-12)      # 0°N 0°E
    assert np.allclose(xyz[2], [0, 1, 0], atol=1e-12)      # 0°N 90°E


def test_positions_are_on_the_unit_sphere():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-90, 90, 5000)
    lon = rng.uniform(0, 360, 5000)
    xyz, _, _ = to_xyz(lat, lon)
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-12)


def test_round_trip_returns_the_same_point():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-89.9, 89.9, 3000)
    lon = rng.uniform(0, 360, 3000)
    xyz, _, _ = to_xyz(lat, lon)
    phi2, theta2 = cartesian_to_spherical(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    lat2, lon2 = spherical_to_lat_lon(phi2, theta2)
    assert np.allclose(lat2, lat, atol=1e-9)
    assert np.allclose(np.mod(lon2 - lon + 180, 360) - 180, 0.0, atol=1e-9)


def test_chord_distance_matches_the_great_circle():
    """Расстояние по хорде согласовано с угловым: |a-b| = 2·sin(угол/2)."""
    a, _, _ = to_xyz([0.0], [0.0])
    b, _, _ = to_xyz([0.0], [90.0])
    assert np.linalg.norm(a - b) == pytest.approx(np.sqrt(2.0), abs=1e-12)


# --- повороты в местную систему ---------------------------------------------

@pytest.mark.parametrize("lat, lon", [(0.0, 0.0), (55.0, 93.0), (-33.0, 271.0),
                                      (89.0, 15.0), (-89.0, 300.0)])
def test_both_rotations_put_the_receiver_at_the_x_axis(lat, lon):
    """При обоих поворотах приёмник переезжает ровно в (1, 0, 0).

    Это и есть смысл местной системы: и широта, и долгота приёмника обнуляются.
    """
    _, phi, theta = to_xyz([lat], [lon])
    m = get_rotation_matrices_to_local_coordinates(phi, theta, True, True)
    pos, _, _ = to_xyz([lat], [lon])
    got = np.einsum("bji,bi->bj", m, pos)
    assert np.allclose(got[0], [1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("lat, lon", [(55.0, 93.0), (-33.0, 271.0), (10.0, 5.0)])
def test_longitude_only_rotation_zeroes_y_and_keeps_latitude(lat, lon):
    """Поворот только по долготе обнуляет y, но широту не трогает."""
    _, phi, theta = to_xyz([lat], [lon])
    m = get_rotation_matrices_to_local_coordinates(phi, theta, False, True)
    pos, _, _ = to_xyz([lat], [lon])
    got = np.einsum("bji,bi->bj", m, pos)[0]
    assert got[1] == pytest.approx(0.0, abs=1e-12)
    assert got[2] == pytest.approx(pos[0][2], abs=1e-12), "широта изменилась"


@pytest.mark.parametrize("lat, lon", [(55.0, 93.0), (-33.0, 271.0), (10.0, 5.0)])
def test_latitude_only_rotation_zeroes_z_and_keeps_longitude(lat, lon):
    """Поворот только по широте сажает приёмник на экватор, сохраняя долготу."""
    _, phi, theta = to_xyz([lat], [lon])
    m = get_rotation_matrices_to_local_coordinates(phi, theta, True, False)
    pos, _, _ = to_xyz([lat], [lon])
    got = np.einsum("bji,bi->bj", m, pos)[0]
    assert got[2] == pytest.approx(0.0, abs=1e-12), "приёмник не сел на экватор"
    got_lon = np.rad2deg(np.arctan2(got[1], got[0])) % 360
    assert (got_lon - lon % 360 + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


def test_no_rotation_requested_is_an_error():
    _, phi, theta = to_xyz([10.0], [10.0])
    with pytest.raises(ValueError):
        get_rotation_matrices_to_local_coordinates(phi, theta, False, False)


def test_rotation_matrices_are_orthonormal():
    rng = np.random.default_rng(2)
    lat = rng.uniform(-89, 89, 200)
    lon = rng.uniform(0, 360, 200)
    _, phi, theta = to_xyz(lat, lon)
    for rl, rlo in ((True, True), (True, False), (False, True)):
        m = get_rotation_matrices_to_local_coordinates(phi, theta, rl, rlo)
        prod = np.einsum("bij,bkj->bik", m, m)
        assert np.allclose(prod, np.eye(3)[None], atol=1e-10), (rl, rlo)
        assert np.allclose(np.linalg.det(m), 1.0, atol=1e-10), "не поворот, а отражение"


# --- признаки рёбер ---------------------------------------------------------

def rel_pos(lat_s, lon_s, lat_r, lon_r, rot_lat, rot_lon):
    _, s_phi, s_theta = to_xyz(lat_s, lon_s)
    _, r_phi, r_theta = to_xyz(lat_r, lon_r)
    n = len(np.atleast_1d(lat_s))
    return get_bipartite_relative_position_in_receiver_local_coordinates(
        s_phi, s_theta, np.arange(n), r_phi, r_theta, np.arange(n),
        rot_lat, rot_lon)


@pytest.mark.parametrize("rot", [(False, False), (True, True), (True, False),
                                 (False, True)])
def test_edge_length_is_preserved_by_the_rotation(rot):
    """Поворот — движение: длина ребра от него не меняется."""
    rng = np.random.default_rng(3)
    lat_s, lon_s = rng.uniform(-80, 80, 500), rng.uniform(0, 360, 500)
    lat_r, lon_r = lat_s + rng.uniform(-5, 5, 500), lon_s + rng.uniform(-5, 5, 500)
    plain = rel_pos(lat_s, lon_s, lat_r, lon_r, False, False)
    got = rel_pos(lat_s, lon_s, lat_r, lon_r, *rot)
    assert np.allclose(np.linalg.norm(got, axis=1), np.linalg.norm(plain, axis=1),
                       atol=1e-10)


@pytest.mark.parametrize("rot", [(False, False), (True, True), (True, False),
                                 (False, True)])
def test_self_edge_has_zero_relative_position(rot):
    got = rel_pos([55.0], [93.0], [55.0], [93.0], *rot)
    assert np.allclose(got, 0.0, atol=1e-12)


def test_local_coordinates_make_edges_translation_invariant():
    """Главное свойство местной системы, ради которого она и вводится.

    Отправитель на одном и том же удалении и в том же направлении от приёмника
    должен давать ОДИН И ТОТ ЖЕ вектор признаков, где бы приёмник ни стоял на
    сфере. Иначе сеть учит одну и ту же геометрическую связь заново для каждой
    широты, а на полюсах — ещё и в искажённом виде.

    Проверяем на паре «отправитель ровно на градус севернее приёмника» для
    приёмников от экватора до высоких широт.
    """
    lat_r = np.array([0.0, 20.0, 45.0, 60.0, 75.0])
    lon_r = np.array([0.0, 70.0, 150.0, 250.0, 330.0])
    got = rel_pos(lat_r + 1.0, lon_r, lat_r, lon_r, True, True)
    spread = got.max(axis=0) - got.min(axis=0)
    assert np.all(spread < 1e-9), (
        f"вектор ребра зависит от места приёмника: разброс {spread}")


def test_without_rotation_edges_are_not_translation_invariant():
    """А без поворота — зависит, и сильно. Это и есть цена отказа от местной системы."""
    lat_r = np.array([0.0, 20.0, 45.0, 60.0, 75.0])
    lon_r = np.array([0.0, 70.0, 150.0, 250.0, 330.0])
    got = rel_pos(lat_r + 1.0, lon_r, lat_r, lon_r, False, False)
    spread = got.max(axis=0) - got.min(axis=0)
    assert np.any(spread > 1e-3)


def test_bearing_is_distinguished():
    """Север и восток от одного приёмника дают разные векторы.

    Если бы поворот терял направление, все соседи выглядели бы одинаково, и
    модель не смогла бы отличить перенос с севера от переноса с востока.
    """
    north = rel_pos([56.0], [93.0], [55.0], [93.0], True, True)
    east = rel_pos([55.0], [94.0], [55.0], [93.0], True, True)
    assert np.linalg.norm(north - east) > 1e-3
