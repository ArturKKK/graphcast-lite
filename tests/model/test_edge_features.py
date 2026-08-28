"""Признаки рёбер меша: длина и направление.

Требует torch. Эти четыре числа на ребро — всё, что сеть знает о геометрии
связи между узлами. Ошибка здесь не роняет обучение: модель просто выучит
неверную геометрию и будет работать хуже, чем могла бы.

Режим legacy — тот, на котором обучены все модели статьи; менять его умолчание
нельзя, поедут опубликованные числа. Поэтому он проверяется отдельно и жёстко.
"""
import numpy as np
import pytest
from conftest import needs_torch

pytestmark = needs_torch


@pytest.fixture
def feats():
    from src.create_graphs import _compute_mesh_edge_features
    return _compute_mesh_edge_features


def triangle():
    """Три узла и рёбра между ними — минимальный содержательный случай."""
    lats = np.array([0.0, 10.0, 0.0])
    lons = np.array([0.0, 0.0, 30.0])
    edges = np.array([[0, 0, 1], [1, 2, 2]])     # 0->1, 0->2, 1->2
    return lats, lons, edges


def test_four_features_per_edge(feats):
    lats, lons, edges = triangle()
    for mode in ("legacy", "unit_log"):
        got = feats(lats, lons, edges, mode=mode)
        assert tuple(got.shape) == (3, 4), mode


def test_legacy_normalises_by_the_longest_edge(feats):
    """В legacy самое длинное ребро получает длину ровно 1."""
    lats, lons, edges = triangle()
    got = feats(lats, lons, edges, mode="legacy").numpy()
    assert got[:, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert np.all(got[:, 0] > 0)


def test_legacy_lengths_keep_their_ratios(feats):
    """Нормировка общая, поэтому отношения длин сохраняются."""
    lats, lons, edges = triangle()
    got = feats(lats, lons, edges, mode="legacy").numpy()
    xyz = np.stack([np.cos(np.deg2rad(lats)) * np.cos(np.deg2rad(lons)),
                    np.cos(np.deg2rad(lats)) * np.sin(np.deg2rad(lons)),
                    np.sin(np.deg2rad(lats))], axis=-1)
    true = np.linalg.norm(xyz[edges[0]] - xyz[edges[1]], axis=1)
    assert np.allclose(got[:, 0] / got[0, 0], true / true[0], atol=1e-5)


def test_unit_log_direction_is_a_unit_vector(feats):
    lats, lons, edges = triangle()
    got = feats(lats, lons, edges, mode="unit_log").numpy()
    assert np.allclose(np.linalg.norm(got[:, 1:], axis=1), 1.0, atol=1e-5)


def test_unit_log_length_fills_the_unit_interval(feats):
    """Логарифм длины приведён к [0, 1], и края достигаются.

    Ради этого режим и вводился: в legacy при уровнях меша [0..6] почти все
    рёбра спрессованы в полоску шириной 0,06 у нуля, и кодировщику приходится
    различать содержательные масштабы по почти одинаковому входу.
    """
    rng = np.random.default_rng(0)
    n = 60
    lats = rng.uniform(-70, 70, n)
    lons = rng.uniform(0, 360, n)
    edges = np.stack([rng.integers(0, n, 200), rng.integers(0, n, 200)])
    edges = edges[:, edges[0] != edges[1]]
    got = feats(lats, lons, edges, mode="unit_log").numpy()
    assert got[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert got[:, 0].max() == pytest.approx(1.0, abs=1e-6)


def test_unit_log_spreads_scales_better_than_legacy(feats):
    """Прямая проверка довода, ради которого режим введён.

    Берём рёбра пяти масштабов, различающихся вдвое, как уровни дробления меша.
    В legacy короткие рёбра сбиваются к нулю, в unit_log ложатся равномерно.
    """
    lats, lons, senders, receivers = [], [], [], []
    for k in range(5):
        d = 40.0 / (2 ** k)
        i = len(lats)
        lats += [0.0, d]
        lons += [0.0, 0.0]
        senders.append(i); receivers.append(i + 1)
    edges = np.array([senders, receivers])
    lats, lons = np.array(lats), np.array(lons)
    leg = feats(lats, lons, edges, mode="legacy").numpy()[:, 0]
    ulog = feats(lats, lons, edges, mode="unit_log").numpy()[:, 0]
    # у legacy самый короткий масштаб почти неотличим от нуля
    assert leg.min() < 0.08
    # у unit_log уровни встают равномерно: соседние отстоят примерно одинаково
    steps = np.diff(np.sort(ulog))
    assert steps.max() / steps.min() < 1.6, "уровни легли неравномерно"


def test_direction_distinguishes_bearing(feats):
    """Север и восток от одного узла дают разные направления."""
    lats = np.array([0.0, 5.0, 0.0])
    lons = np.array([0.0, 0.0, 5.0])
    edges = np.array([[1, 2], [0, 0]])           # оба ребра приходят в узел 0
    got = feats(lats, lons, edges, mode="unit_log").numpy()
    assert np.linalg.norm(got[0, 1:] - got[1, 1:]) > 0.5


def test_zero_length_edge_does_not_produce_nan(feats):
    """Петля не даёт NaN ни в одном режиме.

    Граница применимости: в unit_log петля вдобавок сминает шкалу — её нулевая
    длина уходит в логарифм как -27,6 и становится нижним краем нормировки, так
    что все настоящие рёбра сжимаются к единице. Петель в меше нет, но знать об
    этом стоит.
    """
    lats = np.array([10.0, 20.0])
    lons = np.array([30.0, 40.0])
    edges = np.array([[0, 0], [0, 1]])           # петля и обычное ребро
    for mode in ("legacy", "unit_log"):
        got = feats(lats, lons, edges, mode=mode).numpy()
        assert np.all(np.isfinite(got)), mode
