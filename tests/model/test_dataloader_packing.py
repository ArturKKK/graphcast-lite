"""Загрузчик: порядок узлов, порядок каналов и деление выборки.

Три инварианта, каждый из которых при поломке НЕ ПАДАЕТ, а тихо портит всё
обучение:

  • порядок выпрямления сетки обязан совпадать с порядком узлов графа. Разойдись
    он — и каждый узел получит данные чужой точки. Обучение пойдёт, ошибка будет
    считаться, модель выучит географию наизнанку.
  • кадры и каналы упакованы в один вектор. Перепутай порядок — и модель увидит
    вместо «два срока по 33 канала» нечто с перемешанными сроками.
  • деление на обучение и контроль хронологическое. Заедь оно — и контроль
    окажется внутри обучения, а все числа статьи станут завышенными.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.dataloader_chunked import TimeseriesChunkDataset

N_LON, N_LAT, N_FEAT = 8, 5, 3


def make_grid_dataset(tmp_path: Path, n_time=40) -> Path:
    """Сетка, где значение узла кодирует его собственные координаты и срок.

    Канал 0 — номер срока, канал 1 — индекс долготы, канал 2 — индекс широты.
    По любому попавшему в выборку числу видно, откуда оно взято.
    """
    d = tmp_path / "grid"
    d.mkdir()
    arr = np.zeros((n_time, N_LON, N_LAT, N_FEAT), dtype=np.float16)
    for t in range(n_time):
        for i in range(N_LON):
            for j in range(N_LAT):
                arr[t, i, j] = (t, i, j)
    arr.tofile(d / "data.npy")
    (d / "dataset_info.json").write_text(json.dumps({
        "n_time": n_time, "n_lon": N_LON, "n_lat": N_LAT, "n_feat": N_FEAT,
        "flat": False}))
    np.savez(d / "scalers.npz", mean=np.zeros(N_FEAT, np.float32),
             std=np.ones(N_FEAT, np.float32))
    return d


def make_flat_dataset(tmp_path: Path, n_time=40, n_nodes=11) -> Path:
    d = tmp_path / "flat"
    d.mkdir()
    arr = np.zeros((n_time, n_nodes, N_FEAT), dtype=np.float16)
    for t in range(n_time):
        for n in range(n_nodes):
            arr[t, n] = (t, n, 0)
    arr.tofile(d / "data.npy")
    (d / "dataset_info.json").write_text(json.dumps({
        "n_time": n_time, "n_nodes": n_nodes, "n_feat": N_FEAT, "flat": True}))
    np.savez(d / "scalers.npz", mean=np.zeros(N_FEAT, np.float32),
             std=np.ones(N_FEAT, np.float32))
    return d


def as_array(x):
    return np.asarray(x)


# --- порядок узлов -----------------------------------------------------------

def test_grid_is_flattened_in_the_same_order_as_the_graph_builds_nodes(tmp_path):
    """Узел k — это широта k // n_lon и долгота k % n_lon.

    Ровно так же граф строит узлы: np.meshgrid(lons, lats).reshape(-1) при
    indexing='xy' даёт массив (n_lat, n_lon), то есть широта меняется медленно,
    долгота быстро. Здесь это сверяется с самим meshgrid, а не с рассуждением.
    """
    d = make_grid_dataset(tmp_path)
    ds = TimeseriesChunkDataset(str(d), obs_window=1, pred_steps=1, split="all")
    X, _ = ds[0]
    X = as_array(X).reshape(N_LAT * N_LON, N_FEAT)

    lons, lats = np.arange(N_LON), np.arange(N_LAT)
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    assert np.allclose(X[:, 1], mesh_lon.reshape(-1)), "долгота узлов не совпала с графом"
    assert np.allclose(X[:, 2], mesh_lat.reshape(-1)), "широта узлов не совпала с графом"


def test_flat_grid_keeps_node_order_as_is(tmp_path):
    d = make_flat_dataset(tmp_path)
    ds = TimeseriesChunkDataset(str(d), obs_window=1, pred_steps=1, split="all")
    X, _ = as_array(ds[0][0]), None
    X = X.reshape(-1, N_FEAT)
    assert np.allclose(X[:, 1], np.arange(X.shape[0]))


# --- порядок кадров и каналов ------------------------------------------------

@pytest.mark.parametrize("maker", ["grid", "flat"])
def test_input_frames_are_packed_step_major(tmp_path, maker):
    """X это [узел, срок, канал], а не [узел, канал, срок].

    train_epoch разбирает вход как X.view(N, G, obs, C). Разойдись упаковка с
    разбором — модель получила бы кашу, не заметив этого ничем.
    """
    d = make_grid_dataset(tmp_path) if maker == "grid" else make_flat_dataset(tmp_path)
    obs = 3
    ds = TimeseriesChunkDataset(str(d), obs_window=obs, pred_steps=1, split="all")
    X, _ = ds[0]
    X = as_array(X)
    n_nodes = X.shape[0]
    frames = X.reshape(n_nodes, obs, N_FEAT)
    for k in range(obs):
        assert np.allclose(frames[:, k, 0], float(k)), (
            f"кадр {k} лежит не на своём месте — упаковка не по срокам")


def test_targets_are_packed_step_major(tmp_path):
    d = make_grid_dataset(tmp_path)
    obs, pred = 2, 4
    ds = TimeseriesChunkDataset(str(d), obs_window=obs, pred_steps=pred, split="all")
    _, Y = ds[0]
    Y = as_array(Y)
    steps = Y.reshape(Y.shape[0], pred, N_FEAT)
    for k in range(pred):
        assert np.allclose(steps[:, k, 0], float(obs + k)), (
            f"цель на шаг {k} взята не из того срока")


def test_normalisation_uses_the_stored_scalers(tmp_path):
    d = make_grid_dataset(tmp_path)
    np.savez(d / "scalers.npz", mean=np.array([10.0, 0.0, 0.0], np.float32),
             std=np.array([2.0, 1.0, 1.0], np.float32))
    ds = TimeseriesChunkDataset(str(d), obs_window=1, pred_steps=1, split="all")
    X = as_array(ds[0][0]).reshape(-1, N_FEAT)
    assert np.allclose(X[:, 0], (0.0 - 10.0) / 2.0)


# --- шаг по времени ----------------------------------------------------------

def test_time_stride_moves_the_target_further(tmp_path):
    """При шаге 4 цель отстоит от входа на четыре срока, а не на один."""
    d = make_grid_dataset(tmp_path)
    ds = TimeseriesChunkDataset(str(d), obs_window=2, pred_steps=1, split="all",
                                time_stride=4)
    X, Y = as_array(ds[0][0]), as_array(ds[0][1])
    last_in = X.reshape(-1, 2, N_FEAT)[0, -1, 0]
    first_out = Y.reshape(-1, 1, N_FEAT)[0, 0, 0]
    assert first_out - last_in == pytest.approx(4.0)


def test_input_stride_can_differ_from_target_stride(tmp_path):
    """Развязка шагов: вход частый, цель дальняя.

    Нужна для прямого прогноза: у суточной модели вход шёл через 24 ч, и по двум
    таким кадрам почти не видно, куда движутся системы.
    """
    d = make_grid_dataset(tmp_path)
    ds = TimeseriesChunkDataset(str(d), obs_window=2, pred_steps=1, split="all",
                                time_stride=4, obs_stride=1)
    X, Y = as_array(ds[0][0]), as_array(ds[0][1])
    frames = X.reshape(-1, 2, N_FEAT)[0, :, 0]
    assert frames[1] - frames[0] == pytest.approx(1.0), "вход взят не с частым шагом"
    assert as_array(Y).reshape(-1, 1, N_FEAT)[0, 0, 0] - frames[1] == pytest.approx(4.0)


# --- деление выборки ---------------------------------------------------------

def _times(ds):
    """Сроки первого кадра каждого примера — по ним видно, какие взяты."""
    return np.array([as_array(ds[i][0]).reshape(-1, ds.obs_window, N_FEAT)[0, 0, 0]
                     for i in range(len(ds))])


def test_split_is_chronological_and_disjoint(tmp_path):
    """Контроль идёт строго ПОСЛЕ обучения и не пересекается с ним.

    Заедь это — и числа статьи окажутся завышенными, а понять по ним ничего
    будет нельзя: обучающая ошибка выглядит как проверочная.
    """
    d = make_grid_dataset(tmp_path, n_time=60)
    kw = dict(obs_window=2, pred_steps=1)
    tr = _times(TimeseriesChunkDataset(str(d), split="train", **kw))
    va = _times(TimeseriesChunkDataset(str(d), split="val", **kw))
    te = _times(TimeseriesChunkDataset(str(d), split="test_only", **kw))

    assert len(tr) and len(va) and len(te)
    assert tr.max() < va.min(), "обучение заходит в отбор"
    assert va.max() < te.min(), "отбор заходит в проверку"
    assert not (set(tr) & set(va)) and not (set(va) & set(te))


def test_test_fraction_is_respected(tmp_path):
    d = make_grid_dataset(tmp_path, n_time=100)
    kw = dict(obs_window=2, pred_steps=1)
    all_n = len(TimeseriesChunkDataset(str(d), split="all", **kw))
    tr = len(TimeseriesChunkDataset(str(d), split="train", test_fraction=0.2, **kw))
    assert tr == pytest.approx(all_n * 0.8, abs=1)


def test_val_and_test_are_halves_of_the_held_out_part(tmp_path):
    d = make_grid_dataset(tmp_path, n_time=100)
    kw = dict(obs_window=2, pred_steps=1)
    va = len(TimeseriesChunkDataset(str(d), split="val", **kw))
    te = len(TimeseriesChunkDataset(str(d), split="test_only", **kw))
    full = len(TimeseriesChunkDataset(str(d), split="test", **kw))
    assert va + te == full
    assert abs(va - te) <= 1
