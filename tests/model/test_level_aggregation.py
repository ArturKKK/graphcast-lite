"""Раздельная агрегация по уровням меша в процессоре (level_aggregation, 01.10.2026).

Второе сгущение меша ухудшило все поля на 0,6–3 %, сильнее всего крупные
масштабы. Гипотеза: вершина усредняет все входящие сообщения разом, и много
коротких рёбер разбавляют длинные. Поправка добавляет к прежнему среднему
средние по каждому уровню с весами, которые стартуют с нуля. Здесь проверено:
рёбра размечены по уровням правильно, слой с нулевыми весами совпадает с
прежним, веса учатся, а модель грузит веса модели без поправки.
"""
import numpy as np
import pytest
from conftest import needs_torch

from src.mesh.create_mesh import (
    get_edges_from_faces,
    get_hierarchy_of_triangular_meshes_for_sphere,
    refine_mesh_in_region,
)

ROI = (50.0, 60.0, 83.0, 98.0)


@needs_torch
def test_edges_are_labelled_by_coarsest_level():
    import torch

    from src.create_graphs import processing_edge_levels
    from src.mesh.create_mesh import filter_mesh

    meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=4)
    meshes.append(refine_mesh_in_region(meshes[-1], *ROI, buffer_deg=2.0))
    levels = [2, 4, 5]
    ei = torch.tensor(get_edges_from_faces(filter_mesh(meshes, levels).faces))
    lv = processing_edge_levels(meshes, levels, ei)

    assert lv.shape == (ei.shape[1],)
    assert set(lv.tolist()) == {0, 1, 2}
    # обе стороны ребра — один уровень
    assert (lv[0::2] == lv[1::2]).all()
    # уровень 2 — длинные рёбра, сгущение — короткие и только над регионом
    V = meshes[-1].vertices
    length = np.linalg.norm(V[ei[0]] - V[ei[1]], axis=1)
    med = [np.median(length[lv.numpy() == k]) for k in range(3)]
    assert med[0] > 3 * med[1] > 6 * med[2] * 0.9
    fine = ei[:, lv == 2]
    lat = np.degrees(np.arcsin(V[fine[0], 2]))
    assert (lat > ROI[0] - 8).all() and (lat < ROI[1] + 8).all()
    # рёбра уровня 4 вне региона не переехали в уровень сгущения
    assert int((lv == 1).sum()) > int((lv == 2).sum())


@needs_torch
def test_zero_gates_reproduce_plain_layer_and_gates_learn():
    import torch

    from src.models import InteractionNetLayer

    torch.manual_seed(0)
    plain = InteractionNetLayer(node_dim=8, edge_dim=8, hidden_dim=8)
    torch.manual_seed(0)
    gated = InteractionNetLayer(node_dim=8, edge_dim=8, hidden_dim=8, num_levels=3)
    n, e = 30, 120
    x = torch.randn(n, 8)
    ei = torch.randint(0, n, (2, e))
    ea = torch.randn(e, 8)
    lv = torch.randint(0, 3, (e,))

    y0, _ = plain(x, ei, ea)
    y1, _ = gated(x, ei, ea, lv)
    torch.testing.assert_close(y0, y1)

    y1.square().sum().backward()
    assert gated.level_gate.grad is not None and gated.level_gate.grad.abs().sum() > 0

    with torch.no_grad():
        gated.level_gate.fill_(0.5)
    y2, _ = gated(x, ei, ea, lv)
    assert not torch.allclose(y0, y2)


@needs_torch
def test_gated_layer_requires_levels():
    import torch

    from src.models import InteractionNetLayer
    layer = InteractionNetLayer(node_dim=4, edge_dim=4, hidden_dim=4, num_levels=2)
    with pytest.raises(ValueError):
        layer(torch.randn(5, 4), torch.tensor([[0, 1], [1, 2]]), torch.randn(2, 4))


@needs_torch
@pytest.mark.parametrize("steps", [1, 2])
def test_model_with_level_aggregation_takes_trained_weights(steps):
    """Модель с поправкой грузит веса модели со сгущением без неё: недостаёт
    только нулевых весов уровней, и прямой проход идёт."""
    import test_mesh_refine as tm
    import torch

    def build(levels, seed):
        from src.models import WeatherPrediction
        lats = np.repeat(np.linspace(-80, 80, 17), 36).astype(np.float32)
        lons = np.tile(np.arange(0, 360, 10.0), 17).astype(np.float32)
        graph, pipeline, data = tm.config(True, steps)
        pipeline.processor.gcn.level_aggregation = levels
        torch.manual_seed(seed)
        return WeatherPrediction(cordinates=(lats, lons), graph_config=graph,
                                 pipeline_config=pipeline, data_config=data,
                                 device=torch.device("cpu"), flat_grid=True).eval()

    old = build(0, 1)
    new = build(2 + steps, 2)
    assert new._processing_edge_level is not None
    assert int(new._processing_edge_level.max()) == 1 + steps
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(k.endswith("level_gate") for k in missing), missing
    X = torch.randn(1, new._num_grid_nodes, tm.N_FEAT * tm.OBS)
    with torch.no_grad():
        y_new = new(X, attention_threshold=0.0)
        y_old = old(X, attention_threshold=0.0)
    torch.testing.assert_close(y_new, y_old)


@needs_torch
def test_wrong_level_count_is_rejected():
    import test_mesh_refine as tm
    import torch

    from src.models import WeatherPrediction
    graph, pipeline, data = tm.config(True, 1)
    pipeline.processor.gcn.level_aggregation = 5
    lats = np.repeat(np.linspace(-80, 80, 17), 36).astype(np.float32)
    lons = np.tile(np.arange(0, 360, 10.0), 17).astype(np.float32)
    with pytest.raises(ValueError, match="level_aggregation"):
        WeatherPrediction(cordinates=(lats, lons), graph_config=graph,
                          pipeline_config=pipeline, data_config=data,
                          device=torch.device("cpu"), flat_grid=True)
