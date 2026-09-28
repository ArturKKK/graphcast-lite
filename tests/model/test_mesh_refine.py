"""Сгущение меша над регионом (refine_region, 28.09.2026).

Над вставкой 0,25° треугольники самого частого уровня делятся на 4, шаг ~55 км
вместо ~110 км. Здесь проверяется то, без чего опыт бессмыслен или вреден:
сетка без дыр и висячих вершин, номера прежних вершин не сдвинулись (иначе
обученные веса и грани грубых уровней указывали бы не туда), признаки прежних
рёбер процессора не изменились, а модель грузит веса модели без сгущения.
"""
from collections import Counter

import numpy as np
import pytest
from conftest import needs_torch

from src.mesh.create_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    refine_mesh_in_region,
)

ROI = (50.0, 60.0, 83.0, 98.0)


@pytest.fixture(scope="module")
def meshes():
    m = get_hierarchy_of_triangular_meshes_for_sphere(splits=5)
    return m[-1], refine_mesh_in_region(m[-1], *ROI, buffer_deg=2.0)


def edge_counts(faces):
    c = Counter()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            c[tuple(sorted((int(a), int(b))))] += 1
    return c


def test_mesh_is_closed_and_consistent(meshes):
    _, r = meshes
    cnt = edge_counts(r.faces)
    assert all(v == 2 for v in cnt.values()), "есть дыра или висячая вершина"
    assert len(r.vertices) - len(cnt) + len(r.faces) == 2, "не сфера по Эйлеру"
    V, F = r.vertices, r.faces
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    assert ((n * V[F].mean(1)).sum(1) > 0).all(), "есть вывернутые грани"
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-9)


def test_old_vertices_keep_their_numbers(meshes):
    m, r = meshes
    assert len(r.vertices) > len(m.vertices)
    np.testing.assert_array_equal(r.vertices[:len(m.vertices)], m.vertices)


def test_new_vertices_lie_over_the_region(meshes):
    m, r = meshes
    new = r.vertices[len(m.vertices):]
    lat = np.degrees(np.arcsin(new[:, 2]))
    lon = np.degrees(np.arctan2(new[:, 1], new[:, 0])) % 360
    # запас 2° плюс полтреугольника на замыкание
    assert (lat > ROI[0] - 5).all() and (lat < ROI[1] + 5).all()
    assert (lon > ROI[2] - 8).all() and (lon < ROI[3] + 8).all()


def test_edges_over_region_are_twice_shorter(meshes):
    m, r = meshes

    def lens(mesh, inside):
        V = mesh.vertices
        out = []
        for a, b in edge_counts(mesh.faces):
            p = (V[a] + V[b]) / 2
            la = np.degrees(np.arcsin(p[2] / np.linalg.norm(p)))
            lo = np.degrees(np.arctan2(p[1], p[0])) % 360
            in_roi = ROI[0] <= la <= ROI[1] and ROI[2] <= lo <= ROI[3]
            # «вне» — с отступом: в запасе и переходной полосе рёбра тоже делятся
            far = not (ROI[0] - 10 <= la <= ROI[1] + 10 and ROI[2] - 15 <= lo <= ROI[3] + 15)
            if (inside and in_roi) or (not inside and far):
                out.append(np.linalg.norm(V[a] - V[b]))
        return np.median(out)

    assert lens(r, True) == pytest.approx(lens(m, True) / 2, rel=0.1)
    assert lens(r, False) == pytest.approx(lens(m, False), rel=0.01), "вне региона сетка изменилась"


# ---------- модель ----------

N_FEAT, OBS = 4, 2


def config(refine):
    from src.config import DataConfig, GraphBuildingConfig, PipelineConfig
    g = {"grid2mesh_edge_creation": "radius", "mesh2grid_edge_creation": "contained",
         "grid2mesh_radius_query": 0.9, "mesh_levels": [1, 3]}
    if refine:
        g["refine_region"] = [30.0, 70.0, 60.0, 120.0]
    mlp = {"mlp_hidden_dims": [16, 16], "output_dim": 16, "use_layer_norm": True,
           "layer_norm_mode": "node"}
    pipeline = PipelineConfig(**{
        "encoder": {"mlp": mlp, "gcn": {"layer_type": "conv_gcn", "hidden_dims": [16, 16],
                                        "output_dim": 16, "activation": "swish",
                                        "edge_refine": True, "edge_feature_dim": 4}},
        "processor": {"gcn": {"layer_type": "interaction_net", "output_dim": 16,
                              "activation": "swish", "use_layer_norm": True,
                              "num_message_passing_steps": 2, "edge_feature_dim": 4}},
        "decoder": {"mlp": {"mlp_hidden_dims": [16, 16], "output_dim": 16,
                            "use_layer_norm": False},
                    "gcn": {"layer_type": "interaction_net_decoder", "hidden_dims": [16],
                            "output_dim": N_FEAT, "activation": "swish",
                            "edge_feature_dim": 4}},
    })
    data = DataConfig(**{"dataset_name": "multires", "num_features_used": N_FEAT,
                         "obs_window_used": OBS, "pred_window_used": 1,
                         "want_feats_flattened": True})
    return GraphBuildingConfig(**g), pipeline, data


def build(refine, seed=0):
    import torch

    from src.models import WeatherPrediction
    lats = np.repeat(np.linspace(-80, 80, 17), 36).astype(np.float32)
    lons = np.tile(np.arange(0, 360, 10.0), 17).astype(np.float32)
    graph, pipeline, data = config(refine)
    torch.manual_seed(seed)
    return WeatherPrediction(cordinates=(lats, lons), graph_config=graph,
                             pipeline_config=pipeline, data_config=data,
                             device=torch.device("cpu"), flat_grid=True).eval()


@needs_torch
def test_refined_model_takes_trained_weights():
    """Все обучаемые веса модели без сгущения подходят; не совпадает по размеру
    только геометрический буфер признаков рёбер процессора (его конструктор
    пересчитывает), и его прежняя часть не изменилась."""
    import torch
    old = build(False, seed=1)
    new = build(True, seed=2)
    assert new._num_mesh_nodes > old._num_mesh_nodes
    own = new.state_dict()
    state = old.state_dict()
    clash = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    assert clash == ["_processing_edge_features"], clash

    # прежние рёбра процессора и их признаки на месте
    e_old = {tuple(e) for e in old.processing_graph.T.tolist()}
    e_new = {tuple(e): i for i, e in enumerate(new.processing_graph.T.tolist())}
    assert e_old <= set(e_new)
    idx_new = [e_new[e] for e in map(tuple, old.processing_graph.T.tolist())]
    # допуск 1e-5: округление float32 при пересчёте координат вершин
    torch.testing.assert_close(new._processing_edge_features[idx_new],
                               old._processing_edge_features, atol=1e-5, rtol=0)

    state = {k: v for k, v in state.items() if k not in clash}
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"_processing_edge_features"}, (missing, unexpected)
    X = torch.randn(1, new._num_grid_nodes, N_FEAT * OBS)
    with torch.no_grad():
        y = new(X, attention_threshold=0.0)
    assert y.shape == (new._num_grid_nodes, N_FEAT) and torch.isfinite(y).all()
