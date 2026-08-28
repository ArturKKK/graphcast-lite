"""Модель целиком: форма вывода, отсутствие весов «по номеру узла», перенумерация.

Требует torch.

Главный тест здесь — перенумерация узлов сетки. 28.08.2026 им был закрыт вопрос
о сборке корпуса постобработки: координаты узлов там восстанавливаются
построением, и я утверждал, что даже разошедшийся порядок не изменил бы прогноз.
Довод был двойной: обучаемых весов, привязанных к номеру узла, у сети нет, а
рёбра строятся по координатам. Здесь проверяются оба — на настоящей модели, а не
на её отдельных блоках.

Модель берётся маленькая: меш уровней [1, 2], скрытая размерность 16, два шага
обмена сообщениями. Проверяемые свойства от размера не зависят, а маленькая
считается за секунды.
"""
import numpy as np
import pytest
from conftest import needs_torch

pytestmark = needs_torch

N_FEAT, OBS = 4, 2


def small_config():
    from src.config import DataConfig, GraphBuildingConfig, PipelineConfig
    graph = GraphBuildingConfig(**{
        "grid2mesh_edge_creation": "radius",
        "mesh2grid_edge_creation": "contained",
        "grid2mesh_radius_query": 0.9,
        "mesh_levels": [1, 2],
    })
    mlp = {"mlp_hidden_dims": [16, 16], "output_dim": 16, "use_layer_norm": True,
           "layer_norm_mode": "node"}
    pipeline = PipelineConfig(**{
        "encoder": {"mlp": mlp,
                    "gcn": {"layer_type": "conv_gcn", "hidden_dims": [16, 16],
                            "output_dim": 16, "activation": "swish"}},
        "processor": {"gcn": {"layer_type": "interaction_net", "output_dim": 16,
                              "activation": "swish", "use_layer_norm": True,
                              "num_message_passing_steps": 2,
                              "edge_feature_dim": 4}},
        "decoder": {"mlp": {"mlp_hidden_dims": [16, 16], "output_dim": 16,
                            "use_layer_norm": False},
                    "gcn": {"layer_type": "conv_gcn", "hidden_dims": [16, 16],
                            "output_dim": N_FEAT, "activation": "swish"}},
    })
    data = DataConfig(**{"dataset_name": "multires", "num_features_used": N_FEAT,
                         "obs_window_used": OBS, "pred_window_used": 1,
                         "want_feats_flattened": True})
    return graph, pipeline, data


def build_model(lats, lons):
    import torch

    from src.models import WeatherPrediction
    graph, pipeline, data = small_config()
    torch.manual_seed(0)
    return WeatherPrediction(
        cordinates=(lats, lons), graph_config=graph, pipeline_config=pipeline,
        data_config=data, device=torch.device("cpu"), flat_grid=True).eval()


def flat_nodes(seed=0, n_lat=6, n_lon=10):
    """Плоская сетка: координаты спарены по узлам, как у мультимасштабной."""
    lats = np.linspace(-60.0, 60.0, n_lat)
    lons = np.linspace(0.0, 324.0, n_lon)
    LO, LA = np.meshgrid(lons, lats)
    return LA.reshape(-1).astype(np.float32), LO.reshape(-1).astype(np.float32)


@pytest.fixture(scope="module")
def built():
    lats, lons = flat_nodes()
    return build_model(lats, lons), lats, lons


def test_output_has_one_value_per_node_and_feature(built):
    import torch
    model, lats, _ = built
    x = torch.randn(1, len(lats), N_FEAT * OBS)
    with torch.no_grad():
        out = model(X=x, attention_threshold=0.0)
    # Модель отдаёт (N, F) без размерности батча при батче в одну выборку —
    # это и разбирает train_epoch, добавляя её обратно через unsqueeze(0).
    assert out.shape[-2:] == (len(lats), N_FEAT)
    assert out.dim() in (2, 3)


def test_no_learnable_weight_is_indexed_by_node(built):
    """Ни один вес не пронумерован по узлам — основа довода о перенумерации.

    Если бы у сети было вложение узла, как у постпроцессора вложение станции,
    порядок узлов стал бы частью модели, и восстановленные построением
    координаты пришлось бы сверять с обучением до последнего узла.
    """
    model, lats, _ = built
    n_grid = len(lats)
    n_mesh = model._num_mesh_nodes
    forbidden = {n_grid, n_mesh, n_grid + n_mesh}
    bad = [(n, tuple(p.shape)) for n, p in model.named_parameters()
           if forbidden & set(p.shape)]
    assert not bad, f"веса, привязанные к числу узлов: {bad}"


def test_no_embedding_layers_at_all(built):
    import torch.nn as nn
    model, _, _ = built
    embs = [n for n, m in model.named_modules() if isinstance(m, nn.Embedding)]
    assert not embs, f"найдены вложения: {embs}"


def test_renumbering_nodes_permutes_the_forecast(built):
    """ГЛАВНЫЙ ТЕСТ. Перенумеровали узлы — прогноз переставился так же.

    Строим вторую модель на переставленных координатах, переносим в неё веса
    первой и подаём переставленные данные. Прогноз обязан совпасть с прогнозом
    первой модели, переставленным тем же порядком.

    Именно это утверждение закрывало вопрос о сборке корпуса постобработки: даже
    разойдись нумерация узлов, прогноз был бы тот же, лишь переставленный.
    """
    import torch
    model, lats, lons = built
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(lats))

    model2 = build_model(lats[perm], lons[perm])
    model2.load_state_dict(model.state_dict())
    model2.eval()

    x = torch.randn(1, len(lats), N_FEAT * OBS)
    with torch.no_grad():
        out = model(X=x, attention_threshold=0.0)
        out2 = model2(X=x[:, perm, :], attention_threshold=0.0)
    # Модель может отдать как (N, F), так и (B, N, F) — приводим к узлам.
    a = out.reshape(-1, N_FEAT)[perm]
    b = out2.reshape(-1, N_FEAT)
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, (
        f"прогноз зависит от нумерации узлов (расхождение {diff:.2e}) — "
        f"довод о корпусе неверен")


def test_forward_is_deterministic(built):
    import torch
    model, lats, _ = built
    x = torch.randn(1, len(lats), N_FEAT * OBS)
    with torch.no_grad():
        a = model(X=x, attention_threshold=0.0)
        b = model(X=x, attention_threshold=0.0)
    assert torch.allclose(a, b, atol=0.0), "два прогона дали разное"


def test_forecast_depends_on_the_input(built):
    """Вырожденная проверка: сеть не выдаёт одно и то же на любой вход."""
    import torch
    model, lats, _ = built
    with torch.no_grad():
        a = model(X=torch.zeros(1, len(lats), N_FEAT * OBS), attention_threshold=0.0)
        b = model(X=torch.ones(1, len(lats), N_FEAT * OBS), attention_threshold=0.0)
    assert not torch.allclose(a, b, atol=1e-6)
