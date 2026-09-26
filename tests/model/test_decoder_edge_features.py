"""Декодировщик с признаками рёбер (InteractionNetDecoder).

Зачем он. 24.09.2026 разложение ошибки по масштабам показало отпечаток
треугольников меша в прогнозе: разность ошибок соседних узлов через сторону
треугольника в 1,8–2,8 раза больше, чем внутри него, а у GraphCast около 1,1
(scripts/paper_triangle_imprint.py). Причина в GCNConv: на графе «три вершины →
узел» нормировка даёт вершинам одинаковый вес, и положение узла внутри
треугольника не учитывается.

Здесь проверяется, что новый декодировщик это чинит и что его можно поставить
поверх обученной модели: кодировщик, процессор и MLP декодировщика грузятся из
чекпойнта со старым декодировщиком.
"""
import numpy as np
import pytest
from conftest import needs_torch

pytestmark = needs_torch

N_FEAT, OBS = 4, 2


def config(decoder_type, encoder_type="conv_gcn"):
    from src.config import DataConfig, GraphBuildingConfig, PipelineConfig
    graph = GraphBuildingConfig(**{
        "grid2mesh_edge_creation": "radius", "mesh2grid_edge_creation": "contained",
        "grid2mesh_radius_query": 0.9, "mesh_levels": [1, 2]})
    mlp = {"mlp_hidden_dims": [16, 16], "output_dim": 16, "use_layer_norm": True,
           "layer_norm_mode": "node"}
    if decoder_type == "conv_gcn":
        dec_gcn = {"layer_type": "conv_gcn", "hidden_dims": [16, 16],
                   "output_dim": N_FEAT, "activation": "swish"}
    else:
        dec_gcn = {"layer_type": "interaction_net_decoder", "hidden_dims": [16],
                   "output_dim": N_FEAT, "activation": "swish", "edge_feature_dim": 4}
    if encoder_type in ("conv_gcn", "refine"):
        enc_gcn = {"layer_type": "conv_gcn", "hidden_dims": [16, 16],
                   "output_dim": 16, "activation": "swish"}
        if encoder_type == "refine":
            enc_gcn.update(edge_refine=True, edge_feature_dim=4)
    else:
        enc_gcn = {"layer_type": "interaction_net_encoder", "hidden_dims": [16],
                   "output_dim": 16, "activation": "swish", "edge_feature_dim": 4}
    pipeline = PipelineConfig(**{
        "encoder": {"mlp": mlp, "gcn": enc_gcn},
        "processor": {"gcn": {"layer_type": "interaction_net", "output_dim": 16,
                              "activation": "swish", "use_layer_norm": True,
                              "num_message_passing_steps": 2, "edge_feature_dim": 4}},
        "decoder": {"mlp": {"mlp_hidden_dims": [16, 16], "output_dim": 16,
                            "use_layer_norm": False}, "gcn": dec_gcn},
    })
    data = DataConfig(**{"dataset_name": "multires", "num_features_used": N_FEAT,
                         "obs_window_used": OBS, "pred_window_used": 1,
                         "want_feats_flattened": True})
    return graph, pipeline, data


def flat_nodes(n_lat=6, n_lon=10):
    lats = np.linspace(-60.0, 60.0, n_lat)
    lons = np.linspace(0.0, 324.0, n_lon)
    LO, LA = np.meshgrid(lons, lats)
    return LA.reshape(-1).astype(np.float32), LO.reshape(-1).astype(np.float32)


def build(decoder_type, lats=None, lons=None, flat=True, seed=0, encoder_type="conv_gcn"):
    import torch

    from src.models import WeatherPrediction
    if lats is None:
        lats, lons = flat_nodes()
    graph, pipeline, data = config(decoder_type, encoder_type)
    torch.manual_seed(seed)
    return WeatherPrediction(cordinates=(lats, lons), graph_config=graph,
                             pipeline_config=pipeline, data_config=data,
                             device=torch.device("cpu"), flat_grid=flat).eval()


def randomise_output(model, seed=1):
    """Снять нулевую инициализацию последнего слоя, чтобы выход был ненулевым."""
    import torch
    g = torch.Generator().manual_seed(seed)
    last = model.decoder.graph_layer.layers.out[-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.1)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * 0.1)


@pytest.fixture(scope="module")
def new_model():
    return build("interaction_net_decoder")


def test_untrained_decoder_gives_zero_increment(new_model):
    """Нулевая инициализация: до обучения прогноз равен инерционному."""
    import torch
    n = new_model._num_grid_nodes
    X = torch.randn(1, n, N_FEAT * OBS)
    with torch.no_grad():
        out = new_model(X, attention_threshold=0.0)
    assert out.shape == (n, N_FEAT)
    assert torch.count_nonzero(out) == 0


def test_edge_features_three_per_node_and_normalised(new_model):
    ef = new_model._decoding_edge_features
    ei = new_model.decoding_graph
    n = new_model._num_grid_nodes
    assert ef.shape == (ei.shape[1], 4)
    counts = np.bincount(ei[1].numpy(), minlength=n)
    assert (counts == 3).all(), "у каждого узла сетки ровно три вершины треугольника"
    d = ef[:, 0].numpy()
    assert d.max() == pytest.approx(1.0) and d.min() > 0
    # длина равна норме смещения: признаки согласованы между собой
    np.testing.assert_allclose(np.linalg.norm(ef[:, 1:].numpy(), axis=1), d, rtol=1e-5)


def test_regular_and_flat_grid_give_same_features():
    """Для регулярной сетки координаты узлов разворачиваются по строкам широт."""
    lat_ax = np.linspace(-60.0, 60.0, 6).astype(np.float32)
    lon_ax = np.linspace(0.0, 324.0, 10).astype(np.float32)
    reg = build("interaction_net_decoder", lat_ax, lon_ax, flat=False)
    flat = build("interaction_net_decoder", *flat_nodes(), flat=True)
    assert torch_equal(reg.decoding_graph, flat.decoding_graph)
    np.testing.assert_allclose(reg._decoding_edge_features.numpy(),
                               flat._decoding_edge_features.numpy(), atol=1e-6)


def torch_equal(a, b):
    return a.shape == b.shape and bool((a == b).all())


def test_position_inside_triangle_matters():
    """Главное свойство: два узла с одинаковыми признаками и одними вершинами,
    но в разных местах треугольника, получают разный выход. У GCNConv он
    одинаковый — это и есть отпечаток треугольников."""
    import torch
    from torch_geometric.nn import GCNConv

    from src.models import InteractionNetDecoderLayer
    torch.manual_seed(0)
    # узлы 0..2 — вершины, 3 и 4 — узлы сетки с одинаковыми признаками
    x = torch.randn(5, 8)
    x[4] = x[3]
    ei = torch.tensor([[0, 1, 2, 0, 1, 2], [3, 3, 3, 4, 4, 4]])
    ef = torch.tensor([[.2, .1, .1, 0], [.9, .6, .6, 0], [.9, -.6, .6, 0],     # узел 3 у вершины 0
                       [.6, .4, .4, 0], [.6, -.4, .4, 0], [.6, 0, -.6, 0.]])   # узел 4 в центре
    layer = InteractionNetDecoderLayer(node_dim=8, raw_edge_dim=4, hidden_dim=16, output_dim=3)
    torch.nn.init.normal_(layer.out[-1].weight, std=0.1)
    with torch.no_grad():
        y = layer(x, ei, ef)
        g = GCNConv(8, 3)(x, ei)
    assert not torch.allclose(y[3], y[4], atol=1e-4), "выход не зависит от положения узла"
    assert torch.allclose(g[3], g[4], atol=1e-6), "контроль: у GCN выход одинаковый"


def test_trained_weights_load_except_new_decoder_layer():
    """Кодировщик, процессор и MLP декодировщика грузятся из модели со старым
    декодировщиком; новым остаётся только сам слой сообщений."""
    old = build("conv_gcn", seed=3)
    new = build("interaction_net_decoder", seed=4)
    state = old.state_dict()
    assert not any(k.startswith("_decoding_edge_features") for k in new.state_dict())
    own = new.state_dict()
    clash = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    assert not clash, f"совпали имена с иным размером: {clash}"
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert all(k.startswith("decoder.graph_layer.") for k in missing), missing
    assert all(k.startswith("decoder.graph_layer.") for k in unexpected), unexpected
    for k, v in new.state_dict().items():
        if k.startswith(("encoder.", "processor.", "decoder.mlp.")):
            assert torch_equal(v, state[k]), k


def test_node_renumbering_permutes_output():
    """Весов по номеру узла нет и у нового декодировщика."""
    import torch
    lats, lons = flat_nodes()
    perm = np.random.default_rng(0).permutation(len(lats))
    a = build("interaction_net_decoder", lats, lons)
    b = build("interaction_net_decoder", lats[perm], lons[perm])
    b.load_state_dict(a.state_dict())
    randomise_output(a)
    randomise_output(b)
    X = torch.randn(1, len(lats), N_FEAT * OBS)
    with torch.no_grad():
        ya = a(X, attention_threshold=0.0)
        yb = b(X[:, perm], attention_threshold=0.0)
    assert torch.allclose(ya[perm], yb, atol=1e-5)


# ---------- кодировщик с признаками рёбер (25.09.2026) ----------

ENC = "interaction_net_encoder"


def test_encoder_edge_features_point_grid_to_mesh():
    m = build("interaction_net_decoder", encoder_type=ENC)
    ef, ei, n = m._encoding_edge_features, m.encoding_graph, m._num_grid_nodes
    assert ef.shape == (ei.shape[1], 4)
    assert (ei[0] < n).all() and (ei[1] >= n).all(), "рёбра идут от сетки к мешу"
    d = ef[:, 0].numpy()
    assert d.max() == pytest.approx(1.0) and d.min() >= 0
    np.testing.assert_allclose(np.linalg.norm(ef[:, 1:].numpy(), axis=1), d, rtol=1e-5)


def test_encoder_position_matters():
    """Одна вершина, два узла сетки с одинаковыми признаками на разных местах:
    новый кодировщик различает, откуда пришёл узел, GCN — нет."""
    import torch

    from src.models import InteractionNetEncoderLayer
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    x[1] = x[0]
    ei_a = torch.tensor([[0], [2]])
    ei_b = torch.tensor([[1], [2]])
    layer = InteractionNetEncoderLayer(node_dim=8, raw_edge_dim=4, hidden_dim=16)
    with torch.no_grad():
        ya = layer(x, ei_a, torch.tensor([[.3, .2, .2, 0.]]))
        yb = layer(x, ei_b, torch.tensor([[.9, -.6, .6, 0.]]))
    assert not torch.allclose(ya[2], yb[2], atol=1e-4)


def test_encoder_and_decoder_load_trained_rest():
    """От модели со старыми блоками грузятся MLP кодировщика, процессор и MLP
    декодировщика; новыми остаются только два слоя сообщений."""
    old = build("conv_gcn", seed=3)
    new = build("interaction_net_decoder", seed=4, encoder_type=ENC)
    state = old.state_dict()
    own = new.state_dict()
    assert not [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    missing, unexpected = new.load_state_dict(state, strict=False)
    prefixes = ("encoder.graph_layer.", "decoder.graph_layer.")
    assert all(k.startswith(prefixes) for k in missing), missing
    assert all(k.startswith(prefixes) for k in unexpected), unexpected
    for k, v in new.state_dict().items():
        if k.startswith(("encoder.mlp.", "processor.", "decoder.mlp.")):
            assert torch_equal(v, state[k]), k


def test_encoder_node_renumbering_permutes_output():
    import torch
    lats, lons = flat_nodes()
    perm = np.random.default_rng(1).permutation(len(lats))
    a = build("interaction_net_decoder", lats, lons, encoder_type=ENC)
    b = build("interaction_net_decoder", lats[perm], lons[perm], encoder_type=ENC)
    b.load_state_dict(a.state_dict())
    randomise_output(a)
    randomise_output(b)
    X = torch.randn(1, len(lats), N_FEAT * OBS)
    with torch.no_grad():
        ya = a(X, attention_threshold=0.0)
        yb = b(X[:, perm], attention_threshold=0.0)
    assert torch.allclose(ya[perm], yb, atol=1e-5)


# ---------- кодировщик: GCN + поправка с нуля (26.09.2026) ----------

def test_refine_starts_exactly_from_trained_model():
    """Полная замена кодировщика провалилась: процессор терял вход. Поправка
    стартует с нуля, и модель с ней до обучения выдаёт ровно то же, что без неё."""
    import torch
    base = build("interaction_net_decoder", seed=5)
    randomise_output(base)
    ref = build("interaction_net_decoder", seed=6, encoder_type="refine")
    missing, unexpected = ref.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(k.startswith("encoder.graph_layer.refine.") for k in missing), missing
    assert not unexpected, unexpected
    X = torch.randn(1, base._num_grid_nodes, N_FEAT * OBS)
    with torch.no_grad():
        assert torch.equal(base(X, attention_threshold=0.0), ref(X, attention_threshold=0.0))


def test_refine_uses_position_after_training_step():
    """После ненулевой поправки выход зависит от признаков рёбер."""
    import torch
    m = build("interaction_net_decoder", seed=7, encoder_type="refine")
    randomise_output(m)
    last = m.encoder.graph_layer.refine.update[-1]
    with torch.no_grad():
        last.weight.normal_(std=0.1)
    # Смотрим на выход кодировщика: до выхода модели поправка доходит
    # ослабленной нормировками процессора, и сравнение там было бы шатким.
    X = m._preprocess_input(grid_node_features=torch.randn(m._num_grid_nodes, N_FEAT * OBS))
    with torch.no_grad():
        e1 = m._encode(X)
        m._encoding_edge_features.mul_(-1)
        e2 = m._encode(X)
    assert (e1 - e2).abs().max() > 1e-4
