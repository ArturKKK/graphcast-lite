"""Независимость обмена сообщениями от порядка узлов и рёбер.

Требует torch.

Зачем это проверять. На независимости от порядка узлов держится довод, которым
28.08.2026 был закрыт вопрос о сборке корпуса постобработки: координаты узлов
там восстанавливаются построением, а не читаются из слитого набора, и я
утверждал, что даже разошедшаяся нумерация не изменила бы прогноз — обучаемых
весов, привязанных к номеру узла, у сети нет, а рёбра строятся по координатам.
Утверждение было верным по чтению кода, но ни разу не измерено. Здесь измеряется.

Второе, что проверяется, — порядок РЁБЕР. Рёбра приходят из запроса по радиусу,
и их порядок зависит от того, как kd-дерево обошло дерево. Зависимость от него
означала бы, что прогноз меняется от перестроения графа.
"""
import pytest
from conftest import needs_torch

pytestmark = needs_torch

NODE_DIM, EDGE_DIM, HIDDEN, N_NODES, N_EDGES = 8, 5, 16, 40, 160


@pytest.fixture
def layer_cls():
    from src.models import InteractionNetLayer
    return InteractionNetLayer


@pytest.fixture
def proc_cls():
    from src.models import InteractionNetProcessor
    return InteractionNetProcessor


def make_graph(seed=0, n_nodes=N_NODES, n_edges=N_EDGES):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, NODE_DIM, generator=g)
    ei = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    ea = torch.randn(n_edges, EDGE_DIM, generator=g)
    return x, ei, ea


def permute_graph(x, edge_index, perm):
    """Переставить узлы. perm[i] — какой СТАРЫЙ узел стоит на новом месте i."""
    import torch
    pos = torch.argsort(perm)                 # pos[старый] = новый
    return x[perm], pos[edge_index]


def test_node_permutation_permutes_the_output(layer_cls):
    """Переставили узлы — вывод переставился так же, и ничего больше.

    Это и есть утверждение «сети безразличен порядок узлов», проверенное
    измерением, а не чтением.
    """
    import torch
    torch.manual_seed(0)
    layer = layer_cls(NODE_DIM, EDGE_DIM, HIDDEN).eval()
    x, ei, ea = make_graph()
    with torch.no_grad():
        out_x, out_e = layer(x, ei, ea)
        perm = torch.randperm(N_NODES)
        x2, ei2 = permute_graph(x, ei, perm)
        out_x2, out_e2 = layer(x2, ei2, ea)
    assert torch.allclose(out_x2, out_x[perm], atol=1e-5), "вывод узлов не переставился"
    assert torch.allclose(out_e2, out_e, atol=1e-5), "признаки рёбер изменились"


def test_edge_permutation_does_not_change_node_output(layer_cls):
    """Порядок рёбер на узлы не влияет.

    Рёбра приходят из запроса по радиусу, и их порядок задан обходом kd-дерева.
    Зависимость от него означала бы, что прогноз меняется от перестроения графа.
    """
    import torch
    torch.manual_seed(1)
    layer = layer_cls(NODE_DIM, EDGE_DIM, HIDDEN).eval()
    x, ei, ea = make_graph(seed=1)
    with torch.no_grad():
        out_x, out_e = layer(x, ei, ea)
        eperm = torch.randperm(N_EDGES)
        out_x2, out_e2 = layer(x, ei[:, eperm], ea[eperm])
    assert torch.allclose(out_x2, out_x, atol=1e-5)
    assert torch.allclose(out_e2, out_e[eperm], atol=1e-5)


@pytest.mark.parametrize("agg", ["mean", "sum"])
def test_both_aggregations_are_equivariant(layer_cls, agg):
    import torch
    torch.manual_seed(2)
    layer = layer_cls(NODE_DIM, EDGE_DIM, HIDDEN, aggregation=agg).eval()
    x, ei, ea = make_graph(seed=2)
    with torch.no_grad():
        out_x, _ = layer(x, ei, ea)
        perm = torch.randperm(N_NODES)
        x2, ei2 = permute_graph(x, ei, perm)
        out_x2, _ = layer(x2, ei2, ea)
    assert torch.allclose(out_x2, out_x[perm], atol=1e-5)


def test_sum_and_mean_actually_differ(layer_cls):
    """Способ сведения — не косметика.

    При сумме длинные рёбра ДОБАВЛЯЮТ сигнал, при среднем РАЗБАВЛЯЮТ локальный.
    Это подозреваемый номер один в том, почему полный меш [0..6] проиграл.
    Тест держит различие явным: если кто-то сведёт их к одному, это заметят.
    """
    import torch
    torch.manual_seed(3)
    x, ei, ea = make_graph(seed=3)
    outs = []
    for agg in ("mean", "sum"):
        torch.manual_seed(3)                  # одинаковые начальные веса
        layer = layer_cls(NODE_DIM, EDGE_DIM, HIDDEN, aggregation=agg).eval()
        with torch.no_grad():
            outs.append(layer(x, ei, ea)[0])
    assert not torch.allclose(outs[0], outs[1], atol=1e-4)


def test_isolated_node_is_not_nan(layer_cls):
    """Узел без входящих рёбер обновляется от собственных признаков, а не в NaN.

    Такой узел возможен: радиус поиска задаётся долей наибольшего ребра меша, и
    при неудачном множителе узлы сетки остаются без связей. Отдельная проверка
    на это стоит в create_graphs, но и здесь поведение должно быть определённым.
    """
    import torch
    torch.manual_seed(4)
    layer = layer_cls(NODE_DIM, EDGE_DIM, HIDDEN).eval()
    x = torch.randn(5, NODE_DIM)
    ei = torch.tensor([[0, 1], [1, 2]])       # узлы 3 и 4 без рёбер вовсе
    ea = torch.randn(2, EDGE_DIM)
    with torch.no_grad():
        out, _ = layer(x, ei, ea)
    assert torch.isfinite(out).all()


def test_processor_is_equivariant(proc_cls):
    """Весь процессор целиком — двенадцать шагов, как в модели статьи."""
    import torch
    torch.manual_seed(5)
    proc = proc_cls(node_dim=NODE_DIM, raw_edge_dim=4, edge_latent_dim=EDGE_DIM,
                    hidden_dim=HIDDEN, num_steps=12).eval()
    x, ei, _ = make_graph(seed=5)
    raw = torch.randn(N_EDGES, 4)
    with torch.no_grad():
        out = proc(x, ei, raw)
        perm = torch.randperm(N_NODES)
        x2, ei2 = permute_graph(x, ei, perm)
        out2 = proc(x2, ei2, raw)
    got = out[0] if isinstance(out, tuple) else out
    got2 = out2[0] if isinstance(out2, tuple) else out2
    assert torch.allclose(got2, got[perm], atol=1e-4), (
        "процессор зависит от нумерации узлов")


def test_processor_steps_do_not_share_weights(proc_cls):
    """Шаги процессора независимы, как в GraphCast.

    Общие веса — другая модель с другой ёмкостью; если кто-то их свяжет,
    сравнение с опубликованными числами станет несравнением.
    """
    import torch
    torch.manual_seed(6)
    proc = proc_cls(node_dim=NODE_DIM, raw_edge_dim=4, edge_latent_dim=EDGE_DIM,
                    hidden_dim=HIDDEN, num_steps=3)
    layers = [m for m in proc.modules() if type(m).__name__ == "InteractionNetLayer"]
    assert len(layers) == 3
    first = dict(layers[0].named_parameters())
    second = dict(layers[1].named_parameters())
    assert not all(torch.equal(first[k], second[k]) for k in first), "веса шагов совпали"
