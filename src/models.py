"""Contains all the torch model definitions.

- Этот модуль описывает все блоки модели:
  * MLP: линейные слои с PReLU (+опц. LayerNorm) для подготовки признаков перед графовым слоем.
  * SparseGATConv: наследник GATConv, который может ПРОРЕЖИВАТЬ рёбра по порогу attention и возвращать новый edge_index.
  * GraphLayer: обёртка над конкретным типом графового слоя (SimpleConv/GCN/GAT/SparseGAT) с возможностью стека (GCN, GAT).
  * Model: композиция «(опц.) MLP → GraphLayer»; выравнивает интерфейс, сообщает output_dim вниз по пайплайну.
  * WeatherPrediction: основной pipeline (а-ля GraphCast):
      (опц.) продукт-граф (время×пространство) → подготовка фичей → ENCODER (Grid→Mesh) → PROCESSOR (Mesh↔Mesh) → DECODER (Mesh→Grid) → выход по grid.

КОНВЕНЦИИ:
- edge_index — тензор формы [2, E] с парами (sender→receiver).
- Узлы объединяются в единый массив: сначала ВСЕ grid (0..N-1), затем ВСЕ mesh (N..N+M-1).
- encoder/decoder работают на бипартийных графах (Grid↔Mesh), processor — только на mesh-графе.
"""

from typing import Tuple

import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, SimpleConv, GATConv, LayerNorm
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import dense_to_sparse, softmax
from torch_geometric.nn import summary

from src.config import (
    ModelConfig,
    MLPBlock,
    GraphBlock,
    GraphLayerType,
    GraphBuildingConfig,
    DataConfig,
    PipelineConfig,
    ProductGraphConfig,
    ProductGraphType,
)
from src.create_graphs import (
    create_decoding_graph,
    create_processing_graph,
    create_encoding_graph,
)


from src.mesh.create_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
)

from src.utils import get_mesh_lat_long


# Зачем: подготовить/сжать каналы перед графовым слоем.
# Пример: при obs_window=4, num_features=15 у каждого grid-узла вход 60 каналов (+ статика). MLP переводит это, скажем, в 64 канала — удобно для GNN.
class MLP(nn.Module):
    """Многоуровневый перцептрон для подготовки входных признаков (до GNN).

    - Конфиг задаёт список скрытых размеров и выходной размер (output_dim), а также LayerNorm.
    - Активация — PReLU (устойчив к «мертвым» нейронам, как у ReLU).
    - ВАЖНО: output_dim здесь фиксируется конфигом и далее служит входом в графовый слой.
    """
    def __init__(self, mlp_config: MLPBlock, input_dim):
        super().__init__()
        hidden_dims = mlp_config.mlp_hidden_dims
        output_dim = (
            mlp_config.output_dim
        )  # TODO: при желании можно связать с data.num_features, но сейчас жёстко в конфиге

        self.MLP = nn.ModuleList()
        in_features_for_last_layer = input_dim
        if hidden_dims:
            # Первый скрытый слой
            self.MLP.extend(
                [
                    nn.Linear(
                        in_features=input_dim,
                        out_features=hidden_dims[0],
                    ),
                    nn.PReLU(),
                ]
            )

            # Промежуточные скрытые слои
            for h_index in range(1, len(hidden_dims)):
                self.MLP.extend(
                    [
                        nn.Linear(
                            in_features=hidden_dims[h_index - 1],
                            out_features=hidden_dims[h_index],
                        ),
                        nn.PReLU(),
                    ]
                )
            in_features_for_last_layer = hidden_dims[-1]

        # Выходной слой проецирует в output_dim (каналы, с которыми дальше будет работать GNN)
        self.MLP.append(
            nn.Linear(in_features=in_features_for_last_layer, out_features=output_dim)
        )

        # (Опционально) LayerNorm поверх выходного пространства
        if mlp_config.use_layer_norm:
            self.MLP.append(
                LayerNorm(in_channels=output_dim, mode=mlp_config.layer_norm_mode)
            )

    def forward(self, X: torch.Tensor):
        for layer in self.MLP:
            X = layer(X)
        return X


class SparseGATConv(GATConv):
    """Вариант GAT, который может возвращать И новый edge_index после прореживания по attention.

    Идея:
    - Считаем обычный GAT: получаем out и attention_scores на рёбрах.
    - Если batch_num == 0 (например, первый батч эпохи), применяем порог attention_threshold:
      отбрасываем рёбра с малыми α, обновляя edge_index (сохранение разрежения на следующих шагах/эпохах вне этого слоя — ответственность вызывающего кода).
    - Возвращаем (out, (edge_index, attention_scores)) — чтобы выше по стеку можно было заменить граф.

    ВАЖНО:
    - concat=False → головы усредняются/агрегируются на уровне каналов (не конкатенируются), поэтому out имеет форму [N, out_channels].
    - attention_scores.squeeze(): приводим от [E, 1] к [E].
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=False,
                 dropout=0.0, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat=concat,
                                                      dropout=dropout, bias=bias, **kwargs)

    def forward(self, x, edge_index, attention_threshold=0.0, **kwargs):

        batch_num = kwargs.get('batch_num', 1)  # внешний код может прокинуть номер батча

        # 1) Стандартный прямой проход GAT с возвратом attention-весов на рёбрах
        out, (edge_index, attention_scores) = super().forward(x, edge_index, return_attention_weights=True)
        attention_scores = attention_scores.squeeze()  # [E]

        if batch_num == 0:
            # 2) Применяем порог, отбрасывая «слабые» рёбра
            mask = attention_scores >= attention_threshold
            mask = mask.type(torch.bool)

            # Небольшой лог — можно выключить при желании
            # print('attention threshold: ', attention_threshold)
            print('edge_index', edge_index.shape)
            # print('avg attn score: ', torch.mean(attention_scores))
            
            edge_index = edge_index[:, mask]
            attention_scores = attention_scores[mask]

        return out, (edge_index, attention_scores)


def _get_activation(name: str = "prelu"):
    """Возвращает модуль активации по имени."""
    if name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class InteractionNetLayer(nn.Module):
    """Один шаг InteractionNetwork а-ля GraphCast.

    На каждом шаге:
    1) Edge update: MLP_edge(concat(sender, receiver, edge_features)) → new edge_features
    2) Aggregate: scatter_mean обновлённых edge features для каждого receiver node
    3) Node update: MLP_node(concat(node, aggregated_edges)) → new node_features
    4) Residual: node = node + new_node, edge = edge + new_edge

    Поддерживает edge features и residual connections — два ключевых компонента,
    которых не хватало в старой модели.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 activation: str = "swish", use_layer_norm: bool = True):
        super().__init__()
        from torch_geometric.nn import LayerNorm as PygLayerNorm

        act = _get_activation(activation)

        # Edge MLP: [sender_features || receiver_features || edge_features] → edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, edge_dim),
        )

        # Node MLP: [node_features || aggregated_edge_features] → node_features
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, node_dim),
        )

        # LayerNorm (как в GraphCast)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.edge_norm = PygLayerNorm(edge_dim, mode="graph")
            self.node_norm = PygLayerNorm(node_dim, mode="node")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        x: [num_nodes, node_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        senders = edge_index[0]
        receivers = edge_index[1]

        # 1) Edge update
        edge_input = torch.cat([x[senders], x[receivers], edge_attr], dim=-1)
        edge_update = self.edge_mlp(edge_input)

        # 2) Aggregate edges → nodes (scatter_mean по receivers)
        from torch_geometric.utils import scatter
        aggregated = scatter(edge_update, receivers, dim=0, dim_size=x.size(0), reduce="mean")

        # 3) Node update
        node_input = torch.cat([x, aggregated], dim=-1)
        node_update = self.node_mlp(node_input)

        # 4) Residual connections
        new_edge_attr = edge_attr + edge_update
        new_x = x + node_update

        # 5) LayerNorm
        if self.use_layer_norm:
            new_edge_attr = self.edge_norm(new_edge_attr)
            new_x = self.node_norm(new_x)

        return new_x, new_edge_attr


class InteractionNetProcessor(nn.Module):
    """Processor из N шагов InteractionNetwork с UNSHARED weights (как GraphCast).

    Каждый шаг — отдельный InteractionNetLayer с собственными весами.
    Edge features сначала проецируются из raw (4D) в latent space, затем
    обновляются на каждом шаге.
    """

    def __init__(self, node_dim: int, raw_edge_dim: int, edge_latent_dim: int,
                 hidden_dim: int, num_steps: int,
                 activation: str = "swish", use_layer_norm: bool = True):
        super().__init__()

        act = _get_activation(activation)

        # Проецируем raw edge features (4D) в латентное пространство
        self.edge_encoder = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_latent_dim),
            act,
        )

        # N шагов message passing, каждый с отдельными весами
        self.steps = nn.ModuleList([
            InteractionNetLayer(
                node_dim=node_dim,
                edge_dim=edge_latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_steps)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr_raw: torch.Tensor):
        """
        x: [num_mesh_nodes, node_dim]
        edge_index: [2, E]
        edge_attr_raw: [E, raw_edge_dim] — сырые edge features (4D: distance + 3D position)
        """
        # Проецируем edge features в латентное пространство
        edge_attr = self.edge_encoder(edge_attr_raw)

        # Message passing
        for step in self.steps:
            x, edge_attr = step(x, edge_index, edge_attr)

        return x



class GraphLayer(nn.Module):
    """Обёртка, создающая конкретный графовый слой (или стек слоёв) по конфигу.

    Поддерживаемые типы:
    - SimpleConv: простая агрегация соседей (mean), без обучаемых весов (кроме, возможно, нормализации дальше).
    - ConvGCN: стек GCNConv с PReLU между слоями.
    - GATConv: стек GATConv (heads из конфига), concat=False → каналы не растут по числу голов.
    - SparseGATConv: один слой SparseGATConv (обычно достаточно одного, дальше нормализация по желанию).

    На выходе сохраняем output_dim (нужен для построения следующих блоков).
    """
    def __init__(self, graph_config: GraphBlock, input_dim):
        super().__init__()

        self.layer_type: GraphLayerType = graph_config.layer_type
        self.output_dim = None

        if graph_config.layer_type == GraphLayerType.SimpleConv:
            # SimpleConv не меняет число каналов: out_dim = in_dim
            self.output_dim = input_dim
            self.layers = SimpleConv(aggr="mean")

        elif graph_config.layer_type in [
            GraphLayerType.ConvGCN,
            GraphLayerType.GATConv,
            GraphLayerType.SparseGATConv,
        ]:
            self.activation = _get_activation(graph_config.activation or "prelu")
            self.output_dim = graph_config.output_dim
            self.layers = torch.nn.ModuleList()
            hidden_dims = graph_config.hidden_dims

            if graph_config.layer_type == GraphLayerType.ConvGCN:
                # Стек GCNConv: [in→h0] → act → [h0→h1] → act → ... → [h_{k-1}→out]
                self.layers.append(GCNConv(input_dim, hidden_dims[0]))
                self.layers.append(self.activation)

                for i in range(1, len(hidden_dims)):
                    self.layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
                    self.layers.append(self.activation)

                self.layers.append(GCNConv(hidden_dims[-1], graph_config.output_dim))

            elif graph_config.layer_type == GraphLayerType.GATConv:
                # Стек GATConv с num_heads головами; concat=False → финальный dim = out_channels
                self.num_heads = num_heads = graph_config.gat_props.num_heads
                self.layers.append(
                    GATConv(input_dim, hidden_dims[0], heads=num_heads, concat=False)
                )
                self.layers.append(self.activation)

                for i in range(1, len(hidden_dims)):
                    self.layers.append(
                        GATConv(
                            hidden_dims[i - 1],
                            hidden_dims[i],
                            heads=num_heads,
                            concat=False,
                        )
                    )
                    self.layers.append(self.activation)

                self.layers.append(
                    GATConv(
                        hidden_dims[-1],
                        graph_config.output_dim,
                        heads=num_heads,
                        concat=False,
                    )
                )
            elif graph_config.layer_type == GraphLayerType.SparseGATConv:
                # Один SparseGATConv: часто хватает одного слоя внимания + (опц.) LayerNorm
                self.num_heads = num_heads = graph_config.gat_props.num_heads
                print(graph_config.layer_type)
                self.layers.append(
                    SparseGATConv(input_dim, graph_config.output_dim, heads=num_heads, concat=False)
                )

            # (Опционально) LayerNorm поверх выходного пространства
            if graph_config.use_layer_norm:
                self.layers.append(
                    LayerNorm(
                        in_channels=graph_config.output_dim,
                        mode=graph_config.layer_norm_mode,
                    )
                )

        elif graph_config.layer_type == GraphLayerType.InteractionNet:
            # InteractionNetProcessor: N шагов message passing с edge features + residuals
            self.output_dim = graph_config.output_dim
            num_steps = graph_config.num_message_passing_steps or 4
            raw_edge_dim = graph_config.edge_feature_dim or 4  # distance + 3D relative position
            activation = graph_config.activation or "swish"
            use_ln = graph_config.use_layer_norm if graph_config.use_layer_norm is not None else True

            # Residual connections требуют output_dim == input_dim
            assert graph_config.output_dim == input_dim, (
                f"InteractionNet requires output_dim ({graph_config.output_dim}) == input_dim ({input_dim}), "
                f"так как используются residual connections."
            )

            self.layers = InteractionNetProcessor(
                node_dim=input_dim,  # должен совпадать с output_dim (latent)
                raw_edge_dim=raw_edge_dim,
                edge_latent_dim=input_dim,  # edge latent = node latent (как в GraphCast)
                hidden_dim=input_dim,       # hidden = latent (как в GraphCast)
                num_steps=num_steps,
                activation=activation,
                use_layer_norm=use_ln,
            )

        else:
            print(graph_config.layer_type)
            raise NotImplementedError(
                f"Layer type {graph_config.layer_type} not supported."
            )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, attention_threshold=0.0, **kwargs):
        """Единый интерфейс прямого прохода для разных типов слоёв.

        - SimpleConv: просто применяем слой к (X, edge_index).
        - GCNConv/GATConv: пробегаем по self.layers, где conv-слои чередуются с активациями.
        - SparseGATConv: возвращаем (X, edge_index) — так как edge_index может обновиться (прореживание).
        """
        if self.layer_type == GraphLayerType.SimpleConv:
            return self.layers(x=X, edge_index=edge_index)

        elif self.layer_type == GraphLayerType.ConvGCN:
            for layer in self.layers:
                if type(layer) == GCNConv:
                    X = layer(X, edge_index)
                else:
                    X = layer(X)
        elif self.layer_type == GraphLayerType.GATConv:
            for layer in self.layers:
                if type(layer) == GATConv:
                    X = layer(X, edge_index)
                else:
                    X = layer(X)
        elif self.layer_type == GraphLayerType.SparseGATConv:
            for layer in self.layers:
                if type(layer) == SparseGATConv:
                    X, (edge_index, _) = layer.forward(X, edge_index, attention_threshold, **kwargs)
                else:
                    X = layer(X)
            return X, edge_index
        elif self.layer_type == GraphLayerType.InteractionNet:
            edge_attr = kwargs.get("edge_attr", None)
            if edge_attr is None:
                raise ValueError("InteractionNet requires edge_attr (edge features)")
            return self.layers(x=X, edge_index=edge_index, edge_attr_raw=edge_attr)
        return X


class Model(nn.Module):
    """Композиция «(опц.) MLP → GraphLayer».

    Зачем:
    - MLP выравнивает/подготавливает каналы (например, склеенные динамические + статические фичи) → аккуратный вход в GNN.
    - GraphLayer делает собственно message passing.

    Выставляет self.output_dim, чтобы следующий блок знал размерность входа.
    """
    def __init__(self, model_config: ModelConfig, input_dim: int):
        super().__init__()
        self.mlp = None
        self.output_dim = None
        graph_input_dim = input_dim
        if model_config.mlp:
            self.mlp = MLP(mlp_config=model_config.mlp, input_dim=input_dim)
            graph_input_dim = model_config.mlp.output_dim

        self.graph_layer = GraphLayer(
            graph_config=model_config.gcn, input_dim=graph_input_dim
        )
        self.output_dim = self.graph_layer.output_dim

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, attention_threshold=0.0, **kwargs):

        if self.mlp:
            X = self.mlp(X=X)

        out = self.graph_layer(X=X, edge_index=edge_index, attention_threshold=attention_threshold, **kwargs)

        return out


class WeatherPrediction(nn.Module):
    """Главная модель прогноза погоды (а-ля GraphCast).

    Работает с тремя графами:
    * Encoding graph (Grid→Mesh): бипаритный граф, переносит признаки с grid на mesh; на выходе — латенты для mesh-узлов.
    * Processing graph (Mesh↔Mesh): внутри-треугольные рёбра, обновляет латенты mesh.
    * Decoding graph (Mesh→Grid): бипаритный граф, переносит обновлённые mesh-латенты обратно на grid (каждый grid-узел связан с 3 вершинами покрывающего треугольника mesh).

    Опционально: продукт-граф (время×пространство) поверх входов grid для обогащения последнего временного среза.
    """

    def __init__(
        self,
        cordinates: Tuple[np.array, np.array],
        graph_config: GraphBuildingConfig,
        pipeline_config: PipelineConfig,
        data_config: DataConfig,
        device,
    ):
        super().__init__()

        self.device = device
        self.obs_window = data_config.obs_window_used  # T — окно наблюдений (временные шаги)
        self.num_features = data_config.num_features_used  # F — число динамических фичей на один шаг
        self.total_feature_size = self.num_features * self.obs_window  # T*F — если склеиваем время в канал
        self.use_product_graph = pipeline_config.product_graph is not None

        # Инициализация свойств grid и mesh (координаты, числа узлов, иерархия сеток)
        self._init_grid_properties(grid_lat=cordinates[0], grid_lon=cordinates[1])
        self._init_mesh_properties(graph_config)
        self.using_sparse_gat = pipeline_config.processor.gcn.layer_type == GraphLayerType.SparseGATConv

        self._total_nodes = self._num_grid_nodes + self._num_mesh_nodes

        # (Опционально) создаём продукт-граф (время×пространство) и маленькую модель для прохода по нему.
        if self.use_product_graph:
            self.product_graph = self._create_product_graph(
                product_graph_config=pipeline_config.product_graph
            ).to(self.device)
            self.product_graph_model = Model(
                model_config=pipeline_config.product_graph.model,
                input_dim=self.num_features,
            ).to(self.device)

        # Строим ENCODING-граф и считаем статические фичи для grid/mesh узлов
        self.encoding_graph, self.init_grid_features, self.init_mesh_features = (
            create_encoding_graph(
                grid_node_lats=self._grid_lat,
                grid_node_longs=self._grid_lon,
                mesh_node_lats=self._mesh_nodes_lat,
                mesh_node_longs=self._mesh_nodes_lon,
                mesh=self._finest_mesh,
                graph_building_config=graph_config,
                num_grid_nodes=self._num_grid_nodes,
            )
        )

        # Переносим статические фичи на устройство
        self.init_grid_features, self.init_mesh_features = self.init_grid_features.to(
            device
        ), self.init_mesh_features.to(device)

        # Размер статических фичей на каждом узле (одинаковый для grid/mesh)
        self._init_feature_size = self.init_grid_features.shape[1]

        # PROCESSING-граф: рёбра внутри mesh для message passing
        self.using_interaction_net = pipeline_config.processor.gcn.layer_type == GraphLayerType.InteractionNet

        proc_graph_result = create_processing_graph(
            meshes=self._meshes, mesh_levels=graph_config.mesh_levels,
            mesh_node_lats=self._mesh_nodes_lat,
            mesh_node_longs=self._mesh_nodes_lon,
        )
        if isinstance(proc_graph_result, tuple):
            self.processing_graph, proc_edge_features = proc_graph_result
            self.register_buffer("_processing_edge_features", proc_edge_features)
        else:
            self.processing_graph = proc_graph_result
            self._processing_edge_features = None

        # DECODING-граф: для каждого grid — 3 входа от вершин треугольника mesh, который его содержит
        self.decoding_graph = create_decoding_graph(
            cordinates=cordinates,
            mesh=self._finest_mesh,
            graph_building_config=graph_config,
            num_grid_nodes=self._num_grid_nodes,
        )

        # Размер входа в ENCODER:
        # - Если используем продукт-граф, в ENCODER идёт последний временной срез с F каналами.
        # - Иначе склеиваем T шагов во вход: T*F.
        encoder_input_dim = (
            self.num_features + self._init_feature_size
            if self.use_product_graph
            else self.total_feature_size + self._init_feature_size
        )
        self.encoder = Model(
            model_config=pipeline_config.encoder, input_dim=encoder_input_dim
        ).to(device)

        # PROCESSOR: работает по mesh, получает на вход encoder.output_dim
        self.processor = Model(
            model_config=pipeline_config.processor,
            input_dim=self.encoder.output_dim,
        ).to(device)

        # DECODER: работает по объединённым (grid+mesh) признакам, получает processor.output_dim
        self.decoder = Model(
            model_config=pipeline_config.decoder,
            input_dim=self.processor.output_dim,
        ).to(device)

        # Переносим графы на устройство
        self.encoding_graph, self.decoding_graph, self.processing_graph = (
            self.encoding_graph.to(self.device),
            self.decoding_graph.to(device),
            self.processing_graph.to(device),
        )

        # Печать краткого summary для отладки размеров и графов
        if self.use_product_graph:
            print("Product Graph summary: ")
            print(
                summary(
                    self.product_graph_model,
                    torch.randn(
                        self._num_grid_nodes * self.obs_window, self.num_features
                    ).to(device),
                    self.product_graph,
                )
            )
            print()

        print("Encoder summary: ")
        print(
            summary(
                self.encoder,
                torch.randn(
                    self._num_grid_nodes + self._num_mesh_nodes, encoder_input_dim
                ).to(device),
                self.encoding_graph,
            )
        )
        print()

        print("Processor summary: ")
        try:
            proc_kwargs = {}
            if self._processing_edge_features is not None:
                proc_kwargs["edge_attr"] = self._processing_edge_features
            print(
                summary(
                    self.processor,
                    torch.randn(self._num_mesh_nodes, self.encoder.output_dim).to(device),
                    self.processing_graph,
                    **proc_kwargs,
                )
            )
        except Exception as e:
            print(f"  (summary skipped: {e})")
        print()

        print("Decoder summary: ")
        print(
            summary(
                self.decoder,
                torch.randn(
                    self._num_grid_nodes + self._num_mesh_nodes,
                    self.processor.output_dim,
                ).to(device),
                self.decoding_graph,
            )
        )
        print()

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Сохраняем координаты grid и считаем число узлов N = |lat| × |lon|."""
        self._grid_lat = grid_lat.astype(np.float32)
        self._grid_lon = grid_lon.astype(np.float32)
        self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

    def _init_mesh_properties(self, graph_config: GraphBuildingConfig):
        """Строим иерархию сеток на сфере и извлекаем координаты узлов самого тонкого уровня."""
        self._meshes = get_hierarchy_of_triangular_meshes_for_sphere(
            splits=max(graph_config.mesh_levels)
        )
        self._finest_mesh = self._meshes[-1]
        self._num_mesh_nodes = len(self._finest_mesh.vertices)

        self._mesh_nodes_lat, self._mesh_nodes_lon = get_mesh_lat_long(
            finest_mesh=self._finest_mesh
        )

        self._mesh_nodes_lat, self._mesh_nodes_lon = self._mesh_nodes_lat.astype(
            np.float32
        ), self._mesh_nodes_lon.astype(np.float32)

    def _create_product_graph(self, product_graph_config: ProductGraphConfig):
        """Создаёт граф на декартовом произведении времени (T) и пространства (grid) — product-graph.

        Идея:
        - Время: простой ориентированный «цепочка» граф (i → i+1).
        - Пространство: KNN-граф по координатам (lat, lon) всех grid-точек.
        - Итоговая матрица смежности — линейная комбинация кронекеровых произведений:

          s00 * (I_T ⊗ I_N)   +   s01 * (I_T ⊗ A_space)   +   s10 * (A_time ⊗ I_N)   +   s11 * (A_time ⊗ A_space)

          где коэффициенты (s00,s01,s10,s11) зависят от типа графа: KRONECKER / CARTESIAN / STRONG.

        Возвращает edge_index в формате PyG.
        """

        def _construct_temporal_graph(T):
            # Простой цепочечный ориентированный граф по времени: i → i+1
            temporal_graph = np.zeros((T, T))
            for i in range(T - 1):
                temporal_graph[i, i + 1] = 1
            return temporal_graph

        def _construct_adjacency_matrix(grid_lat, grid_lon, k):
            # Формируем список всех grid-точек (lat, lon) → строим KNN-граф по евклидовой метрике
            lat_lon_grid = np.array(
                [[lat, lon] for lat in grid_lat for lon in grid_lon]
            )
            adjacency = kneighbors_graph(
                lat_lon_grid,
                n_neighbors=k,
                mode="connectivity",
                include_self=False,
            ).toarray()

            # NOTE: при необходимости можно проверить/перекинуть транспонирование adjacency,
            # если ожидается другой порядок (отправитель/получатель) — сейчас используем как есть.
            return adjacency

        T = self.obs_window
        N = self._num_grid_nodes
        s00 = 0  # всегда 0 в текущей схеме (см. ниже)
        if product_graph_config.type == ProductGraphType.KRONECKER:
            s00, s01, s10, s11 = s00, 0, 0, 1
        elif product_graph_config.type == ProductGraphType.CARTESIAN:
            s00, s01, s10, s11 = s00, 1, 1, 0
        elif product_graph_config.type == ProductGraphType.STRONG:
            s00, s01, s10, s11 = s00, 1, 1, 1

        temporal_graph = _construct_temporal_graph(T)

        adjacency = _construct_adjacency_matrix(
            grid_lat=self._grid_lat,
            grid_lon=self._grid_lon,
            k=product_graph_config.num_k,
        )

        # Комбинируем слагаемые продукт-графа через Кронекер
        product_graph = (
            s00 * np.kron(np.eye(T), np.eye(N))
            + s01 * np.kron(np.eye(T), adjacency)
            + s10 * np.kron(temporal_graph, np.eye(N))
            + s11 * np.kron(temporal_graph, adjacency)
        )

        # Переводим плотную матрицу смежности в edge_index
        edge_index, _ = dense_to_sparse(torch.tensor(product_graph, dtype=torch.float))

        return edge_index

    def _preprocess_input(self, grid_node_features: torch.Tensor):
        """Готовит единый вход X для ENCODER:
        - Конкатенирует динамические входы grid с их статическими фичами.
        - Инициализирует нулями динамику для mesh и конкатенирует со статикой mesh.
        - Склеивает [grid; mesh] вдоль оси узлов → единый тензор X формы [(N+M), C].
        """
        # Concatenate the initial grid node features with the incoming input
        updated_grid_node_features = torch.cat(
            (grid_node_features, self.init_grid_features), dim=-1
        )

        total_feature_size = (
            self.num_features if self.use_product_graph else self.total_feature_size
        )

        # Initialise the mesh node features to 0s and append the initial mesh features
        mesh_node_features = torch.zeros(
            (
                self._num_mesh_nodes,
                total_feature_size,
            )
        ).to(self.device)

        updated_mesh_node_features = torch.cat(
            (mesh_node_features, self.init_mesh_features), dim=-1
        )

        # Concatenate them into one single tensor so that they can be passed through graph layers
        X = torch.cat((updated_grid_node_features, updated_mesh_node_features), dim=0)

        return X

    def forward(self, X: torch.Tensor, attention_threshold, **kwargs):
        """Основной прямой проход:
        1) (опц.) Product-graph: прогоняем T×N входов через маленькую GNN и берём последний временной срез (N×F).
        2) _preprocess_input: добавляем статику и дополняем mesh нулями → общий X для ENCODER.
        3) ENCODER (Grid→Mesh): получаем латенты и делим на grid-часть и mesh-часть.
        4) PROCESSOR (Mesh↔Mesh): обновляем mesh-латенты. Если SparseGAT — можем проредить processing_graph.
        5) DECODER (Mesh→Grid): склеиваем обратно [grid; processed_mesh] и получаем выход только по grid.

        Параметры
        ----------
        X : torch.Tensor
          Входные данные формы [batch, num_grid_nodes, num_features] (часто batch=1). Внутри сплющивается.
        """

        X = X.squeeze()  # ожидается batch=1; делаем [N, T*F] или [N, F]
        if self.use_product_graph:
            # Приводим к форме [T*N, F] и прогоняем через модель на продукт-графе
            X = X.view(self._num_grid_nodes * self.obs_window, self.num_features)
            X = self.product_graph_model(X=X, edge_index=self.product_graph)
            # Берём только последний временной срез (последние N узлов)
            X = X[-self._num_grid_nodes :, :]

        # Подготовка общего X с добавлением статических фичей
        X = self._preprocess_input(grid_node_features=X)

        # ENCODER: работает на бипартитном графе Grid→Mesh
        encoded_features = self.encoder.forward(X=X, edge_index=self.encoding_graph)

        # Разделяем на grid и mesh части
        grid_node_features = encoded_features[: self._num_grid_nodes, :]
        mesh_node_features = encoded_features[self._num_grid_nodes :, :]

        # PROCESSOR: обновляем mesh латенты
        if self.using_sparse_gat:
            processed_mesh_node_features, new_processor_edge_index = self.processor.forward(
                X=mesh_node_features, edge_index=self.processing_graph, attention_threshold=attention_threshold, **kwargs
            )
            # ВАЖНО: сохраняем новый прореженный граф для следующих проходов
            self.processing_graph = new_processor_edge_index
        elif self.using_interaction_net:
            # InteractionNet: передаём edge features
            processed_mesh_node_features = self.processor.forward(
                X=mesh_node_features, edge_index=self.processing_graph,
                attention_threshold=attention_threshold,
                edge_attr=self._processing_edge_features,
            )
        else:
            processed_mesh_node_features = self.processor.forward(
                X=mesh_node_features, edge_index=self.processing_graph, attention_threshold=attention_threshold
            )

        # Склеиваем обратно grid + обновлённый mesh для DECODER
        processed_features = torch.cat(
            (grid_node_features, processed_mesh_node_features), dim=0
        )

        # DECODER: Mesh→Grid бипарит, на выходе берём только grid-узлы
        decoded_grid_node_features = self.decoder.forward(
            X=processed_features,
            edge_index=self.decoding_graph,
        )

        decoded_grid_node_features = decoded_grid_node_features[
            : self._num_grid_nodes, :
        ]

        return decoded_grid_node_features
