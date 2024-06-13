from typing import Tuple

import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, SimpleConv
import numpy as np

from src.config import (
    ModelConfig,
    MLPBlock,
    GraphBlock,
    GraphLayerType,
    GraphBuildingConfig,
    DataConfig,
    PipelineConfig,
)
from src.create_graphs import (
    create_decoding_graph,
    create_processing_graph,
    create_encoding_graph,
)


from src.mesh.create_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
)


class MLP(nn.Module):
    def __init__(self, mlp_config: MLPBlock, input_dim):
        super().__init__()
        hidden_dims = mlp_config.mlp_hidden_dims
        output_dim = mlp_config.output_dim

        self.MLP = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_dim,
                    out_features=hidden_dims[0],
                ),
                nn.ReLU(),
            ]
        )

        for h_index in range(1, len(hidden_dims)):
            self.MLP.extend(
                [
                    nn.Linear(
                        in_features=hidden_dims[h_index - 1],
                        out_features=hidden_dims[h_index],
                    ),
                    nn.ReLU(),
                ]
            )

        self.MLP.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))

    def forward(self, X: torch.Tensor):
        for layer in self.MLP:
            X = layer(X)

        return X


class GraphLayer(nn.Module):
    def __init__(self, graph_config: GraphBlock, input_dim):
        super().__init__()

        self.layer_type: GraphLayerType = graph_config.layer_type
        self.output_dim = None

        if graph_config.layer_type == GraphLayerType.SimpleConv:
            self.output_dim = input_dim
            self.layers = SimpleConv()

        elif graph_config.layer_type == GraphLayerType.ConvGCN:
            self.output_dim = graph_config.output_dim
            self.layers = torch.nn.ModuleList()
            hidden_dims = graph_config.hidden_dims

            self.layers.append(GCNConv(input_dim, hidden_dims[0]))
            for i in range(1, len(hidden_dims)):
                self.layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))

            self.layers.append(GCNConv(hidden_dims[-1], graph_config.output_dim))
            self.activation = torch.nn.ReLU()

        else:
            raise NotImplementedError(
                f"Layer type {graph_config.layer_type} not supported."
            )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor):
        if self.layer_type == GraphLayerType.SimpleConv:
            return self.layers(x=X, edge_index=edge_index)

        elif self.layer_type == GraphLayerType.ConvGCN:
            for layer in self.layers[:-1]:
                X = self.activation(layer(X, edge_index))

            X = self.layers[-1](X, edge_index)

        return X


class Model(nn.Module):
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

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor):

        if self.mlp:
            X = self.mlp(X=X)

        X = self.graph_layer(X=X, edge_index=edge_index)

        return X


class WeatherPrediction(nn.Module):
    """This is our main weather prediction model. Similar to GraphCast, this model will
      operate on three graphs -

      * Grid2Mesh graph: This graph contains all the nodes. This graph is strictly
      bipartite with edges going from grid nodes to the mesh nodes.The encoder model will
      operate on this graph. The output of this stage will be a latent representation for
      the mesh nodes.

    * Mesh graph: This graph contains only the mesh nodes. The process model will
      operate on this graph. It will update the latent state of the mesh nodes.

    * Mesh2Grid graph: This graph contains all nodes. This graph is strictly
      bipartite with edges going from mesh nodes to grid nodes such that each grid
      nodes is connected to 3 nodes of the mesh triangular face that contains
      the grid points. The decode model will operate on this graph. It will
      process the updated latent state of the mesh nodes, and the latent state
      of the grid nodes, to produce the final output for the grid nodes.
    """

    def __init__(
        self,
        cordinates: Tuple[np.array, np.array],
        graph_config: GraphBuildingConfig,
        pipeline_config: PipelineConfig,
        data_config: DataConfig,
    ):
        super().__init__()

        self.timesteps = data_config.num_timesteps
        self.num_features = data_config.num_features
        self.total_feature_size = self.timesteps * self.num_features

        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(
            splits=graph_config.mesh_size
        )
        finest_mesh = _meshes[-1]

        self.num_grid_nodes = len(cordinates[0]) * len(cordinates[1])
        self.num_mesh_nodes = finest_mesh.vertices.shape[0]

        self.encoding_graph = create_encoding_graph(
            cordinates=cordinates,
            mesh=finest_mesh,
            graph_building_config=graph_config,
        )

        self.processing_graph = create_processing_graph(
            meshes=_meshes, mesh_levels=graph_config.mesh_levels
        )

        self.decoding_graph = create_decoding_graph(
            cordinates=cordinates,
            mesh=finest_mesh,
            graph_building_config=graph_config,
        )

        self.encoder = Model(
            model_config=pipeline_config.encoder, input_dim=self.total_feature_size
        )
        

        self.processor = Model(
            model_config=pipeline_config.processor, input_dim=self.encoder.output_dim
        )

        self.decoder = Model(
            model_config=pipeline_config.decoder, input_dim=self.processor.output_dim
        )

    def forward(self, X: torch.Tensor):
        """The forward method takes the features of the grid nodes and passes them through the three graphs defined above.
        Grid2Mesh performs the encoding and calculates the

        Parameters
        ----------
        X : torch.Tensor
          The input data of the shape [batch, num_grid_nodes, num_features].
        """

        mesh_node_features = self.encoder(X=X, edge_index=self.encoding_graph)

        processed_mesh_node_features = self.processor(
            X=mesh_node_features, edge_index=self.processing_graph
        )

        decoded_grid_node_features = self.decoder(
            X=processed_mesh_node_features,
            edge_index=self.decoding_graph,
        )
        
        return decoded_grid_node_features
