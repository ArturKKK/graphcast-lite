"""Defines the main Weather Prediction pytorch model."""

import torch
import torch.nn as nn
import numpy as np
from src.config import GraphBuildingConfig, ModelConfig
from src.mesh.create_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    filter_mesh,
    get_edges_from_faces,
)
from src.graph import create as graph_create
from typing import Optional, Tuple
from .utils import (
    get_encoder_from_encoder_config,
    get_processor_from_process_config,
    get_decoder_from_decode_config,
)


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
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        self.num_grid_nodes = len(cordinates[0]) * len(cordinates[1])

        self._meshes = get_hierarchy_of_triangular_meshes_for_sphere(
            splits=graph_config.mesh_size
        )

        self.grid2mesh_graph = graph_create.create_grid_to_mesh_graph(
            cordinates=cordinates,
            mesh=self._meshes[-1],
            graph_building_config=graph_config,
        )  # TODO edge array should be [2, num_edges] for torch-geometric. We should have the same here, for sake of consistency.

        self.mesh_we_want = filter_mesh(self._meshes, graph_config.mesh_level)
        self.mesh_edeges = torch.tensor(get_edges_from_faces(self.mesh_we_want.faces))

        self.mesh2grid_graph = graph_create.create_mesh_to_grid_graph(
            cordinates=cordinates,
            mesh=self.mesh_we_want,
            graph_building_config=graph_config,
        )

        self.encoder = get_encoder_from_encoder_config(
            encoder_config=model_config.encoder,
            num_mesh_nodes=self._meshes[-1].vertices.shape[0],
        )

        self.processor = get_processor_from_process_config(
            process_config=model_config.processor
        )

        self.decoder = get_decoder_from_decode_config(
            decoder_config=model_config.decoder, num_grid_nodes=self.num_grid_nodes
        )

    @staticmethod
    def _create_grid_lat_lon_cordinates(lats: np.array, longs: np.array):
        """This is what GraphCast does to they latitude and longitudes. Skipping this for now and using the original
        latitudes and longitudes directly."""
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(lats, longs)
        grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

        return grid_nodes_lat, grid_nodes_lon

    def forward(self, X: torch.Tensor):
        """The forward method takes the features of the grid nodes and passes them through the three graphs defined above.
        Grid2Mesh performs the encoding and calculates the

        Parameters
        ----------
        X : torch.Tensor
          The input data of the shape [batch, num_grid_nodes, num_features].
        """

        mesh_node_features = self.encoder(
            grid_node_features=X,
            edge_index=self.grid2mesh_graph,  # TODO edge_index should be a tensor for processor as torch-geometric is used. We should have the same here, for sake of consistency.
        )

        processed_mesh_node_features = self.processor(
            mesh_node_features, edge_index=self.mesh_edeges
        )

        decoded_grid_node_features = self.decoder(
            mesh_node_features=processed_mesh_node_features,
            edge_index=self.mesh2grid_graph,
        )

        return decoded_grid_node_features
