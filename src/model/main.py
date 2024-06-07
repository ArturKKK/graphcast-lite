"""Defines the main Weather Prediction pytorch model."""

import torch
import torch.nn as nn
import numpy as np
from src.config import GraphBuildingConfig, ModelConfig, Grid2MeshEdgeCreation
from src.mesh.create_mesh import get_hierarchy_of_triangular_meshes_for_sphere
from src.graph import create as graph_create
from typing import Optional, Tuple
from utils import (
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
        num_grid_nodes: int,
        model_config: Optional[ModelConfig] = None,
    ):
        self._meshes = get_hierarchy_of_triangular_meshes_for_sphere(
            splits=graph_config.mesh_size
        )

        self.grid2mesh_graph_index = graph_create.create_grid_to_mesh_graph(
            cordinates=cordinates,
            mesh=self._meshes[-1],
            graph_building_config=graph_config,
        )

        self.mesh2mesh_graph = graph_create.create_mesh_to_mesh_graph(
            graph_building_config=graph_config
        )

        self.mesh2grid_graph = graph_create.create_mesh_to_grid_graph(
            graph_building_config=graph_config
        )

        self.encoder = get_encoder_from_encoder_config(
            encoder_config=model_config.encoder,
            num_grid_nodes=num_grid_nodes,
            num_mesh_nodes=len(self._meshes[-1]),
        )

        self.processor = get_processor_from_process_config(
            process_config=model_config.processor
        )

        self.decoder = get_decoder_from_decode_config(
            decoder_config=model_config.decoder
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
          The input data of the shape [batch, num_grid_nodes, num_features, timesteps].
        """
        pass


if __name__ == "__main__":

    num_grid_nodes = 100
    lats = np.random.uniform(low=-1, high=1, size=(num_grid_nodes,))
    longs = np.random.uniform(low=-1, high=1, size=(num_grid_nodes,))

    graph_config = GraphBuildingConfig(
        resolution=0.25,
        mesh_size=1,
        grid2mesh_radius_query=0.5,
        grid2mesh_edge_creation=Grid2MeshEdgeCreation.RADIUS,
    )

    model = WeatherPrediction(cordinates=(lats, longs), graph_config=graph_config)
