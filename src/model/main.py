"""Defines the main Weather Prediction pytorch model."""

import torch.nn as nn
from src.config import GraphBuildingConfig, ModelConfig
from src.mesh.create_mesh import get_hierarchy_of_triangular_meshes_for_sphere
from typing import Optional
from torch_geometric.data import Data


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
        graph_config: GraphBuildingConfig,
        model_config: Optional[ModelConfig] = None,
    ):
        self._meshes = get_hierarchy_of_triangular_meshes_for_sphere(
            splits=graph_config.mesh_size
        )
    
    
    def forward(self, data: Data):
      pass


if __name__ == "__main__":
    graph_config = GraphBuildingConfig(
        resolution=0.25, mesh_size=1, radius_query_fraction_edge_length=0.5
    )
    
    model = WeatherPrediction(graph_config=graph_config)
