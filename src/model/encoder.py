import torch
import torch.nn as nn
from src.config import AggregationEncoderConfig


class AggregationEncoder(nn.Module):
    """The encoder creates the latent representation of the mesh nodes from the grid nodes.

    The AggregationEncoder simply aggregates nodes connected to a mesh node.

    """

    def __init__(
        self,
        encoder_config: AggregationEncoderConfig,
        num_grid_nodes: int,
        num_mesh_nodes: int,
    ):
        self.encoder_config = encoder_config
        self.num_mesh_nodes = num_mesh_nodes
        self.num_grid_nodes: num_grid_nodes

    def forward(self, grid_node_features: torch.Tensor, edge_index: torch.Tensor):
        """Calculates the latent representation of the mesh nodes by aggregating the grid nodes that connect to the mesh nodes.
        Output shape is [batch, num_mesh_nodes, mesh_embedding_size]. Here, mesh_embedding_size is the same as the grid_embedding_size.

        Parameters
        ----------
        grid_node_features : torch.Tensor
            This has shape [batch, num_grid_nodes, grid_embedding_size].
        edge_index : torch.Tensor
            This has a shape [num_edges, 2]. This represents the edges between the grid and the mesh nodes for each batch.
        """
        pass
