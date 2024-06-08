import torch
import torch.nn as nn
from src.config import AggregationDecoderConfig
from src.utils import get_adjacency_matrix_from_edge_index


class AggregationDecoder(nn.Module):
    """The decoder creates the retreives the grid nodes back from the mesh nodes

    The AggregationDecoder simply aggregates the mesh nodes that are connected to a grid node to create the
    final grid node representation.
    """

    def __init__(
        self,
        decoder_config: AggregationDecoderConfig,
        num_grid_nodes: int,
    ):
        super().__init__()
        self.decoder_config = decoder_config
        self.num_grid_nodes = num_grid_nodes

    def forward(self, mesh_node_features: torch.Tensor, edge_index: torch.Tensor):
        """Calculates the final representation of the grid nodes by aggregating the mesh nodes that connect to the grid nodes.
        Output shape is [batch, num_grid_nodes, grid_embedding_size]. Here, grid_embedding_size is the same as the mesh_embedding_size.

        Parameters
        ----------
        mesh_node_features : torch.Tensor
            This has shape [batch, num_mesh_nodes, mesh_embedding_size].
        edge_index : torch.Tensor
            This has a shape [num_edges, 2]. This represents the edges between the mesh and the grid nodes for each batch.
        """

        batch_size, num_mesh_nodes, _ = mesh_node_features.shape

        # Convert edge_index to adjacency matrix
        adjacency_matrix = get_adjacency_matrix_from_edge_index(
            edge_index=edge_index,
            num_sender_nodes=num_mesh_nodes,
            num_receiver_nodes=self.num_grid_nodes,
        )

        # Normalize the adjacency matrix to average the contributions from each connected grid node
        normalization_factor = adjacency_matrix.sum(dim=0, keepdim=True)
        normalization_factor[normalization_factor == 0] = 1  # To avoid division by zero
        adjacency_matrix = adjacency_matrix / normalization_factor

        adjacency_matrix = (
            adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2)
        )

        grid_node_features = torch.bmm(adjacency_matrix, mesh_node_features)

        return grid_node_features
