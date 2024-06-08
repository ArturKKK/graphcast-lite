import torch
import torch.nn as nn
from src.config import AggregationEncoderConfig
from src.utils import get_adjacency_matrix_from_edge_index


class AggregationEncoder(nn.Module):
    """The encoder creates the latent representation of the mesh nodes from the grid nodes.

    The AggregationEncoder simply aggregates nodes connected to a mesh node.

    """

    def __init__(
        self,
        encoder_config: AggregationEncoderConfig,
        num_mesh_nodes: int,
    ):
        super().__init__()
        self.encoder_config = encoder_config
        self.num_mesh_nodes = num_mesh_nodes

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

        batch_size, num_grid_nodes, _ = grid_node_features.shape

        # Convert edge_index to adjacency matrix
        adjacency_matrix = get_adjacency_matrix_from_edge_index(
            edge_index=edge_index,
            num_sender_nodes=num_grid_nodes,
            num_receiver_nodes=self.num_mesh_nodes,
        )

        # Normalize the adjacency matrix to average the contributions from each connected grid node
        normalization_factor = adjacency_matrix.sum(dim=0, keepdim=True)
        normalization_factor[normalization_factor == 0] = 1  # To avoid division by zero
        adjacency_matrix = adjacency_matrix / normalization_factor

        adjacency_matrix = (
            adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2)
        )

        mesh_node_features = torch.bmm(adjacency_matrix, grid_node_features)

        return mesh_node_features
