import torch
import torch.nn as nn
from src.config import SimpleAggregationEncoder
from torch_geometric.data import Data

class AggregationEncoder(nn.Module):
    """The encoder creates the latent representation of the mesh nodes from the grid nodes.
    
    The AggregationEncoder simply aggregates nodes connected to a mesh node.
    
    """
    
    def __init__(self, encoder_config: SimpleAggregationEncoder, num_mesh_nodes: int, mesh_embedding_size: int):
        self.encoder_config = encoder_config
        self.num_mesh_nodes = num_mesh_nodes
        self.mesh_embedding_size = mesh_embedding_size
    
    def forward(self, grid_node_features: torch.Tensor, edge_index: torch.Tensor):
        """Calculates the latent representation of the mesh nodes by aggregating the grid nodes that connect to the mesh nodes.
        Output shape is [batch, num_mesh_nodes, mesh_embedding_size]

        Parameters
        ----------
        grid_node_features : torch.Tensor
            This has shape [batch, num_grid_nodes, num_features]. These are the features for each grid node for a batch.
        edge_index : torch.Tensor
            This has a shape [batch, num_edges, 2]. This represents the edges between the grid and the mesh nodes for each batch.
        """
        pass
    