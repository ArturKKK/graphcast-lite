import torch
from torch_geometric.nn import GCNConv
from typing import List


class SimpleProcessor(torch.nn.Module):
    """ """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Parameters
        ----------

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(GCNConv(hidden_dims[-1], output_dim))

        self.activation = torch.nn.ReLU()

    def forward(self, mesh_node_features: torch.Tensor, edge_index: torch.Tensor):
        """The forward method takes the features of the mesh nodes and passes them through the mesh2mesh graph.
        The output is the processed mesh node features.

        Parameters
        ----------
        mesh_node_features : torch.Tensor
            This has shape [batch, num_mesh_nodes, num_features].
        edge_index : torch.Tensor
            This has a shape [2, num_edges]. This represents the edges between the mesh nodes.
        """
        for layer in self.layers:
            mesh_node_features = self.activation(layer(mesh_node_features, edge_index))
        return mesh_node_features
