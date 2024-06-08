import torch
from torch_geometric.nn import GCNConv

class SimpleProcessor(torch.nn.Module):
    """

    """

    def __init__(self, in_out_dim, hidden_dims):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_out_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(GCNConv(hidden_dims[-1], in_out_dim))

        self.activation = torch.nn.ReLU()

    def forward(self, mesh_node_features: torch.Tensor, edge_index: torch.Tensor):
        """The forward method takes the features of the mesh nodes and passes them through the mesh2mesh graph.
        The output is the processed mesh node features.

        Parameters
        ----------
        mesh_node_features : torch.Tensor
            This has shape [batch, num_mesh_nodes, num_features].
        edge_index : torch.Tensor
            This has a shape [num_edges, 2]. This represents the edges between the mesh nodes.
        """
        for layer in self.layers:
            mesh_node_features = self.activation(layer(mesh_node_features, edge_index))
        return mesh_node_features