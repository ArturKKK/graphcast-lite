import torch


def get_adjacency_matrix_from_edge_index(
    edge_index: torch.Tensor,
    num_sender_nodes: int,
    num_receiver_nodes: int,
):
    """Converts the edge_index array of shape (num_edges, 2) into an adjacency matrix of shape [num_sender_nodes, num_receiver_nodes] where
    the edge indices in edge_index array are marked as 1 in the adjacency_matrix.
    
    Note: The edges are unidirectional. To add bidirectional edges, add both sender to receiver and receiver to sender edges in the edge_index.

    Parameters
    ----------
    edge_index : torch.Tensor
        The edge tensor of shape (num_edges, 2).
    num_sender_nodes : int
        Number of sender nodes.
    num_receiver_nodes : int
        Number of receiver nodes.
    """
    
    adjacency_matrix = torch.zeros((num_sender_nodes, num_receiver_nodes))
    adjacency_matrix[edge_index[:, 0], edge_index[:, 1]] = 1
    
    return adjacency_matrix
    
    
