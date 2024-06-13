import torch
import json
from typing import Dict, Any


def save_to_json_file(data_dict: Dict[str, Any], save_path: str):
    """Dumps a dictionary into JSON

    Args:
        data_dict (Dict[str, Any]): The dictionary to save into a JSON file.
        save_path (str): The path to save the JSON to.
    """
    with open(save_path, "w") as outfile:
        json.dump(data_dict, outfile)


def load_from_json_file(data_path: str) -> Dict[str, Any]:
    """Loads data from a JSON file.

    Args:
        data_path (str): Path to load data from

    Returns:
        Dict[str, Any]: Returns the loaded dictionary.
    """

    with open(data_path, "r") as infile:
        loaded_dict = json.load(infile)

    return loaded_dict


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
