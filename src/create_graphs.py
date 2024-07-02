"""Utility methods to create the encoding, processing and decoding graphs."""

from typing import Tuple, List
import numpy as np
from src.mesh import (
    TriangularMesh,
    radius_query_indices,
    get_max_edge_distance,
    in_mesh_triangle_indices,
)
from src.config import GraphBuildingConfig, Grid2MeshEdgeCreation, Mesh2GridEdgeCreation
import torch

from src.mesh.create_mesh import filter_mesh, get_edges_from_faces
from src.utils import get_bipartite_graph_spatial_features


def create_encoding_graph(
    grid_node_lats: np.ndarray,
    grid_node_longs: np.ndarray,
    mesh_node_lats: np.ndarray,
    mesh_node_longs: np.ndarray,
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
    num_grid_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates the edges between the grid and the mesh based on the strategy specified in the graph building config. Also contructs
        the initial static features of the grid and the mesh nodes like the latitudes, longitudes etc.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns a tuple of torch.Tensors where the first element is the edge index of the encoding graph of shape [2, num_edges] which defines the edges between
        grid nodes and the mesh nodes. The second element is a tensor with the initial grid features. The third element is a tensor with the initial mesh features.

    """
    if graph_building_config.grid2mesh_edge_creation == Grid2MeshEdgeCreation.RADIUS:
        radius = (
            get_max_edge_distance(mesh=mesh)
            * graph_building_config.grid2mesh_radius_query
        )

        grid_indices, mesh_indices = radius_query_indices(
            grid_latitude=grid_node_lats,
            grid_longitude=grid_node_longs,
            mesh=mesh,
            radius=radius,
        )

        edge_index = np.stack([grid_indices, mesh_indices], axis=0)

        # Making sure the mesh indices start after the node_indices
        edge_index[1] += num_grid_nodes
        edge_index = torch.tensor(edge_index, dtype=torch.int64)

    else:
        raise NotImplementedError(
            f"There is no support for {graph_building_config.grid2mesh_edge_creation} to create Grid2Mesh edges."
        )

    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_node_longs, grid_node_lats)
    grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
    grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    grid_node_features, mesh_node_features, _ = get_bipartite_graph_spatial_features(
        senders_node_lat=grid_nodes_lat,
        senders_node_lon=grid_nodes_lon,
        receivers_node_lat=mesh_node_lats,
        receivers_node_lon=mesh_node_longs,
        senders=edge_index[0],
        receivers=edge_index[1],
    )

    return (
        edge_index,
        torch.tensor(grid_node_features, dtype=torch.float32),
        torch.tensor(mesh_node_features, dtype=torch.float32),
    )


def create_processing_graph(
    meshes: List[TriangularMesh], mesh_levels: List[int]
) -> torch.Tensor:
    """Returns the edges within the mesh in the processing graph based on the mesh resolution levels.

    Parameters
    ----------
    meshes : List[TriangularMesh]
        All the meshes constructed using the mesh_levels specified
    mesh_levels : List[int]
        The mesh levels for the experiment.

    Returns
    -------
    torch.Tensor
        Returns the edges in the mesh based on the resolution levels. Returns tensor of shape [2, num_edges].

    """

    # This will have to be updated on taking multiple levels of edges
    meshes_we_want = filter_mesh(meshes=meshes, mesh_levels=mesh_levels)
    return torch.tensor(get_edges_from_faces(meshes_we_want.faces), dtype=torch.int64)


def create_decoding_graph(
    cordinates: Tuple[np.array, np.array],
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
    num_grid_nodes: int,
) -> torch.Tensor:
    """Creates the edges between the mesh and the grid based on the strategy specified for mesh to grid in the graph building config.

    Parameters
    ----------
    cordinates : Tuple[np.array, np.array]
        A tuple of the latitude and the longitudes of the grid nodes.
    mesh : TriangularMesh
        The mesh created over the grid nodes.
    graph_building_config : GraphBuildingConfig
        The graph building configuration for the experiment
    num_grid_nodes: int
        Number of grid nodes based on the resolution of the spatial grid.

    Returns
    -------
    torch.Tensor
        Returns a tensor of shape [2, num_edges] which defines the edges between the mesh nodes and the grid nodes.
    """

    if graph_building_config.mesh2grid_edge_creation == Mesh2GridEdgeCreation.CONTAINED:
        grid_indices, mesh_indices = in_mesh_triangle_indices(
            grid_latitude=cordinates[0], grid_longitude=cordinates[1], mesh=mesh
        )

        # Making sure the mesh indices start after the node_indices
        edge_index = np.stack([mesh_indices, grid_indices], axis=0)
        edge_index[0] += num_grid_nodes
        edge_index = torch.tensor(edge_index, dtype=torch.int64)

        return edge_index

    else:
        raise NotImplementedError(
            f"There is no support for {graph_building_config.mesh2grid_edge_creation} to create Mesh2Grid edges."
        )
