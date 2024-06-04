from typing import Tuple, List
import numpy as np
from src.mesh import TriangularMesh, radius_query_indices, get_max_edge_distance
from src.config import GraphBuildingConfig, Grid2MeshEdgeCreation


def create_grid_to_mesh_graph(
    cordinates: Tuple[np.array, np.array],
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
) -> np.array:
    """Creates the edges between the grid and the mesh based on the strategy specified in the graph building config.

    Parameters
    ----------
    cordinates : Tuple[np.array, np.array]
        A tuple of the latitude and the longitudes of the grid nodes.
    mesh : TriangularMesh
        The mesh created over the grid nodes.
    graph_building_config : GraphBuildingConfig
        The graph building configuration for the experiment

    Returns
    -------
    np.array
        Returns a numpy array of shape [num_edges, 2] which defines the edges between the grid nodes and the mesh nodes.
    """
    if graph_building_config.grid2mesh_edge_creation == Grid2MeshEdgeCreation.RADIUS:
        radius = (
            get_max_edge_distance(mesh=mesh)
            * graph_building_config.grid2mesh_radius_query
        )

        grid_indices, mesh_indices = radius_query_indices(
            grid_latitude=cordinates[0],
            grid_longitude=cordinates[1],
            mesh=mesh,
            radius=radius,
        )

        return np.stack([grid_indices, mesh_indices], axis=-1)
    else:
        raise NotImplementedError(
            f"There is no support for {graph_building_config.grid2mesh_edge_creation} to create Grid2Mesh indices."
        )


def create_mesh_to_mesh_graph(graph_building_config: GraphBuildingConfig):
    pass


def create_mesh_to_grid_graph(graph_building_config: GraphBuildingConfig):
    pass
