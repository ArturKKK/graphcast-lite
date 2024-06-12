from typing import Tuple, List
import numpy as np
from src.mesh import (
    TriangularMesh,
    radius_query_indices,
    get_max_edge_distance,
    in_mesh_triangle_indices,
)
from src.config import GraphBuildingConfig, Grid2MeshEdgeCreation, Mesh2GridEdgeCreation


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
            f"There is no support for {graph_building_config.grid2mesh_edge_creation} to create Grid2Mesh edges."
        )


def create_mesh_graph_edges(mesh: TriangularMesh, graph_building_config: GraphBuildingConfig):
    """ ...

    Parameters
    ----------
    mesh : TriangularMesh
        The mesh created over the grid nodes.
    graph_building_config : GraphBuildingConfig
        The graph building configuration for the experiment

    Returns
    -------
    np.array
        Returns a numpy array of shape [num_edges, 2] which defines the edges between the mesh nodes.
    """
    pass

    


def create_mesh_to_grid_graph(
    cordinates: Tuple[np.array, np.array],
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
):
    """Creates the edges between the mesh and the grid based on the strategy specified for mesh to grid in the graph building config.

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
        Returns a numpy array of shape [num_edges, 2] which defines the edges between the mesh nodes and the grid nodes.
    """

    if graph_building_config.mesh2grid_edge_creation == Mesh2GridEdgeCreation.CONTAINED:
        grid_indices, mesh_indices = in_mesh_triangle_indices(
            grid_latitude=cordinates[0], grid_longitude=cordinates[1], mesh=mesh
        )

        return np.stack([mesh_indices, grid_indices], axis=-1)

    else:
        raise NotImplementedError(
            f"There is no support for {graph_building_config.mesh2grid_edge_creation} to create Mesh2Grid edges."
        )
