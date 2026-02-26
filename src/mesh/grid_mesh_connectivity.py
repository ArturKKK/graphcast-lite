"""Defines some utility functions to create grid-mesh edges."""

from .create_mesh import TriangularMesh
import numpy as np
import scipy
import trimesh
from typing import Tuple


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray, flat: bool = False
) -> np.ndarray:
    """Copied from GraphCast (extended for flat grids).

    Regular mode (flat=False):
      Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3].
    Flat mode (flat=True):
      Lat [N] lon [N] (paired per-node) to 3d coordinates [N, 3].
    """
    if flat:
        # Уже спаренные координаты — пропускаем meshgrid
        phi = np.deg2rad(grid_longitude)
        theta = np.deg2rad(90 - grid_latitude)
        return np.stack(
            [
                np.cos(phi) * np.sin(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(theta),
            ],
            axis=-1,
        )  # (N, 3)

    # Regular grid mode: meshgrid из 1D осей
    # Convert to spherical coordinates phi and theta defined in the grid.
    # Each [num_latitude_points, num_longitude_points]
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    )

    # [num_latitude_points, num_longitude_points, 3]
    # Note this assumes unit radius, since for now we model the earth as a
    # sphere of unit radius, and keep any vertical dimension as a regular grid.
    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    radius: float,
    flat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Copied from GraphCast (extended for flat grids).

    Returns mesh-grid edge indices for radius query.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points] (regular)
                     or [N] (flat, paired with grid_longitude).
      grid_longitude: Longitude values for the grid [num_lon_points] (regular)
                      or [N] (flat, paired with grid_latitude).
      mesh: Mesh object.
      radius: Radius of connectivity in R3. for a sphere of unit radius.
      flat: If True, grid_latitude and grid_longitude are already-paired
            flat coordinate arrays of shape (N,) instead of 1D axes.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges.
      * grid_indices: Indices of shape [num_edges], indexing into flattened grid.
      * mesh_indices: Indices of shape [num_edges], indexing into mesh.vertices.
    """
    # [num_grid_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude, flat=flat
    ).reshape([-1, 3])

    # [num_mesh_points, 3]
    mesh_positions = mesh.vertices
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices

def get_max_edge_distance(mesh):
  senders, receivers = faces_to_edges(mesh.faces)
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max()

def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Copied from GraphCast
  
  Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
    faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
    Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
  return senders, receivers


def in_mesh_triangle_indices(
    *, grid_latitude: np.ndarray, grid_longitude: np.ndarray,
    mesh: TriangularMesh, flat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Copied from GraphCast (extended for flat grids).

    Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
      grid_latitude: Latitude values [num_lat_points] (regular) or [N] (flat).
      grid_longitude: Longitude values [num_lon_points] (regular) or [N] (flat).
      mesh: Mesh object.
      flat: If True, coords are already-paired flat arrays.

    Returns:
      tuple with `grid_indices` and `mesh_indices`.
      * grid_indices: Indices of shape [num_edges], indexing into flattened grid.
      * mesh_indices: Indices of shape [num_edges], indexing into mesh.vertices.
    """

    # [num_grid_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude, flat=flat
    ).reshape([-1, 3])

    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # [num_grid_points] with mesh face indices for each grid point.
    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions
    )

    # [num_grid_points, 3] with mesh node indices for each grid point.
    mesh_edge_indices = mesh.faces[query_face_indices]

    # [num_grid_points, 3] with grid node indices, where every row simply contains
    # the row (grid_point) index.
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    # Flatten to get a regular list.
    # [num_edges=num_grid_points*3]
    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices
