"""Utility methods to create the encoding, processing and decoding graphs.

РУССКИЕ ПОЯСНЕНИЯ:
- В этом модуле строятся три типа графов, которые затем использует модель:
  1) ENCODING (Grid → Mesh): как «заливать» признаки с регулярной сетки (grid) на триангуляционную сетку (mesh).
  2) PROCESSING (Mesh ↔ Mesh): как соединены вершины внутри самой треугольной сетки для propagation/message passing.
  3) DECODING (Mesh → Grid): как «снимать» обновлённые признаки с mesh обратно на grid, чтобы получить прогноз на исходной сетке.

- В коде принята конвенция индексации узлов: сначала идут ВСЕ узлы Grid (индексы 0..N-1), а затем ВСЕ узлы Mesh (индексы N..N+M-1).
  Поэтому каждый раз, когда мы формируем рёбра с участием Mesh, мы сдвигаем индексы Mesh на +num_grid_nodes.

- Термины:
  * Бипартиный граф: рёбра только между двумя множествами узлов (у нас Grid↔Mesh).
  * TriangularMesh: структура с координатами вершин на сфере и списком треугольных граней (faces).
  * edge_index: тензор формы [2, E] с парами (sender → receiver) для рёбер.
"""

# Бипартиный граф: узлы разбиты на два множества, рёбра идут только между ними (у нас Grid↔Mesh).
# TriangularMesh: структура с координатами вершин (широты/долготы на сфере)
# faces (треугольники по индексам вершин). Из faces легко получить список рёбер.

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
from src.utils import get_bipartite_graph_spatial_features, get_mesh_lat_long


def _compute_mesh_edge_features(
    mesh_node_lats: np.ndarray,
    mesh_node_longs: np.ndarray,
    edge_index: np.ndarray,
) -> torch.Tensor:
    """Вычисляет пространственные фичи рёбер для processing-графа (mesh↔mesh).

    Для каждого ребра:
    - relative_distance (нормированная длина) — 1 фича
    - relative_position (3D вектор в локальных координатах получателя) — 3 фичи
    Итого 4 фичи на ребро.

    Это именно то, что делает Google GraphCast для mesh-рёбер.
    """
    from src.utils import lat_lon_deg_to_spherical, spherical_to_cartesian
    from src.utils import get_bipartite_relative_position_in_receiver_local_coordinates

    senders = edge_index[0]
    receivers = edge_index[1]

    sender_phi, sender_theta = lat_lon_deg_to_spherical(
        mesh_node_lats[senders], mesh_node_longs[senders]
    )
    receiver_phi, receiver_theta = lat_lon_deg_to_spherical(
        mesh_node_lats[receivers], mesh_node_longs[receivers]
    )

    # Используем ту же функцию, что и в GraphCast — относительные позиции
    # в локальных координатах получателя.
    relative_position = get_bipartite_relative_position_in_receiver_local_coordinates(
        senders_node_phi=lat_lon_deg_to_spherical(mesh_node_lats, mesh_node_longs)[0],
        senders_node_theta=lat_lon_deg_to_spherical(mesh_node_lats, mesh_node_longs)[1],
        receivers_node_phi=lat_lon_deg_to_spherical(mesh_node_lats, mesh_node_longs)[0],
        receivers_node_theta=lat_lon_deg_to_spherical(mesh_node_lats, mesh_node_longs)[1],
        senders=senders,
        receivers=receivers,
        latitude_local_coordinates=True,
        longitude_local_coordinates=True,
    )

    # Нормируем по максимальной длине ребра
    relative_edge_distances = np.linalg.norm(relative_position, axis=-1, keepdims=True)
    max_dist = relative_edge_distances.max()
    if max_dist > 0:
        relative_edge_distances_norm = relative_edge_distances / max_dist
        relative_position_norm = relative_position / max_dist
    else:
        relative_edge_distances_norm = relative_edge_distances
        relative_position_norm = relative_position

    # [num_edges, 4]: distance + 3D position
    edge_features = np.concatenate(
        [relative_edge_distances_norm, relative_position_norm], axis=-1
    )
    return torch.tensor(edge_features, dtype=torch.float32)


# Задача. Построить рёбра от узлов Grid к узлам Mesh, чтобы «залить» исходные признаки с Grid в ближайшие Mesh-узлы.
# Плюс — посчитать статические фичи узлов (широты/долготы и т.п.), которые модель будет знать всегда.
def create_encoding_graph(
    grid_node_lats: np.ndarray,
    grid_node_longs: np.ndarray,
    mesh_node_lats: np.ndarray,
    mesh_node_longs: np.ndarray,
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
    num_grid_nodes: int,
    flat_grid: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates the edges between the grid and the mesh based on the strategy specified in the graph building config. Also contructs
        the initial static features of the grid and the mesh nodes like the latitudes, longitudes etc.

    Parameters
    ----------
    grid_node_lats : np.ndarray
        1D latitude axis [num_lat] for regular grid, or flat [N] for flat_grid=True.
    grid_node_longs : np.ndarray
        1D longitude axis [num_lon] for regular grid, or flat [N] for flat_grid=True.
    flat_grid : bool
        If True, grid_node_lats and grid_node_longs are already-paired flat arrays.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns a tuple of torch.Tensors where the first element is the edge index of the encoding graph of shape [2, num_edges] which defines the edges between
        grid nodes and the mesh nodes. The second element is a tensor with the initial grid features. The third element is a tensor with the initial mesh features.

    """
    # Вариант построения рёбер Grid→Mesh выбирается параметром grid2mesh_edge_creation в конфиге.
    if graph_building_config.grid2mesh_edge_creation == Grid2MeshEdgeCreation.RADIUS:
        # 1) Вычисляем радиус поиска соседей Mesh для каждого узла Grid.
        #    Берём максимальную длину ребра на выбранном mesh (характерный масштаб сетки)
        #    и умножаем на коэффициент из конфига. Это даёт радиус в единицах той же метрики,
        #    что внутри TriangularMesh (обычно chordal distance в 3D или геодезическая приближенка).
        radius = (
            get_max_edge_distance(mesh=mesh)
            * graph_building_config.grid2mesh_radius_query
        )

        # 2) Находим пары индексов (grid_index, mesh_index) таких, что узел mesh находится в указанном радиусе
        #    от узла grid. Возвращаются два параллельных массива одинаковой длины.
        grid_indices, mesh_indices = radius_query_indices(
            grid_latitude=grid_node_lats,
            grid_longitude=grid_node_longs,
            mesh=mesh,
            radius=radius,
            flat=flat_grid,
        )

        # 3) Собираем edge_index формы [2, E], где первая строка — индексы отправителей (Grid),
        #    вторая — получателей (Mesh). Пока индексы Mesh ещё «локальные» (0..M-1).
        edge_index = np.stack([grid_indices, mesh_indices], axis=0)

        # 4) Сдвигаем индексы Mesh на +num_grid_nodes, чтобы они попадали в диапазон [N, N+M-1]
        #    при конкатенации узлов (Grid идёт первым блоком, Mesh — вторым).
        edge_index[1] += num_grid_nodes
        edge_index = torch.tensor(edge_index, dtype=torch.int64)

    else:
        # Сейчас реализован только сценарий RADIUS для Grid→Mesh.
        raise NotImplementedError(
            f"There is no support for {graph_building_config.grid2mesh_edge_creation} to create Grid2Mesh edges."
        )

    # 5) Подготавливаем координаты узлов Grid в векторной форме.
    if flat_grid:
        # Flat grid: координаты уже спарены, пропускаем meshgrid
        grid_nodes_lat = grid_node_lats.reshape([-1]).astype(np.float32)
        grid_nodes_lon = grid_node_longs.reshape([-1]).astype(np.float32)
    else:
        #    Входные grid_node_lats/longs — это 1D массивы «осей». meshgrid даёт 2D сетку (ширина×высота),
        #    затем мы её выравниваем (flatten) в длину N и приводим к float32.
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_node_longs, grid_node_lats)
        grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    # 6) Считаем статические пространственные признаки для бипартийного графа Grid→Mesh.
    #    Функция вернёт:
    #    - признаки узлов-отправителей (Grid),
    #    - признаки узлов-получателей (Mesh),
    #    - признаки рёбер (здесь не используются в возвращаемом значении этой функции, но внутри они также формируются).
    #    Внутри обычно кодируются широта/долгота/xyz, а для рёбер — относительные смещения и нормированные расстояния.
    grid_node_features, mesh_node_features, _ = get_bipartite_graph_spatial_features(
        senders_node_lat=grid_nodes_lat,
        senders_node_lon=grid_nodes_lon,
        receivers_node_lat=mesh_node_lats,
        receivers_node_lon=mesh_node_longs,
        senders=edge_index[0],
        receivers=edge_index[1],
    )

    # Возвращаем:
    # - edge_index: тензор [2, E] с рёбрами Grid→Mesh (Mesh-индексы уже сдвинуты на +N);
    # - grid_node_features: тензор [N, F_grid_stat] статических фичей для всех узлов Grid;
    # - mesh_node_features: тензор [M, F_mesh_stat] статических фичей для всех узлов Mesh.
    return (
        edge_index,
        torch.tensor(grid_node_features, dtype=torch.float32),
        torch.tensor(mesh_node_features, dtype=torch.float32),
    )


def create_processing_graph(
    meshes: List[TriangularMesh], mesh_levels: List[int],
    mesh_node_lats: np.ndarray = None,
    mesh_node_longs: np.ndarray = None,
) -> Tuple[torch.Tensor, ...]:
    """Returns the edges within the mesh in the processing graph based on the mesh resolution levels.

    Parameters
    ----------
    meshes : List[TriangularMesh]
        All the meshes constructed using the mesh_levels specified
    mesh_levels : List[int]
        The mesh levels for the experiment.
    mesh_node_lats : np.ndarray, optional
        Latitudes of mesh nodes (needed for edge features).
    mesh_node_longs : np.ndarray, optional
        Longitudes of mesh nodes (needed for edge features).

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Returns (edge_index, edge_features) or just edge_index if no coordinates provided.
    """

    # 1) Выбираем нужные уровни иерархической сетки (например, один или несколько уровней разбиения икосаэдра).
    #    filter_mesh вернёт «слитую» структуру с faces для указанных уровней.
    meshes_we_want = filter_mesh(meshes=meshes, mesh_levels=mesh_levels)

    # 2) Преобразуем треугольники (faces) в рёбра графа. В неориентированном случае каждое ребро обычно даётся
    #    в обе стороны (u→v и v→u), что удобно для message passing. Функция возвращает массив shape [2, E].
    edge_index = torch.tensor(get_edges_from_faces(meshes_we_want.faces), dtype=torch.int64)

    # 3) Если переданы координаты, вычисляем edge features
    if mesh_node_lats is not None and mesh_node_longs is not None:
        edge_features = _compute_mesh_edge_features(
            mesh_node_lats=mesh_node_lats,
            mesh_node_longs=mesh_node_longs,
            edge_index=edge_index.numpy(),
        )
        return edge_index, edge_features

    return edge_index



def create_decoding_graph(
    cordinates: Tuple[np.array, np.array],  # NOTE: орфография параметра сохранена как в исходнике
    mesh: TriangularMesh,
    graph_building_config: GraphBuildingConfig,
    num_grid_nodes: int,
    flat_grid: bool = False,
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

    # Вариант построения рёбер Mesh→Grid задаётся параметром mesh2grid_edge_creation.
    if graph_building_config.mesh2grid_edge_creation == Mesh2GridEdgeCreation.CONTAINED:
        # 1) Для каждого узла Grid находим индекс треугольника Mesh, который «накрывает» эту точку на сфере.
        #    Возвращаются два массива одинаковой длины: grid_indices (получатели) и mesh_indices (отправители).
        #    Для каждого grid-узла будет ровно 3 mesh-вершины (вершины одного треугольника).
        grid_indices, mesh_indices = in_mesh_triangle_indices(
            grid_latitude=cordinates[0], grid_longitude=cordinates[1], mesh=mesh,
            flat=flat_grid,
        )

        # 2) Формируем edge_index формы [2, E], но теперь рёбра направлены от Mesh → Grid,
        #    потому что мы будем «снимать» признаки с mesh-вершин на узлы grid.
        edge_index = np.stack([mesh_indices, grid_indices], axis=0)

        # 3) Сдвигаем индексы Mesh на +num_grid_nodes (единая индексация узлов, как в ENCODING).
        edge_index[0] += num_grid_nodes
        edge_index = torch.tensor(edge_index, dtype=torch.int64)

        # Итог: на каждый узел Grid приходится ровно 3 входящих ребра от вершин соответствующего треугольника Mesh.
        return edge_index

    else:
        # Сейчас реализована только стратегия CONTAINED для Mesh→Grid.
        raise NotImplementedError(
            f"There is no support for {graph_building_config.mesh2grid_edge_creation} to create Mesh2Grid edges."
        )
