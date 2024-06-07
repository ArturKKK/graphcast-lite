from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, List
from enum import Enum


class Grid2MeshEdgeCreation(str, Enum):
    """The different stratehies to create grid to mesh edges."""
    
    K_NEAREST = "k_nearest"
    RADIUS = "radius"
    
class AggregationTypes(str, Enum):
    """The different aggregation types we support. An aggregation is used to aggregate information from
    connected nodes without any paramters.
    """

    MEAN = "mean"


class Encoders(str, Enum):
    """The different encoder models we support. The encoder creates the latent representation of the mesh
    nodes using the grid nodes and the edges in the grid2mesh graph.
    """

    AGGREGATION = "aggregation"
    MLP = "mlp"


class GraphBuildingConfig(BaseModel):
    """This defines the parameters for building the graph.

    TODO: Support multiple ways of building the grid2mesh edges here. GraohCase uses radius_query_fraction_edge_length.
    But we could simply do k-nearest mesh nodes for every grid node for example.

    resolution: float
        The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    mesh_size: int
        How many refinements to do on the multi-mesh.
    grid2mesh_edge_creation: Grid2MeshEdgeCreation
        The strategy to create the Grid2Mesh edges.
    grid2mesh_radius_query: Optional[float]
        This needs to be passed if grid2mesh_edge_creation is 'radius'.
        Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.
    grid2mesh_k: Optional[int]:
        This needs to be passed if grid2mesh_edge_creation is 'k_nearest'. Each grid node
        will be connected to the nearest grid2mesh_k mesh nodes.
    mesh2grid_edge_normalization_factor: float
        Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.

    """

    resolution: float
    mesh_size: int
    grid2mesh_edge_creation: Grid2MeshEdgeCreation
    grid2mesh_radius_query: Optional[float] = None
    grid2mesh_k: Optional[int] = None
    mesh2grid_edge_normalization_factor: Optional[float] = None


class AggregationEncoderConfig(BaseModel):
    encoder_name: Literal[Encoders.AGGREGATION]
    aggregation_type: AggregationTypes


class MLPEncoderConfig(BaseModel):
    encoder_name: Literal[Encoders.MLP]
    num_hidden_layers: int
    hidden_sizes: List[int]


class ProcessConfig(BaseModel):
    pass


class DecoderConfig(BaseModel):
    pass


class ModelConfig(BaseModel):
    encoder: Union[MLPEncoderConfig, AggregationEncoderConfig] = Field(
        ..., discriminator="encoder_name"
    )
    processor: ProcessConfig
    decoder: DecoderConfig


class ExperimentConfig(BaseModel):
    graph_config: GraphBuildingConfig
    model: ModelConfig
