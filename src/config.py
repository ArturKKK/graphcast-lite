from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

    
class Grid2MeshEdgeCreation(str, Enum):
    """The different strategies to create grid to mesh edges."""

    K_NEAREST = "k_nearest"
    RADIUS = "radius"


class Mesh2GridEdgeCreation(str, Enum):
    """The different strategies to create mesh to grid edges."""

    CONTAINED = "contained"


class GraphLayerType(str, Enum):
    ConvGCN = "conv_gcn"
    SimpleConv = "simple_conv"
    GATConv = "conv_gat"


class ProductGraphType(str, Enum):
    KRONECKER = "kronecker"
    CARTESIAN = "cartesian"
    STRONG = "strong"


class DatasetNames(str, Enum):
    _64x32_10f_5y_3obs = "64x32_10f_5y_3obs"
    _64x32_33f_5y_5obs_uns = "64x32_33f_5y_5obs_uns"
    
    
class GraphBuildingConfig(BaseModel):
    """This defines the parameters for building the graph.

    mesh_size: int
        How many refinements to do on the multi-mesh.
    grid2mesh_edge_creation: Grid2MeshEdgeCreation
        The strategy to create the Grid2Mesh edges for encoding.
    mesh2grid_edge_creation: Mesh2GridEdgeCreation
        The strategy to create the Mesh2Grid edges for decoding.
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
    mesh_levels: List[int]
        The list of mesh levels to use for processing.

    """

    # grid-to-mesh graph configs
    grid2mesh_edge_creation: Grid2MeshEdgeCreation
    grid2mesh_radius_query: Optional[float] = None
    grid2mesh_k: Optional[int] = None

    # mesh graph configs
    mesh_levels: List[int]

    # mesh-to-grid graph configs
    mesh2grid_edge_creation: Mesh2GridEdgeCreation


class MLPBlock(BaseModel):
    mlp_hidden_dims: Optional[List[int]] = None
    output_dim: int
    use_layer_norm: bool
    layer_norm_mode: Optional[str] = None


class GraphBlock(BaseModel):
    layer_type: GraphLayerType
    hidden_dims: Optional[List[int]] = None
    output_dim: Optional[int] = None
    use_layer_norm: Optional[bool] = None
    layer_norm_mode: Optional[str] = None


class ModelConfig(BaseModel):
    mlp: Optional[MLPBlock] = None
    gcn: GraphBlock


class ProductGraphConfig(BaseModel):
    model: ModelConfig
    num_k: int
    self_loop: bool
    type: ProductGraphType


class PipelineConfig(BaseModel):
    product_graph: Optional[ProductGraphConfig] = None
    encoder: ModelConfig
    processor: ModelConfig
    decoder: ModelConfig
    residual_output: bool = False


class DataConfig(BaseModel):
    dataset_name: DatasetNames
    num_features_used: int
    obs_window_used: int
    pred_window_used: int
    want_feats_flattened: bool


class ExperimentConfig(BaseModel):
    batch_size: int
    learning_rate: float
    num_epochs: int
    random_seed: Optional[int] = None
    graph: GraphBuildingConfig
    pipeline: PipelineConfig
    data: DataConfig
    wandb_log: bool = True
    wandb_name: Optional[str] = None
    wandb_key: str = "3a59363c20cd4fdf2b95dfd7a9cd72398d15321e"
