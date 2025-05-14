"""Defines the configuration for an experiment."""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class Grid2MeshEdgeCreation(str, Enum):
    """The different strategies to create grid to mesh edges."""

    K_NEAREST = "k_nearest" # K ближайших соседей
    RADIUS = "radius" # Берутся все mesh-вершины в определенном радиусе


class Mesh2GridEdgeCreation(str, Enum):
    """The different strategies to create mesh to grid edges."""

    # для каждого узла grid находятся 3 узла mesh – вершины треугольника, внутри которого находится данная точка
    CONTAINED = "contained"


class GraphLayerType(str, Enum):
    """The different types of GNN layers we support."""

    ConvGCN = "conv_gcn"
    SimpleConv = "simple_conv"
    GATConv = "conv_gat"
    SparseGATConv = "sparse_gat"

class ProductGraphType(str, Enum):
    """The different types of product graph."""

    KRONECKER = "kronecker"
    CARTESIAN = "cartesian"
    STRONG = "strong"


class DatasetNames(str, Enum):
    """The different datasets to run the experiment on."""

    _64x32_10f_5y_3obs = "64x32_10f_5y_3obs"
    _64x32_33f_5y_5obs_uns = "64x32_33f_5y_5obs_uns"
    _64x32_12f_2y_2obs_1pred_uns = "64x32_12f_2y_2obs_1pred_uns"


class GraphBuildingConfig(BaseModel):
    """This defines the parameters for building the graph.

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
    """This defines the configuration for an MLPBlock.

    mlp_hidden_dims: Optional[List[int]]
        The hidden dimensions in the MLP. Can be empty for single layer. PReLU is applied after every hidden layer.
    output_dim: int
        The output dim for the MLP block
    use_layer_norm: bool
        Whether to use layer norm or not. It is applied after every layer in the MLP.
    layer_norm_mode: Optional[str] = None
        The mode of the layer norm. Can be either "node" or "graph".
    """

    mlp_hidden_dims: Optional[List[int]] = None
    output_dim: int
    use_layer_norm: bool
    layer_norm_mode: Optional[str] = None


class GATProps(BaseModel):
    """This defines the configuration for Sparse Attention.

    num_heads: int
        The number of attention heads
    sparsity_thresholds: List[float]
        The sparsity thresholds for each layer.
    """

    num_heads: int
    sparsity_thresholds: List[float]


class GraphBlock(BaseModel):
    """This defines the configuration for the graph block.
    
    layer_type: GraphLayerType
        The type of GNN layer that is to be used.
    gat_props: Optional[GATProps]
        The configuration for sparse attention if sparse attention is to be used.
    hidden_dims: Optional[List[int]]
        Hidden dims in the GNN. Not passed for simple_conv.
    output_dim: Optional[int]
        Output dims in the GNN. Not passed for simple_conv.
    use_layer_norm: Optional[bool]
        Whether to use layer norm or not. Applied after every GNN layer. 
    layer_norm_mode: Optional[str]
         The mode of the layer norm. Can be either "node" or "graph".
    """
    layer_type: GraphLayerType
    gat_props: Optional[GATProps] = None
    hidden_dims: Optional[List[int]] = None
    output_dim: Optional[int] = None
    use_layer_norm: Optional[bool] = None
    layer_norm_mode: Optional[str] = None


class ModelConfig(BaseModel):
    """A model is defined using an MLP block and a GraphBlock
    
    mlp: Optional[MLPBlock]
        The MLP block for the model. Can be empty if no MLP is needed.
    gcn: GraphBlock
        The GraphBlock for the model.
    """
    mlp: Optional[MLPBlock] = None
    gcn: GraphBlock


class ProductGraphConfig(BaseModel):
    """Defines the configuration of the product graph.
    
    model: ModelConfig
        The model for the product graph message passing.
    num_k: int
        The number of edges for each grid node created using k-nearest-neighbours
    self_loop: bool
        Whether a self loop is added in the product graph or not.
    type: ProductGraphType
        The type of the product graph.
    """
    model: ModelConfig
    num_k: int
    self_loop: bool
    type: ProductGraphType


class PipelineConfig(BaseModel):
    """ Defines the configuration of the entire pipeline.
    product_graph: Optional[ProductGraphConfig]
        Config for the product graph. This is optional and only needs to be set if experiment has product graph.
    encoder: ModelConfig
        Config for the encoder.
    processor: ModelConfig
        Config for the processor.
    decoder:
        Config of the decoder.    
    """
    product_graph: Optional[ProductGraphConfig] = None
    encoder: ModelConfig
    processor: ModelConfig
    decoder: ModelConfig


class DataConfig(BaseModel):
    dataset_name: DatasetNames
    num_features_used: int
    obs_window_used: int
    pred_window_used: int
    want_feats_flattened: bool


class ExperimentConfig(BaseModel):
    batch_size: int = 1
    learning_rate: float = 1e-5
    # Когда счетчик превышает patience (то есть нет улучшения в течение 10 эпох подряд), 
    # цикл обучения преждевременно прекращается.
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    num_epochs: int = 100
    random_seed: Optional[int] = 42
    graph: GraphBuildingConfig
    pipeline: PipelineConfig
    data: DataConfig
    wandb_log: bool = True
    wandb_name: Optional[str] = None
    wandb_key: str = "3a59363c20cd4fdf2b95dfd7a9cd72398d15321e"
