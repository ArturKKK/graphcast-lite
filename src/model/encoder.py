import torch.nn as nn
from src.config import SimpleAggregationEncoder
from torch_geometric.data import Data

class AggregationEncoder(nn.Module):
    """The encoder creates the latent representation of the mesh nodes from the grid nodes.
    
    The AggregationEncoder simply aggregates nodes connected to a mesh node.
    
    """
    
    def __init__(self, encoder_config: SimpleAggregationEncoder):
        self.encoder_config = encoder_config
    
    def forward(self, data: Data):
        pass