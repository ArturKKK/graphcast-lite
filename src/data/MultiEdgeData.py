from torch_geometric.data import Data

class MultiEdgeData(Data):
    def __init__(self, x, y, edge_index_encoder, edge_index_processor, edge_index_decoder, **kwargs):
        super(MultiEdgeData, self).__init__(x=x, y=y, **kwargs)
        self.edge_index_encoder = edge_index_encoder
        self.edge_index_processor = edge_index_processor
        self.edge_index_decoder = edge_index_decoder


def create_multi_edge_data(x, y, edge_index_encoder, edge_index_processor, edge_index_decoder):
    data_list = []
    num_samples, _, _ = x.shape
    for i in range(num_samples):
        data = MultiEdgeData(
            x=x[i],
            y=y[i],
            edge_index_encoder=edge_index_encoder,
            edge_index_processor=edge_index_processor,
            edge_index_decoder=edge_index_decoder
        )
        data_list.append(data)
    return data_list