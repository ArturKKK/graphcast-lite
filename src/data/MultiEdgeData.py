from torch_geometric.data import Data

class MultiEdgeData(Data):
    def __init__(self, x, y, edge_index_encoder, edge_index_processor, edge_index_decoder, num_grid_nodes, num_mesh_nodes, **kwargs):
        super(MultiEdgeData, self).__init__(x=x, y=y, **kwargs)
        self.edge_index_encoder = edge_index_encoder
        self.edge_index_processor = edge_index_processor
        self.edge_index_decoder = edge_index_decoder
        self.num_grid_nodes = num_grid_nodes
        self.num_mesh_nodes = num_mesh_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_encoder':
            return self.num_grid_nodes + self.num_mesh_nodes
        if key == 'edge_index_processor':
            return self.num_mesh_nodes
        if key == 'edge_index_decoder':
            return self.num_grid_nodes + self.num_mesh_nodes
        return super().__inc__(key, value, *args, **kwargs)


def create_multi_edge_data(x, y, edge_index_encoder, edge_index_processor, edge_index_decoder, num_grid_nodes, num_mesh_nodes):
    data_list = []
    num_samples, _, _ = x.shape
    for i in range(num_samples):
        data = MultiEdgeData(
            x=x[i],
            y=y[i],
            edge_index_encoder=edge_index_encoder,
            edge_index_processor=edge_index_processor,
            edge_index_decoder=edge_index_decoder,
            num_grid_nodes=num_grid_nodes,
            num_mesh_nodes=num_mesh_nodes
        )
        data_list.append(data)
    return data_list