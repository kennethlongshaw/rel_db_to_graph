import torch
from torch_geometric.nn import GAT
from torch_geometric.nn import to_hetero
import pytorch_lightning as pl


def train():
    data = torch.load(f'data/graph.bin')

    class InnerProductDecoder(torch.nn.Module):
        def forward(self, graph):
            x_dict, edge_label_index = graph.x_dict, graph.edge_label_index
            x_src = x_dict['user'][edge_label_index[0]]
            x_dst = x_dict['movie'][edge_label_index[1]]
            return (x_src * x_dst).sum(dim=-1)

    class Model(pl.LightningModule):
        def __init__(self, gnn_kwargs):
            super().__init__()
            self.encoder = GAT(**gnn_kwargs)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
            self.decoder = InnerProductDecoder()

        def forward(self, graph):
            x_dict = graph.x_dict
            edge_index_dict, edge_label_index = graph.edge_index_dict, graph.edge_index_dict
            x_dict = self.encoder(x_dict, edge_index_dict)
            return self.decoder(x_dict, edge_label_index)

    model = Model()

