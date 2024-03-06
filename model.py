from torch_geometric.nn import GAT, to_hetero, InnerProductDecoder
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy
import torch
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict


# GATConfig = create_config_from_class(GAT)
@dataclass
class GATConfig:
    in_channels: int | tuple
    hidden_channels: int
    num_layers: int
    out_channels: Optional[int] = None
    dropout: Optional[float] = None
    act: Optional[str | Callable] = 'relu'
    act_first: Optional[bool] = None
    act_kwargs: Optional[dict] = None
    norm: Optional[str | Callable] = None
    norm_kwargs: Optional[dict[str, Any]] = None
    jk: Optional[str] = None
    add_self_loops: Optional[bool] = None
    v2: Optional[bool] = None


class LinkPredModel(pl.LightningModule):
    def __init__(self,
                 target_edge: tuple[str, str, str],
                 metadata,
                 gnn_kwargs: GATConfig,
                 lr: float
                 ):
        super().__init__()
        self.encoder = to_hetero(GAT(**asdict(gnn_kwargs)),
                                 metadata=metadata,
                                 aggr='sum')
        self.decoder = InnerProductDecoder()
        self.target_edge = target_edge
        self.lr = lr

    # def forward(self, batch):
    #     x_dict = self.encoder(batch.x_dict,
    #                           batch.edge_index_dict)
    #     return self.decoder(x_dict, batch[self.target_edge].edge_label_index)

    def forward(self, batch):
        # Encode the graph data to get node embeddings
        z_dict = self.encoder(batch.x_dict, batch.edge_index_dict)

        edge_label_index = batch[
            self.target_edge].edge_label_index  # Adjust based on how edge indices are stored in your batch

        # Extract embeddings for the source and target node types involved in the target edge
        source_node_type, _, target_node_type = self.target_edge
        z = torch.cat([z_dict[source_node_type], z_dict[target_node_type]], dim=0)

        # Predict the presence of edges
        edge_probabilities = self.decoder(z, edge_label_index, sigmoid=True)

        return edge_probabilities

    def model_step(self, batch, step_name):
        pred = self(batch)
        loss = binary_cross_entropy(pred, batch[self.target_edge].edge_label)
        self.log(name=f'{step_name}_loss',
                 value=loss,
                 batch_size=batch[self.target_edge].edge_label.shape[0]
                 )
        return loss

    def training_step(self, batch):
        return self.model_step(batch, 'train')

    def validation_step(self, batch):
        return self.model_step(batch, 'val')

    def test_step(self, batch):
        return self.model_step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.lr)
        # fused=True)
