from torch_geometric.nn import GAT, to_hetero, InnerProductDecoder
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy
import torch
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict
from torchmetrics import Accuracy, Precision, Recall, F1Score

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


@dataclass
class TrainConfig:
    batch_size: int
    num_layers: int
    num_neighbors: int
    learning_rate: float
    epochs: int
    dropout: Optional[float] = None
    betas: Optional[tuple[float, float]] = None
    decay_lr: Optional[bool] = None
    min_lr: Optional[float] = None
    lr_decay_iters: Optional[int] = None
    weight_decay: Optional[float] = None
    fused: Optional[bool] = None

    @property
    def depth_sizes(self):
        return [self.num_neighbors] * self.num_layers


class LinkPredModel(pl.LightningModule):
    def __init__(self,
                 target_edge: tuple[str, str, str],
                 metadata,
                 gnn_kwargs: GATConfig,
                 train_cfg: TrainConfig
                 ):
        super().__init__()
        self.encoder = to_hetero(GAT(**asdict(gnn_kwargs)),
                                 metadata=metadata,
                                 aggr='sum')
        self.decoder = InnerProductDecoder()
        self.target_edge = target_edge
        self.train_cfg = train_cfg


        # Initialize the metric objects
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1_score = F1Score(task="binary")

    def forward(self, batch):
        # Encode the graph data to get node embeddings
        z_dict = self.encoder(batch.x_dict, batch.edge_index_dict)

        edge_label_index = batch[self.target_edge].edge_label_index

        # Extract embeddings for the source and target node types involved in the target edge
        source_node_type, _, target_node_type = self.target_edge
        z = torch.cat([z_dict[source_node_type], z_dict[target_node_type]], dim=0)

        # Predict the presence of edges
        edge_probabilities = self.decoder(z, edge_label_index, sigmoid=True)

        return edge_probabilities

    def model_step(self, batch, step_name):
        pred = self(batch)
        target = batch[self.target_edge].edge_label
        loss = binary_cross_entropy(pred, target)
        batch_size = batch[self.target_edge].edge_label.shape[0]
        self.log(name=f'{step_name}_loss',
                 value=loss,
                 batch_size=batch_size,
                 on_step=True,
                 prog_bar=True
                 )

        if step_name == 'val':
            # Calculate metrics only for the validation step
            self.accuracy(pred, target)
            self.precision(pred, target)
            self.recall(pred, target)
            self.f1_score(pred, target)

            log_args = {'batch_size': batch_size,
                        'on_epoch': True,
                        'prog_bar': True
                        }

            self.log(name=f'{step_name}_accuracy', value=self.accuracy, **log_args)
            self.log(name=f'{step_name}_precision', value=self.precision, **log_args)
            self.log(name=f'{step_name}_recall', value=self.recall, **log_args)
            self.log(name=f'{step_name}_f1_score', value=self.f1_score, **log_args)

        return loss

    def on_validation_epoch_end(self):
        self.log(name='val_accuracy_epoch', value=self.accuracy.compute())
        self.log(name='val_precision_epoch', value=self.precision.compute())
        self.log(name='val_recall_epoch', value=self.recall.compute())
        self.log(name='val_f1_score_epoch', value=self.f1_score.compute())

    def training_step(self, batch):
        return self.model_step(batch, 'train')

    def validation_step(self, batch):
        return self.model_step(batch, 'val')

    def test_step(self, batch):
        return self.model_step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.train_cfg.learning_rate)