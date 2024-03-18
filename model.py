from torch_geometric.nn import GAT, to_hetero
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


class EdgeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_dict: dict, edge_label_index: torch.Tensor, target_edge: tuple) -> torch.Tensor:
        """
        Decodes edges by calculating the dot product between embeddings of node pairs
        and then applying a sigmoid function to map the scores to probabilities.

        Args:
            z_dict (dict): Dictionary of node embeddings, keyed by node type.
            edge_label_index (Tensor): The indices of the edges to be decoded, shape [2, num_edges].
            target_edge (tuple): A tuple representing the edge type in the form (source_node_type, relation_type, target_node_type).

        Returns:
            Tensor: Probabilities for each edge, indicating the likelihood of edge existence.
        """

        # Unpack the source and target node types from the edge type
        source_node_type, _, target_node_type = target_edge

        # Extract source and target indices for edges
        src_indices, tgt_indices = edge_label_index

        # Retrieve the embeddings for the source and target nodes
        src_embeddings = z_dict[source_node_type][src_indices]
        tgt_embeddings = z_dict[target_node_type][tgt_indices]

        # Compute the dot product between source and target embeddings
        edge_scores = (src_embeddings * tgt_embeddings).sum(dim=1)

        # Apply sigmoid to map scores to probabilities
        edge_probabilities = torch.sigmoid(edge_scores)

        return edge_probabilities


class LinkPredModel(pl.LightningModule):
    def __init__(self,
                 target_edge: tuple[str, str, str],
                 metadata,
                 gnn_kwargs: GATConfig,
                 learning_rate: float
                 ):
        super().__init__()
        self.encoder = to_hetero(GAT(**asdict(gnn_kwargs)),
                                 metadata=metadata,
                                 aggr='sum')
        self.decoder = EdgeDecoder()
        self.target_edge = target_edge
        self.learning_rate = learning_rate

        # Initialize the metric objects
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.best_acc = 0

    def forward(self, batch):
        # Encode the graph data to get node embeddings
        z_dict = self.encoder(batch.x_dict, batch.edge_index_dict)

        # Predict probabilities for edge labels
        return self.decoder(z_dict=z_dict,
                            edge_label_index=batch[self.target_edge].edge_label_index,
                            target_edge=self.target_edge
                            )

    def model_step(self, batch, step_name):
        pred = self(batch)
        target = batch[self.target_edge].edge_label
        loss = binary_cross_entropy(pred, target)
        batch_size = batch[self.target_edge].edge_label.shape[0]
        self.log(name=f'{step_name}_loss',
                 value=loss,
                 batch_size=batch_size,
                 prog_bar=True,
                 )

        if step_name == 'val':
            # Calculate metrics only for the validation step
            self.accuracy(pred, target)
            self.precision(pred, target)
            self.recall(pred, target)

        return loss

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        self.best_acc = max(acc, self.best_acc)
        self.log(name='val_accuracy', value=acc)
        self.log(name='val_best_accuracy', value=self.best_acc)
        self.log(name='val_precision', value=self.precision.compute())
        self.log(name='val_recall', value=self.recall.compute())

    def training_step(self, batch):
        return self.model_step(batch, 'train')

    def validation_step(self, batch):
        return self.model_step(batch, 'val')

    def test_step(self, batch):
        return self.model_step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
