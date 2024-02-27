from torch_geometric.nn import GAT, to_hetero, InnerProductDecoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy
import torch
from dataclasses import dataclass, field
import inspect
from typing import Optional, Any

def create_config_from_class(cls):
    """
    Helper function to create a config for any class so that its config can be tracked
    """
    # Inspect the constructor (__init__ method) of the provided class.
    init_signature = inspect.signature(cls.__init__)
    fields = {}  # This will hold field definitions for our dataclass.
    annotations = {}  # This will hold type annotations for our dataclass fields.

    # Iterate over the parameters of the __init__ method.
    for name, param in init_signature.parameters.items():
        # Skip "self" as it's not a field, and also skip "**kwargs" if present.
        if name == "self" or name == "kwargs":
            continue

        # Determine the type hint of the parameter. Use typing.Any if not specified.
        type_hint = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        annotations[name] = type_hint  # Store the type hint in annotations.

        # Check if the parameter has a default value.
        if param.default is inspect.Parameter.empty:
            # For parameters without default values, just create a field without a default.
            fields[name] = field()
        else:
            # For parameters with default values, create a field with the specified default.
            fields[name] = field(default=param.default)

    # Dynamically create the dataclass.
    # The type function is used to create a new class at runtime.
    # f"{cls.__name__}Config" gives the new class a name based on the original class's name.
    # (object,) specifies that the new class inherits from object (common practice for dataclasses).
    # The dictionary passed as the third argument combines annotations and field definitions.
    return dataclass(type(f"{cls.__name__}Config", (object,), {'__annotations__': annotations, **fields}))


GATConfig = create_config_from_class(GAT)


class OptimizerArgs:
    learning_rate: float
    betas: tuple[float, float]
    decay_lr: bool
    min_lr: Optional[float]
    lr_decay_iters: Optional[int]
    weight_decay: Optional[float]


class LinkPredModel(pl.LightningModule):
    def __init__(self,
                 target_edge: tuple[str, str, str],
                 data: HeteroData,
                 gnn_kwargs: GATConfig,
                 optimizer_args: OptimizerArgs
                 ):
        super().__init__()
        self.encoder = to_hetero(GAT(**gnn_kwargs),
                                 data.metadata(),
                                 aggr='sum')
        self.decoder = InnerProductDecoder()
        self.target_edge = target_edge
        self.op_args = optimizer_args

    def forward(self, batch):
        x_dict = self.encoder(x=batch.x_dict,
                              edge_index_dict=batch.edge_index_dict)
        return self.decoder(x_dict, batch[self.target_edge].edge_label_index)

    def model_step(self, batch, step_name):
        pred = self(batch)
        loss = binary_cross_entropy(pred, batch[self.target_edge].edge_label)
        self.log(f'{step_name}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, 'val')

    def configure_optimizers(self):
        """ Regularization Config"""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.op_args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=self.op_args.learning_rate,
                                      betas=self.op_args.betas,
                                      fused=True)

        if self.op_args.decay_lr:
            scheduler = {
                'scheduler': CosineAnnealingLR(optimizer,
                                               T_max=self.op_args.lr_decay_iters,
                                               eta_min=self.op_args.min_lr
                                               ),
                'name': 'cosine_annealing_lr',
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
        else:
            return optimizer
