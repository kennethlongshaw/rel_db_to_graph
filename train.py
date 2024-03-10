import pytorch_lightning as pl
import torch

from dataloader import HeteroGraphLinkDataModule, SplitConfig
from model import LinkPredModel, GATConfig
from pytorch_lightning.callbacks import ModelCheckpoint


def train():
    pl.seed_everything(seed=42, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    data = torch.load(f'data/graph.bin')
    target_edge = ('playlists', 'hasTrack', 'tracks')

    num_layers = 3
    num_neighbors = 10

    split_config = SplitConfig(is_undirected=False,
                               edge_types=target_edge,
                               num_val=.1,
                               num_test=.00,
                               add_negative_train_samples=True,
                               )

    datamodule = HeteroGraphLinkDataModule(data=data,
                                           target_edge=target_edge,
                                           split_config=split_config,
                                           batch_size=32,
                                           num_neighbors=[num_neighbors] * num_layers,
                                           shuffle=True
                                           )

    gat_config = GATConfig(
        in_channels=(-1, -1),
        hidden_channels=10,
        num_layers=num_layers,
        dropout=.1,
        norm='BatchNorm',
        add_self_loops=False,
        v2=True
    )

    model = LinkPredModel(target_edge=target_edge,
                          metadata=data.metadata(),
                          gnn_kwargs=gat_config,
                          lr=0.0001
                          )

    score = 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor=score,
                                          mode='min',
                                          verbose=True,
                                          )

    trainer = pl.Trainer(deterministic=True,
                         max_epochs=10,
                         enable_progress_bar=True,
                         accelerator='cuda',
                         # precision='bf16-true',
                         callbacks=[checkpoint_callback],
                         )

    trainer.fit(model=model, datamodule=datamodule)

    score = checkpoint_callback.best_model_score
    print(f'Best score was {score}')


if __name__ == '__main__':
    train()
