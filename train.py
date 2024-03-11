import pytorch_lightning as pl
import torch
from dataloader import HeteroGraphLinkDataModule, SplitConfig
from model import LinkPredModel, GATConfig, TrainConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from dvclive.lightning import DVCLiveLogger


def train():
    pl.seed_everything(seed=42, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    data = torch.load(f'data/graph.bin')
    target_edge = ('playlists', 'hasTrack', 'tracks')

    reverse_edge = (target_edge[2], 'REVERSE_' + target_edge[1], target_edge[0])
    if reverse_edge not in data.metadata()[1]:
        reverse_edge = None

    split_config = SplitConfig(is_undirected=False,
                               edge_types=target_edge,
                               rev_edge_types=reverse_edge,
                               num_val=.1,
                               num_test=.00,
                               add_negative_train_samples=False,
                               )

    train_cfg = TrainConfig(num_layers=3,
                            num_neighbors=30,
                            dropout=.2,
                            learning_rate=.0001,
                            batch_size=64,
                            epochs=20
                            )

    gat_config = GATConfig(
        in_channels=(-1, -1),
        hidden_channels=10,
        num_layers=train_cfg.num_layers,
        dropout=.1,
        norm='BatchNorm',
        add_self_loops=False,
        v2=True
    )

    datamodule = HeteroGraphLinkDataModule(data=data,
                                           target_edge=target_edge,
                                           split_config=split_config,
                                           batch_size=train_cfg.batch_size,
                                           num_neighbors=train_cfg.depth_sizes,
                                           shuffle=True
                                           )

    model = LinkPredModel(target_edge=target_edge,
                          metadata=data.metadata(),
                          gnn_kwargs=gat_config,
                          learning_rate=train_cfg.learning_rate
                          )

    score = 'val_accuracy'
    checkpoint_callback = ModelCheckpoint(monitor=score,
                                          mode='max',
                                          verbose=True,
                                          )

    logger = DVCLiveLogger()

    trainer = pl.Trainer(deterministic=True,
                         max_epochs=train_cfg.epochs,
                         enable_progress_bar=True,
                         accelerator='auto',
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         )

    trainer.fit(model=model, datamodule=datamodule)

    score = checkpoint_callback.best_model_score
    print(f'Best score was {score}')
    logger.log_metrics({f'best_{score}': score})



if __name__ == '__main__':
    train()
