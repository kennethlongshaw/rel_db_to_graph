import pytorch_lightning as pl
import torch
from dataloader import HeteroGraphLinkDataModule, SplitConfig
from model import LinkPredModel, GATConfig, TrainConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from dvclive.lightning import DVCLiveLogger
from dvc.api import params_show
from setup import setup


def train():
    setup()

    params = params_show()['train']

    data = torch.load(f'data/graph.bin')
    target_edge = params['target_edge']

    reverse_edge = (target_edge[2], 'REVERSE_' + target_edge[1], target_edge[0])
    if reverse_edge not in data.metadata()[1]:
        reverse_edge = None

    split_config = SplitConfig(is_undirected=False,
                               edge_types=params['target_edge'],
                               rev_edge_types=reverse_edge,
                               num_val=params['num_val'],
                               num_test=params['num_test'],
                               add_negative_train_samples=False
                               )

    train_cfg = TrainConfig(num_layers=params['num_layers'],
                            num_neighbors=params['num_neighbors'],
                            dropout=params['dropout'],
                            learning_rate=params['learning_rate'],
                            batch_size=params['batch_size'],
                            epochs=params['epochs']
                            )

    gat_config = GATConfig(
        in_channels=(-1, -1),
        hidden_channels=params['hidden_channels'],
        num_layers=train_cfg.num_layers,
        dropout=params['dropout'],
        norm=params['norm'],
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
                                          save_top_k=1
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
