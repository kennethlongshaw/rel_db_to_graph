import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from config_maker import create_config_from_class
from pytorch_lightning import LightningDataModule
from dataclasses import asdict

SplitConfig = create_config_from_class(RandomLinkSplit)


def make_link_data_loader(data: HeteroData,
                          target_edge: tuple[str, str, str],
                          split_config: SplitConfig,
                          batch_size: int,
                          num_neighbors: list[int],
                          ):
    assert target_edge in data.metadata()[1], f"Target edge {target_edge} not present in data set provided"
    if split_config.rev_edge_types:
        if isinstance(split_config.rev_edge_types, tuple):
            assert split_config.rev_edge_types in data.metadata()[
                1], f"Reverse target edge {split_config.rev_edge_types} not present in data set provided"
        elif isinstance(split_config.rev_edge_types, list):
            for rev_edge in split_config.rev_edge_types:
                assert rev_edge in data.metadata()[1], \
                    f"Reverse target edge {rev_edge} not present in data set provided"

    random_split = RandomLinkSplit(**asdict(split_config))

    train_data, val_data, test_data = random_split(data)

    return LightningLinkData(
        data=data,
        loader='neighbor',
        batch_size=batch_size,
        num_neighbors=num_neighbors,

        input_pred_edges=(target_edge, train_data[target_edge].edge_index),
        input_train_labels=train_data[target_edge].edge_label,

        input_val_edges=(target_edge, val_data[target_edge].edge_index),
        input_val_labels=val_data[target_edge].edge_label,

        input_test_edges=(target_edge, test_data[target_edge].edge_index),
        input_test_labels=test_data[target_edge].edge_label,
    )


def split(data: HeteroData,
          target_edge: tuple[str, str, str],
          split_config: SplitConfig,
          batch_size: int,
          shuffle: bool,
          num_neighbors: list[int]):
    assert target_edge in data.metadata()[1], f"Target edge {target_edge} not present in data set provided"
    if split_config.rev_edge_types:
        if isinstance(split_config.rev_edge_types, tuple):
            assert split_config.rev_edge_types in data.metadata()[
                1], f"Reverse target edge {split_config.rev_edge_types} not present in data set provided"
        elif isinstance(split_config.rev_edge_types, list):
            for rev_edge in split_config.rev_edge_types:
                assert rev_edge in data.metadata()[1], \
                    f"Reverse target edge {rev_edge} not present in data set provided"

    splitter = RandomLinkSplit(**asdict(split_config))

    train_data, val_data, test_data = splitter(data)

    train_loader_args = {
        'batch_size': batch_size,
        'num_neighbors': num_neighbors,

    }

    nontrain_loader_args = {
        'batch_size': batch_size,
        'num_neighbors': [-1] * len(num_neighbors),

    }

    train_loader = LinkNeighborLoader(train_data,
                                      edge_label_index=(target_edge, train_data[target_edge].edge_label_index),
                                      edge_label=train_data[target_edge].edge_label,
                                      shuffle=shuffle,
                                      **train_loader_args
                                      )

    val_loader = LinkNeighborLoader(val_data,
                                    edge_label_index=(target_edge, val_data[target_edge].edge_label_index),
                                    edge_label=val_data[target_edge].edge_label,
                                    **nontrain_loader_args
                                    )

    test_loader = LinkNeighborLoader(test_data,
                                     edge_label_index=(target_edge, test_data[target_edge].edge_label_index),
                                     edge_label=test_data[target_edge].edge_label,
                                     **nontrain_loader_args
                                     )

    return {
        'train_data': train_data,
        'train_loader': train_loader,
        'val_data': val_data,
        'val_loader': val_loader,
        'test_data': test_data,
        'test_loader': test_loader
    }


class HeteroGraphLinkDataModule(pl.LightningDataModule):
    """Own implementation of LightningLinkData for fallback"""

    def __init__(self,
                 data: HeteroData,
                 batch_size: int,
                 target_edge: tuple[str, str, str],
                 num_neighbors: list[int],
                 split_config: SplitConfig,
                 shuffle: bool
                 ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.target_edge = target_edge
        self.num_neighbors = num_neighbors
        self.split_config = split_config
        self.shuffle = shuffle

    def setup(self, stage: str) -> None:
        self.split_data = split(data=self.data,
                                batch_size=self.batch_size,
                                num_neighbors=self.num_neighbors,
                                target_edge=self.target_edge,
                                split_config=self.split_config,
                                shuffle=self.shuffle
                                )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.split_data['train_loader']

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.split_data['val_loader']

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.split_data['test_loader']
