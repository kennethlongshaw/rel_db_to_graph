from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomLinkSplit
from config_maker import create_config_from_class
from pytorch_lightning import LightningModule
from dataclasses import asdict


SplitConfig = create_config_from_class(RandomLinkSplit)


def make_link_data_loader(data: HeteroData,
                          target_edge: tuple[str, str, str],
                          split_config: SplitConfig,
                          batch_size: int,
                          num_neighbors: list[int],
                          ) -> LightningModule:

    assert target_edge in data.metadata()[1], "Target edge not present in data set provided"
    random_split = RandomLinkSplit(**asdict(split_config))

    train_data, val_data, test_data = random_split(data)

    return LightningLinkData(
        data=data,
        loader='neighbor',
        batch_size=batch_size,
        num_neighbors=num_neighbors,

        input_train_edges=(target_edge, train_data[target_edge].edge_index),
        input_train_labels=train_data[target_edge].edge_label,

        input_val_edges=(target_edge, val_data[target_edge].edge_index),
        input_val_labels=val_data[target_edge].edge_label,

        input_test_edges=(target_edge, test_data[target_edge].edge_index),
        input_test_labels=test_data[target_edge].edge_label
    )
