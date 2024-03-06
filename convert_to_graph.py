import polars as pl
import sqlite3
from torch_geometric.data import HeteroData
import torch
import pandas as pd
from torch_geometric.utils import to_undirected

CONNECTION = sqlite3.connect(r'data/chinook.db')


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def load_node(df: pd.DataFrame,
              index_col: str,
              encoders: dict = None,
              ):
    # The load_node procedure loads a dataframe and encodes each column as directed
    df = df.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge(df: pd.DataFrame,
              src_index_col: str,
              src_mapping: dict,
              dst_index_col: str,
              dst_mapping: dict,
              encoders=None,
              ):
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def torch_encoder(df: pd.DataFrame, dtype: torch.dtype):
    return torch.from_numpy(df.values).view(-1, 1).to(dtype)


def determine_nodes_and_edges():
    # Get table names and ignore system tables that contain 'sql'
    tables = pl.read_database("SELECT name FROM sqlite_master WHERE type='table' and name not like '%sql%'", CONNECTION)

    # Get primary keys
    # PK field will increment by 1 for ones that are PK and will be zero for non-pk
    primary_keys = pl.concat([(pl.read_database(f"PRAGMA table_info({t})", CONNECTION)
                               .filter(pl.col('pk') > 0)
                               .select(table=pl.lit(t),
                                       key=pl.col('name')
                                       ))
                              for t in tables['name']])

    # Select tables to be nodes if they are not junction tables
    # This is very basic logic because we are assuming each table has a single primary key
    # Tables with multiple primary keys are assumed to be junction tables
    nodes = (primary_keys
             .filter(pl.col('table')
                     .is_in(primary_keys
                            .group_by('table')
                            .len()
                            .filter(pl.col('len') == 1)['table']
                            )
                     )
             )

    # Get joins of foreign key to primary keys for any tables that will be nodes
    joins = pl.concat([pl.read_database(f"PRAGMA foreign_key_list({t})", CONNECTION)
                      .select(source_table=pl.lit(t),
                              source_id=pl.col('from'),
                              target_table=pl.col('table'),
                              target_id=pl.col('to')
                              )

                       for t in nodes['table'].to_list()])

    # Relational DBs join data on PK -> FK
    # Graphs DBs join data on PK pairs instead, PK -> PK
    # We will need to adjust our join data for this
    joins = (joins.join(primary_keys, left_on='source_table', right_on='table')
             .rename({'source_id': 'target_id',
                      'key': 'source_id',
                      'target_id': 'target_table_pk'
                      })
             )

    # Get many-to-many mapping tables aka junctions, these will become edges
    junctions = (primary_keys
                 .filter(pl.col('table')
                         .is_in(primary_keys
                                .group_by('table')
                                .len()
                                .filter(pl.col('len') == 2)['table']
                                )
                         )
                 )

    # Add the junction tables we've identified to our join data
    # junction table name can server as edge name for now
    junctions = junctions.join(primary_keys.filter(pl.col('table').is_in(nodes['table'])), on='key').rename(
        {'table': 'junction_table', 'table_right': 'table'})

    # Data is in [edge, table, key] format
    # need to reformat to [edge, source table, source id, target table, target id]
    # will treat first record per junction as source and second as target
    junctions = junctions.with_columns((pl.arange(0, pl.len()) % 2 == 0).alias("is_source"))

    # Split by source and target
    source_df = junctions.filter(pl.col("is_source")).drop("is_source").rename(
        {"key": "source_id", "table": "source_table"})
    target_df = junctions.filter(~pl.col("is_source")).drop("is_source").rename(
        {"key": "target_id", "table": "target_table"})

    # rejoin into desired format
    junctions = source_df.join(target_df, on="junction_table")

    # combine with join data for final listing of joins for edge list
    edges = pl.concat([joins, junctions], how='diagonal')

    return {'nodes': nodes, 'edges': edges}


def get_node_data(nodes):
    # Get all table data
    tables_dfs = {t: pl.read_database(f'select * from {t}', CONNECTION) for t in nodes['table']}

    # Get foreign keys for removal
    foreign_keys = pl.concat([pl.read_database(f"PRAGMA foreign_key_list({t})", CONNECTION)
                             .select(source_table=pl.lit(t),
                                     foriegn_id=pl.col('from'),
                                     )

                              for t in nodes['table']])

    tables_dfs = {t: df.drop(foreign_keys.filter(pl.col('source_table') == t)['foriegn_id'].to_list()) for t, df in
                  tables_dfs.items()}

    # Drop non-numeric data
    tables_dfs = {k: v.select(pl.col(pl.NUMERIC_DTYPES)) for k, v in tables_dfs.items()}

    return {'nodes': nodes, 'dfs': tables_dfs}


def get_edge_data(edges):
    def get_edge_pairs(source_table, source_column, target_column):
        query = f"""
        SELECT {source_column}, {target_column}
        FROM {source_table} 
        """
        return pl.read_database(query, CONNECTION)

    # use junction table as pair source table where appropriate
    edges = edges.with_columns(pair_source_table=pl.coalesce([pl.col('junction_table'),
                                                              pl.col('source_table')])
                               )

    edge_pairs = {
        (row['source_table'], 'has' + row['target_id'].replace('Id', ''), row['target_table']): get_edge_pairs(
            source_table=row['pair_source_table'],
            source_column=row['source_id'],
            target_column=row['target_id']).drop_nulls()
        for row in edges.iter_rows(named=True)
    }

    return edge_pairs


def format_graph(node_data, edge_data) -> HeteroData:
    hgraph = HeteroData()
    mapping_dict = {}
    for node, key in node_data['nodes'].iter_rows():
        node_df = node_data['dfs'][node]
        node_enc = {col: IdentityEncoder(dtype=torch.float) for col in node_df.columns if col != key}
        if len(node_enc) == 0:
            node_enc = None
        hgraph[node].x, mapping_dict[node] = load_node(df=node_df.to_pandas(),
                                                       index_col=key,
                                                       encoders=node_enc
                                                       )
        hgraph[node].num_nodes = len(mapping_dict[node])

    for edge, pairs in edge_data.items():
        source = edge[0]
        target = edge[2]
        hgraph[edge].edge_index, _ = load_edge(df=pairs.to_pandas(),
                                               src_mapping=mapping_dict[source],
                                               src_index_col=pairs.columns[0],
                                               dst_mapping=mapping_dict[target],
                                               dst_index_col=pairs.columns[1])

    return hgraph


def add_degree(hgraph: HeteroData):
    edges = hgraph.metadata()[1]
    for node in hgraph.metadata()[0]:
        degrees = []
        for edge in edges:
            if node in edge:
                # pos will be 0 or 2, but needs to be 0 or 1
                edge_pos = min(edge.index(node), 1)
                edge_index = hgraph[edge].edge_index[edge_pos]
                degrees.append(pl.DataFrame(edge_index.numpy()).group_by('column_0').len())
        total_degrees = pl.concat(degrees).group_by('column_0').sum()
        degree_mapping = {k: v for k, v in total_degrees.iter_rows()}
        degree_tensor = torch.FloatTensor([degree_mapping[i] if i in degree_mapping else 0
                                           for i in range(hgraph[node].num_nodes)
                                           ])
        degree_tensor = torch.unsqueeze(degree_tensor, dim=1)
        if hasattr(hgraph[node], 'x'):
            hgraph[node].x = torch.cat([hgraph[node].x, degree_tensor], dim=1)
        else:
            hgraph[node].x = degree_tensor
        del hgraph[node].num_nodes

    return hgraph


def convert_to_undirected(hgraph: HeteroData):
    for edge in hgraph.metadata()[1]:
        hgraph[edge].edge_index = to_undirected(hgraph[edge].edge_index)

    return hgraph


def reverse(hgraph: HeteroData):
    for edge in hgraph.metadata()[1]:
        if edge[0] != edge[2]:
            reverse_edge = (edge[2], edge[1], edge[0])
            hgraph[reverse_edge].edge_index = torch.stack([hgraph[edge].edge_index[1], hgraph[edge].edge_index[0]])

    return hgraph

def main():
    objs = determine_nodes_and_edges()
    edge_data = get_edge_data(objs['edges'])
    node_data = get_node_data(objs['nodes'])
    graph = format_graph(node_data=node_data, edge_data=edge_data)
    # add degree before converting edges to undirected
    graph = add_degree(graph)
    graph = reverse(graph)
    print(graph)
    torch.save(graph, r'data/graph.bin')


if __name__ == '__main__':
    main()
