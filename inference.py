import streamlit as st
import torch
from model import LinkPredModel, GATConfig
from dvc.api import params_show
import sqlite3
import polars as pl

CONNECTION = sqlite3.connect(r'data/chinook.db')


# @st.cache_resource
def load_model(data):
    checkpoint_path = r'DvcLiveLogger\\dvclive_run\\checkpoints\\epoch=3-step=60-v2.ckpt'
    checkpoint = torch.load(checkpoint_path)

    params = params_show()['train']

    target_edge = tuple(params['target_edge'])

    gat_config = GATConfig(
        in_channels=(-1, -1),
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        dropout=0,
        norm=params['norm'],
        add_self_loops=False,
        v2=True
    )

    model = LinkPredModel.load_from_checkpoint(
        checkpoint_path,
        target_edge=target_edge,
        metadata=data.metadata(),
        gnn_kwargs=gat_config,
        learning_rate=0,
    )

    model.eval()
    model.freeze()

    return model


def predict_songs(data, track_ids: list, top_k: int):
    model = load_model(data)

    # new playlist id
    new_playlist_id = data['playlists'].x.shape[0]

    # add playlist degree with new id
    playlist_degree = torch.tensor([[len(track_ids) + 1]])
    data['playlists'].x = torch.cat([data['playlists'].x, playlist_degree], dim=0)

    # add playlist song edges
    # one edge per track
    source = torch.tensor([new_playlist_id for _ in range(len(track_ids))])
    target = torch.tensor(track_ids)
    data['playlists', 'hasTrack', 'tracks'].edge_index = torch.stack([source, target])
    data['tracks', 'REVERSE_hasTrack', 'playlists'].edge_index = torch.stack([target, source])

    # add self loop of playlist to self
    data['playlists', 'SELF_LOOP', 'playlists'].edge_index = torch.cat(
        [data['playlists', 'SELF_LOOP', 'playlists'].edge_index,
         torch.tensor([[new_playlist_id], [new_playlist_id]])],
        dim=1)

    # Add edge label index for what to predict, use all other songs
    songs_not_on_playlist = list(set(range(0, data['tracks'].x.shape[0])).difference(set(track_ids)))
    source = torch.tensor([new_playlist_id for _ in range(len(songs_not_on_playlist))])
    target = torch.tensor(songs_not_on_playlist)
    data['playlists', 'hasTrack', 'tracks'].edge_label_index = torch.stack([source, target])

    # move to gpu for inference
    data.to(model.device)

    return model(data).topk(top_k)


# @st.cache_data
def load_data():
    return torch.load(f'data/graph.bin')


def main():
    tracks = pl.read_database("""
        SELECT 
        t.Name AS TrackName, 
        a.Name AS ArtistName
        FROM tracks t
        LEFT JOIN albums al ON t.AlbumId = al.AlbumId
        LEFT JOIN artists a ON al.ArtistId = a.ArtistId;
    """, connection=CONNECTION)

    tracks = tracks.with_columns(FullName=pl.concat_str([pl.col('ArtistName'), pl.lit(' - '), pl.col('TrackName')]))
    track_list = tracks['FullName'].to_list()

    # select new songs
    with st.sidebar:
        selected_tracks = st.multiselect(label='Start a playlist', options=track_list)
        track_ids = [track_list.index(song) for song in selected_tracks]
        st.session_state['preds'] = None
        start = st.button('Predict')
        if start:
            st.session_state['preds'] = predict_songs(data=load_data(), track_ids=track_ids, top_k=10)

    st.subheader('Selected Songs')
    st.write(selected_tracks)
    if st.session_state['preds']:
        new_tracks = tracks.with_row_index(name='id').filter(
            pl.col('id').is_in(st.session_state['preds'].indices.tolist()))
        st.subheader('Recommended Songs')
        st.write(new_tracks['FullName'].to_list())


if __name__ == '__main__':
    main()
