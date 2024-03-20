# ChinookDB Graph Model Project
This project demonstrates the conversion of the Chinook sample database—a digital media store model—to a graph database format, and subsequently trains a Graph Neural Network (GNN) for link prediction among songs in playlists. The Chinook database includes tables for artists, albums, media tracks, invoices, and customers, simulating a real-world digital media store environment with data modeled after an Apple iTunes library.

## Overview
The project utilizes DVC pipelines to manage the workflow, which includes downloading the Chinook database, converting the relational database to a graph structure, and training a GNN for link prediction. This approach showcases how to leverage a graph model on an existing relational database in a self-supervised manner.

## Project Structure
DVC Pipelines: Orchestrates the workflow from downloading the database to training the model.
Chinook Database: A sample SQLite database that models a digital media store.
convert_to_graph.py: A script that converts the relational structure of the Chinook database into a graph database format using the heterograph object in PyTorch Geometric.
train.py: Trains a Graph Neural Network using link prediction to forecast playlist compositions based on existing songs.

## Database Model
The Chinook database comprises 11 tables including artists, albums, tracks, playlists, customers, invoices, and more. It models the intricacies of a digital media store, making it an ideal candidate for graph-based analysis.

CPU
`pip install -r requirements_gpu.txt`

GPU
`pip install -r requirements_cpu.txt`

## Usage
To run the project pipeline:
1. Initialize DVC with `dvc init`
2. Pull the data and run the pipeline with `dvc repro`. This command downloads the Chinook database, converts it to a graph format, and trains the GNN model.
3. The training will produce a checkpoint model like `DvcLiveLogger\\dvclive_run\\checkpoints\\epoch=3-step=60-v2.ckpt` To run inference on the model, update `inference.py` with the checkpoint path.
4. Run inference: `streamlit run inference.py`

