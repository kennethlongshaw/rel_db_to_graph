# Transforming Relational Databases with GNNs: The ChinookDB Graph Model Example Project

Unlock the full potential of relational databases through the power of Graph Neural Networks (GNNs). This project exemplifies the transformative approach of converting the Chinook sample database—a comprehensive model of a digital media store—into a graph database format. We then apply a GNN to predict linkages among songs in playlists, showcasing the unique capabilities of GNNs to uncover complex relationships and patterns within traditional database structures.

## Harnessing GNNs for Relational Data

Leveraging DVC pipelines, our workflow transitions from downloading the Chinook database to crafting a graph structure, culminating in the training of a GNN for sophisticated link prediction. This project not only demonstrates the conversion of a standard relational database into a graph format but also highlights the innovative applications of GNNs in predicting and analyzing relationships within data, offering a glimpse into the future of data analysis and machine learning.

### Quick Project Insight

- **DVC Pipelines**: Streamline the transition from data acquisition to model training.
- **Chinook Database**: An SQLite sample embodying a digital media store's complexity.
- **convert_to_graph.py**: Transforms relational data into a graph database leveraging PyTorch Geometric's heterograph object.
- **train.py**: Employs a GNN to predict playlist compositions, illuminating the potential of link prediction in graph data.

### The Power of Graph-Based Analysis

With 11 intricately related tables, the Chinook database serves as an ideal candidate to illustrate how graph-based analysis can unravel the complex interconnections within a digital media store's dataset, offering new perspectives and insights that conventional analysis might overlook.

### Related Slides
https://docs.google.com/presentation/d/e/2PACX-1vTzzX1VYpXD4L7Is9gh3nP_NTyRdW3u1XJQBtRiTZwK806RNnWcm-2zgA5RhBDcm8AsYVShIYe75Pwk/pub?start=false&loop=false&delayms=3000

### Requirements
**For CPU:**
`pip install -r requirements_cpu.txt`

**For GPU:**
`pip install -r requirements_gpu.txt`

## Usage
To run the project pipeline:
1. Initialize DVC with `dvc init`
2. Pull the data and run the pipeline with `dvc repro`. This command downloads the Chinook database, converts it to a graph format, and trains the GNN model.
3. The training will produce a checkpoint model like `DvcLiveLogger\\dvclive_run\\checkpoints\\epoch=3-step=60-v2.ckpt` To run inference on the model, update `inference.py` with the checkpoint path.
4. Run inference: `streamlit run inference.py`

Embark on a journey with us as we demonstrate the untapped potential of GNNs to revolutionize the analysis of relational databases, transcending traditional boundaries and unlocking a new realm of possibilities.

