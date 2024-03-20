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

### Installation Instructions

Ensure your environment is prepared to harness the power of GNNs:

**For CPU:**
`pip install -r requirements_cpu.txt`

**For GPU:**
`pip install -r requirements_gpu.txt`

## Quickstart Guide

1. **Initialize Your Project**: Set the stage with `dvc init`.
2. **Activate the Pipeline**: Employ `dvc repro` to navigate from data acquisition to GNN training effortlessly.
3. **Locate Your Model**: Training finalizes with a model checkpoint, e.g., `DvcLiveLogger/dvclive_run/checkpoints/epoch=3-step=60-v2.ckpt`.
4. **Deploy Inference**: Update `inference.py` with your model's checkpoint and launch your predictions into action with `streamlit run inference.py`.

Embark on a journey with us as we demonstrate the untapped potential of GNNs to revolutionize the analysis of relational databases, transcending traditional boundaries and unlocking a new realm of possibilities.

