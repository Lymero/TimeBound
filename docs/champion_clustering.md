# Champion Clustering

This document describes the champion clustering functionality in the Winrates Analysis Tool.

## Overview

The champion clustering module provides functionality for grouping champions based on the similarity of their win rate graphs. It uses correlation analysis to identify champions with similar performance patterns across game duration and groups them into clusters using various strategies.

## Key Components

### ChampionPathClusterer

The `ChampionPathClusterer` class is the main component of the champion clustering module. It provides methods for:

- Loading champion SVG path data
- Extracting points from SVG paths
- Computing correlations between champions
- Clustering champions using different strategies
- Finding similar champions
- Generating cluster statistics

### Clustering Strategies

The system implements multiple clustering strategies through a strategy pattern:

- **ThresholdClusteringStrategy**: Ensures champions in the same cluster have at least a minimum correlation with each other
- **HierarchicalClusteringStrategy**: Uses hierarchical clustering algorithms to group champions
- **SpectralClusteringStrategy**: Uses spectral clustering techniques that work well for graph-based data

These strategies are created through a `ClusteringStrategyFactory` that allows easy selection of the appropriate algorithm.

### ClusterVisualizer

The `ClusterVisualizer` class provides multiple visualization methods:

- **Cluster scatter plots**: Visualize clusters in 2D space using t-SNE
- **Interactive visualizations**: Generate interactive HTML visualizations 
- **Correlation networks**: Show the champion similarity relationships as a network
- **Cluster profiles**: Plot the win rate curves for all champions in each cluster
- **Clustering metrics**: Evaluate and visualize the quality of clustering results

## Usage

### Basic Usage

```python
from src.analysis.champion_path_clustering import ChampionPathClusterer
from src.analysis.champion_path_clustering.factory import ClusteringStrategyFactory

# Create a clustering strategy
strategy = ClusteringStrategyFactory.create_strategy("threshold")

# Create a champion clusterer with the selected strategy
clusterer = ChampionPathClusterer(
    svg_paths_file="data/champion_svg_paths.json",
    correlation_threshold=0.8,
    clustering_strategy=strategy
)

# Cluster champions
clusters = clusterer.cluster_champions(n_clusters=0)  # 0 for automatic determination

# Get cluster statistics
stats = clusterer.get_cluster_stats(clusters)
print(f"Cluster stats: {stats}")

# Find similar champions to a specific champion
similar_champions = clusterer.get_similar_champions("Ahri")
for champion, similarity in similar_champions:
    print(f"{champion}: {similarity:.4f}")
```

### Visualization Example

```python
from src.visualization.cluster_visualizer import ClusterVisualizer

# Create a visualizer
visualizer = ClusterVisualizer()

# Get champion list and prepare distance matrix
champions = list(clusterer.champion_points.keys())
distance_matrix = 1 - np.array([
    [clusterer.correlation_matrix.get(c1, {}).get(c2, 0) for c2 in champions]
    for c1 in champions
])

# Map champions to their clusters
champion_to_cluster = {
    champion: cluster_id
    for cluster_id, champions_in_cluster in clusters.items()
    for champion in champions_in_cluster
}

# Visualize clusters
visualizer.visualize_clusters(
    champions=champions,
    distance_matrix=distance_matrix,
    champion_to_cluster=champion_to_cluster,
    output_path="data/visualizations/champion_clusters.png"
)

# Create interactive visualization
visualizer.visualize_clusters_interactive(
    champions=champions,
    distance_matrix=distance_matrix,
    champion_to_cluster=champion_to_cluster,
    output_path="data/visualizations/champion_clusters_interactive.html"
)

# Visualize correlation network
visualizer.visualize_correlation_network(
    champions=champions,
    correlation_matrix=clusterer.correlation_matrix,
    correlation_threshold=clusterer.correlation_threshold,
    output_path="data/visualizations/correlation_network.png"
)

# Visualize cluster profiles
visualizer.visualize_cluster_profiles(
    champion_points=clusterer.champion_points,
    clusters=clusters,
    output_dir="data/visualizations/cluster_profiles",
    prefix="cluster"
)
```

### Example Script

The project includes an example script that demonstrates how to use the champion clustering functionality:

```
python -m src.examples.champion_path_clustering_example --clustering-method threshold --correlation-threshold 0.85 --interactive
```

This script accepts various command-line arguments:
- `--svg-paths-file`: Path to JSON file with champion SVG paths
- `--correlation-threshold`: Minimum correlation for similar champions
- `--n-clusters`: Number of clusters (0 for automatic)
- `--output-dir`: Directory for saving outputs
- `--champion`: Find champions similar to this specific one
- `--limit`: Limit number of champions to process
- `--clustering-method`: Clustering strategy to use (threshold, hierarchical, spectral)
- `--interactive`: Generate interactive HTML visualizations
- `--evaluate-metrics`: Evaluate and visualize clustering metrics


## Visualization Features

The champion clustering module provides several visualization methods:

### Cluster Scatter Plot
Shows champions as points in 2D space with different colors for different clusters, using t-SNE for dimensionality reduction.

### Interactive Visualization
Creates an interactive HTML version of the cluster visualization that allows hovering over points to see champion names.

### Correlation Network
Visualizes champions as nodes in a network with edges representing correlations above the threshold. Edge width represents correlation strength.

### Cluster Profiles
For each cluster, plots the win rate curves of all champions in that cluster to show the similar patterns.

### Clustering Metrics
Evaluates clustering quality using metrics like silhouette score and visualizes these metrics for different clustering methods.

## Technical Details

### Data Extraction
Champion win rate data is extracted from SVG paths using the `SVGPathParser` class from the `src.processing.svg_parser` module.

### Correlation Calculation
The correlation between champions is calculated using the `compare` function from the `src.analysis.graph_correlation` module, which:
- Is invariant to vertical shifts (y-axis shifts)
- Is sensitive to horizontal shifts (x-axis shifts)
- Is invariant to vertical scaling (y-axis scaling)

### Clustering Algorithms

#### Threshold Clustering
- Creates a graph where nodes are champions
- Adds edges between champions with correlation above the threshold
- Grows clusters by adding champions that have high correlation with all existing members
- Ensures all champions in a cluster have minimum correlation with all other members

#### Hierarchical Clustering
- Uses agglomerative clustering from scikit-learn
- Merges clusters based on similarity
- Creates a dendrogram structure
- Cuts the dendrogram at the appropriate level to get the desired number of clusters

#### Spectral Clustering
- Constructs a similarity graph from the correlation matrix
- Uses eigenvectors of the graph Laplacian
- Works well for non-convex clusters

## Dependencies

- `networkx`: For graph-based clustering and network visualization
- `matplotlib`: For static visualizations
- `plotly`: For interactive visualizations
- `numpy`: For numerical operations
- `scikit-learn`: For clustering algorithms and dimensionality reduction
- `pandas`: For data processing
- `seaborn`: For enhanced visualizations
- `adjustText`: For text placement in plots 