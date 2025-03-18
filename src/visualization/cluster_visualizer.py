"""
Cluster Visualization Module.

This module provides visualization utilities for champion clustering results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from adjustText import adjust_text
from sklearn.manifold import TSNE

from src.utils.logger import info


class ClusterVisualizer:
    """
    A class to visualize clustering results for champion clustering results.

    This class provides methods to create static and interactive visualizations
    of clustering results, including cluster scatter plots, correlation networks,
    cluster size comparisons, win rate profiles, and clustering metrics.
    """

    def __init__(self, dpi: int = 300, figsize: tuple[int, int] = (12, 10)) -> None:
        """
        Initialize the ClusterVisualizer.

        Args:
            dpi: Resolution for saved figures
            figsize: Default figure size as (width, height) in inches
        """
        self.dpi = dpi
        self.figsize = figsize

    def _apply_tsne(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction to a distance matrix.

        Args:
            distance_matrix: Distance matrix between data points

        Returns:
            2D embeddings of the data points
        """
        n_samples = distance_matrix.shape[0]
        perplexity = min(30, max(5, n_samples // 3))
        learning_rate = max(200, n_samples / 12)

        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            random_state=42,
            perplexity=perplexity,
            early_exaggeration=12,
            learning_rate=learning_rate,
            init="random",
            max_iter=1000,
            n_iter_without_progress=150,
        )

        return tsne.fit_transform(distance_matrix)

    def _save_or_show_plot(self, output_path: str | None = None) -> None:
        """
        Save the current plot to a file or display it.

        Args:
            output_path: Path to save the visualization, or None to display
        """
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            info(f"Saved visualization to {output_path}")
        else:
            plt.show()

    def _prepare_cluster_data(
        self,
        champions: list[str],
        distance_matrix: np.ndarray,
        champion_to_cluster: dict[str, int],
    ) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Prepare common data needed for cluster visualizations.

        Args:
            champions: List of champion names
            distance_matrix: Distance matrix between champions
            champion_to_cluster: Mapping from champion names to cluster IDs

        Returns:
            Tuple containing:
            - embeddings: 2D t-SNE embeddings of the data points
            - unique_clusters: Sorted list of unique cluster IDs
            - cluster_ids: List of cluster IDs for each champion
        """
        # Get 2D coordinates via t-SNE
        embeddings = self._apply_tsne(distance_matrix)

        # Get unique clusters and cluster IDs
        unique_clusters = sorted(set(champion_to_cluster.values()))
        cluster_ids = [champion_to_cluster[c] for c in champions]

        return embeddings, unique_clusters, cluster_ids

    def visualize_clusters(
        self,
        champions: list[str],
        distance_matrix: np.ndarray,
        champion_to_cluster: dict[str, int],
        title: str = "Champion Clusters based on Win Rate vs Game Length",
        output_path: str | None = None,
        figsize: tuple[int, int] | None = None,
        text_size: int = 8,
        text_alpha: float = 0.9,
        point_size: int = 50,
    ) -> None:
        """
        Visualize champion clusters using t-SNE for dimensionality reduction.

        Args:
            champions: List of champion names
            distance_matrix: Distance matrix between champions
            champion_to_cluster: Mapping from champion names to cluster IDs
            title: Plot title
            output_path: Path to save the visualization, or None to display
            figsize: Figure size as (width, height) in inches
            text_size: Font size for champion labels
            text_alpha: Transparency for text labels (0-1)
            point_size: Size of the scatter points
        """
        embeddings, unique_clusters, _ = self._prepare_cluster_data(
            champions, distance_matrix, champion_to_cluster
        )

        _, ax = plt.subplots(figsize=figsize or self.figsize)
        palette = sns.color_palette("tab20", len(unique_clusters))
        ax.grid(alpha=0.2, linestyle="--")

        texts = []
        for i, champion in enumerate(champions):
            cluster_id = champion_to_cluster[champion]
            color_idx = unique_clusters.index(cluster_id)

            ax.scatter(
                embeddings[i, 0],
                embeddings[i, 1],
                s=point_size,
                color=palette[color_idx],
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

            texts.append(
                ax.text(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    champion,
                    fontsize=text_size,
                    alpha=text_alpha,
                    ha="center",
                    va="center",
                    weight="bold",
                    bbox={
                        "facecolor": "white", 
                        "alpha": 0.4, 
                        "pad": 0.1, 
                        "boxstyle": "round,pad=0.1"
                    },
                )
            )

        # Apply text collision avoidance
        adjust_text(
            texts,
            arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6},
            expand_points=(1.5, 1.5),
            force_points=(0.1, 0.25),
        )

        plt.title(title, fontsize=14, weight="bold", pad=20)

        # Create legend with sorted cluster IDs
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette[unique_clusters.index(i)],
                markersize=10,
                label=f"Cluster {i}",
            )
            for i in sorted(unique_clusters)
        ]

        ax.legend(
            handles=handles,
            title="Champion Clusters",
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgray",
            fancybox=True,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
        ).get_title().set_fontweight("bold")

        # Remove axis ticks since t-SNE dimensions don't have meaningful interpretation
        plt.tick_params(axis="both", which="both", length=0)
        plt.xlabel("t-SNE Dimension 1", fontsize=10)
        plt.ylabel("t-SNE Dimension 2", fontsize=10)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("lightgray")
            spine.set_linewidth(0.5)

        self._save_or_show_plot(output_path)

    def visualize_clusters_interactive(
        self,
        champions: list[str],
        distance_matrix: np.ndarray,
        champion_to_cluster: dict[str, int],
        title: str = "Champion Clusters based on Win Rate vs Game Length",
        output_path: str | None = None,
        point_size: int = 12,
    ) -> None:
        """
        Create an interactive visualization of champion clusters using Plotly.

        Args:
            champions: List of champion names
            distance_matrix: Distance matrix between champions
            champion_to_cluster: Mapping from champion names to cluster IDs
            title: Plot title
            output_path: Path to save the HTML visualization, or None to display
            point_size: Size of the scatter points
        """
        embeddings, unique_clusters, _ = self._prepare_cluster_data(
            champions, distance_matrix, champion_to_cluster
        )

        df = pd.DataFrame(
            {
                "x": embeddings[:, 0],
                "y": embeddings[:, 1],
                "champion": champions,
                "cluster": [f"Cluster {champion_to_cluster[c]}" for c in champions],
            }
        )

        color_sequence = (
            px.colors.qualitative.Bold
            + px.colors.qualitative.Pastel
            + px.colors.qualitative.Safe
        )
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_name="champion",
            color_discrete_sequence=color_sequence[: len(unique_clusters)],
            size=[point_size] * len(champions),
            title=title,
            category_orders={
                "cluster": [f"Cluster {i}" for i in sorted(unique_clusters)]
            },
        )

        fig.update_layout(
            template="plotly_white",
            legend_title_text="Cluster",
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hoverlabel={"bgcolor": "white", "font_size": 14, "font_family": "Arial"},
        )

        if output_path:
            fig.write_html(output_path)
            info(f"Saved interactive cluster visualization to {output_path}")
        else:
            fig.show()

    def visualize_correlation_network(
        self,
        champions: list[str],
        correlation_matrix: dict[str, dict[str, float]],
        correlation_threshold: float,
        output_path: str | None = None,
    ) -> None:
        """
        Visualize the correlation network as a graph.

        Args:
            champions: List of champion names
            correlation_matrix: Correlation matrix between champions
            correlation_threshold: Threshold for considering two champions as similar
            output_path: Path to save the visualization, or None to display
        """
        G = nx.Graph()
        for champion in champions:
            G.add_node(champion)

        for champ1, correlations in correlation_matrix.items():
            for champ2, score in correlations.items():
                if champ1 != champ2 and score >= correlation_threshold:
                    G.add_edge(champ1, champ2, weight=score)

        plt.figure(figsize=(20, 20))

        pos = nx.spring_layout(G, seed=42, k=2, iterations=200)
        edges = list(G.edges())

        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="gray", alpha=0.6)
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", alpha=0.9, edgecolors="white"
        )
        nx.draw_networkx_labels(
            G,
            pos,
            font_weight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "lightgray",
                "boxstyle": "round,pad=0.4",
                "linewidth": 0.5,
            },
        )

        title = f"Champion Correlation Network (threshold={correlation_threshold})"
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout(pad=2.0)

        self._save_or_show_plot(output_path)

    def visualize_cluster_profiles(
        self,
        champion_points: dict[str, list[tuple[float, float]]],
        clusters: dict[int, list[str]],
        output_dir: str,
        prefix: str = "cluster_profiles_cluster",
    ) -> None:
        """
        Visualize the win rate profiles for each cluster.

        Args:
            champion_points: Dictionary mapping champion names to their points
            clusters: Dictionary mapping cluster IDs to lists of champions
            output_dir: Directory to save the visualizations
            prefix: Prefix for the output filenames
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for cluster_id, champions in clusters.items():
            plt.figure(figsize=(12, 12))
            colormap = plt.cm.get_cmap("viridis", len(champions))

            for i, champion in enumerate(champions):
                points = champion_points.get(champion, [])
                if points:
                    x, y = zip(*points, strict=False)
                    plt.plot(x, y, label=champion, alpha=0.7, color=colormap(i))

            plt.title(f"Win Rate Profiles for Cluster {cluster_id}")
            plt.xlabel("Game Length (minutes)")
            plt.ylabel("Win Rate")
            plt.grid(True, linestyle="--", alpha=0.7)

            # Place legend outside the plot to the right
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=14,
                ncol=2,
                frameon=True,
                framealpha=0.8,
                fancybox=True,
                title=f"Champions in Cluster {cluster_id}",
            )

            plt.tight_layout()
            plt.subplots_adjust(right=0.75)

            output_path = f"{output_dir}/{prefix}_{cluster_id}.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()

            info(f"Saved cluster profile visualization for cluster {cluster_id} to {output_path}")
