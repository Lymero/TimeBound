"""Spectral clustering strategy for champion path clustering."""

import numpy as np
from sklearn.cluster import SpectralClustering

from src.utils.logger import error, info

from .base import BaseClusteringStrategy


class SpectralClusteringStrategy(BaseClusteringStrategy):
    """Strategy for spectral clustering of champions."""

    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using spectral clustering.

        This method provides an alternative to the hierarchical clustering approach,
        using spectral clustering which can better capture non-linear relationships
        and complex cluster shapes.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        champions = list(correlation_matrix.keys())
        info(f"Performing spectral clustering on {len(champions)} champions with n_clusters={n_clusters}")

        correlation_array = np.array(
            [
                [correlation_matrix[champ1][champ2] for champ2 in champions]
                for champ1 in champions
            ]
        )

        # Transform correlation to a positive value in [0, 1]
        affinity_matrix = (correlation_array + 1) / 2

        # Ensure the matrix is symmetric and has a proper diagonal
        np.fill_diagonal(affinity_matrix, 1.0)

        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
                n_init=10,
                eigen_solver="arpack",
            )

            cluster_labels = spectral.fit_predict(affinity_matrix)
            info(f"Spectral clustering completed with {len(set(cluster_labels))} unique labels")

        except Exception as e:
            error(f"Error during spectral clustering fit_predict: {str(e)}")
            return {}

        return self._organize_champions_by_cluster(cluster_labels, champions, "spectral")

