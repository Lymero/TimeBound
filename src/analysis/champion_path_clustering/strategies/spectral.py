"""Spectral clustering strategy for champion path clustering."""

import numpy as np
from sklearn.cluster import SpectralClustering

from src.utils.logger import error, info

from .base import BaseClusteringStrategy


class SpectralClusteringStrategy(BaseClusteringStrategy):
    """Strategy for spectral clustering of champions."""

    def create_connectivity_matrix(
        self, correlation_matrix: dict[str, dict[str, float]], champions: list[str], correlation_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Create a connectivity matrix based on correlation threshold.
        
        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            champions: List of champion names to include in the matrix
            correlation_threshold: Minimum correlation required for champions to be connected
            
        Returns:
            NumPy array containing the connectivity matrix
        """
        n = len(champions)
        connectivity = np.zeros((n, n))
        
        for i, champ1 in enumerate(champions):
            connectivity[i, i] = 1
            
            for j, champ2 in enumerate(champions):
                if i != j and correlation_matrix[champ1][champ2] >= correlation_threshold:
                    connectivity[i, j] = 1
        
        info(f"Created connectivity matrix with threshold {correlation_threshold}, "
             f"connectivity density: {connectivity.sum() / (n * n):.2%}")
        
        return connectivity

    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int, correlation_threshold: float = 0.7
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using spectral clustering.

        This method provides an alternative to the hierarchical clustering approach,
        using spectral clustering which can better capture non-linear relationships
        and complex cluster shapes.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create
            correlation_threshold: Minimum correlation for champions to be considered connected

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        champions = list(correlation_matrix.keys())
        info(f"Performing spectral clustering on {len(champions)} champions with n_clusters={n_clusters}, "
             f"correlation_threshold={correlation_threshold}")
             
        connectivity_matrix = self.create_connectivity_matrix(
            correlation_matrix, champions, correlation_threshold
        )

        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
                n_init=10,
                eigen_solver="arpack",
            )

            cluster_labels = spectral.fit_predict(connectivity_matrix)
            info(f"Spectral clustering completed with {len(set(cluster_labels))} unique labels")

        except Exception as e:
            error(f"Error during spectral clustering fit_predict: {str(e)}")
            return {}

        return self._organize_champions_by_cluster(cluster_labels, champions, "spectral")

