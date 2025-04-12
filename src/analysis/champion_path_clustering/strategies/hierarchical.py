"""Hierarchical clustering strategy for champion path clustering."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.utils.logger import error, info

from .base import BaseClusteringStrategy


class HierarchicalClusteringStrategy(BaseClusteringStrategy):
    """Strategy for hierarchical clustering of champions."""

    def create_thresholded_distance_matrix(
        self, correlation_matrix: dict[str, dict[str, float]], champions: list[str], correlation_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Create a distance matrix that applies the correlation threshold.
        
        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            champions: List of champion names to include in the matrix
            correlation_threshold: Minimum correlation required for champions to be considered similar
            
        Returns:
            NumPy array containing the thresholded distance matrix
        """
        distance_matrix = self.create_distance_matrix(correlation_matrix, champions)
        distance_threshold = 1 - correlation_threshold
        mask = distance_matrix > distance_threshold
        thresholded_matrix = distance_matrix.copy()
        # weight_multiplier = 10 
        thresholded_matrix[mask] = 1
        
        np.fill_diagonal(thresholded_matrix, 0)
        
        original_avg = distance_matrix.mean()
        thresholded_avg = thresholded_matrix.mean()
        count_affected = mask.sum()
        
        info(f"Applied correlation threshold {correlation_threshold} to distance matrix: "
             f"{count_affected} connections ({count_affected / mask.size:.2%}) were affected, "
             f"average distance changed from {original_avg:.4f} to {thresholded_avg:.4f}")
        
        return thresholded_matrix

    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int, correlation_threshold: float = 0.7
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using hierarchical clustering.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create
            correlation_threshold: Minimum correlation required for champions to be considered similar

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        try:
            champions = list(correlation_matrix.keys())
            info(f"Hierarchical clustering {len(champions)} champions with n_clusters={n_clusters}, "
                 f"correlation_threshold={correlation_threshold}")

            distance_matrix = self.create_thresholded_distance_matrix(
                correlation_matrix, champions, correlation_threshold
            )
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric="precomputed", 
                linkage="average"
            )

            try:
                cluster_labels = clustering.fit_predict(distance_matrix)
                info(f"Hierarchical clustering completed with {len(set(cluster_labels))} unique labels")
            except Exception as e:
                error(f"Error during hierarchical clustering fit_predict: {str(e)}")
                return {}

            return self._organize_champions_by_cluster(cluster_labels, champions, "hierarchical")

        except Exception as e:
            error(f"Unhandled exception in hierarchical clustering: {str(e)}")
            return {}
