"""Hierarchical clustering strategy for champion path clustering."""

from sklearn.cluster import AgglomerativeClustering

from src.utils.logger import error, info

from .base import BaseClusteringStrategy


class HierarchicalClusteringStrategy(BaseClusteringStrategy):
    """Strategy for hierarchical clustering of champions."""

    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using hierarchical clustering.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        try:
            champions = list(correlation_matrix.keys())
            info(f"Hierarchical clustering {len(champions)} champions with n_clusters={n_clusters}")

            distance_matrix = self.create_distance_matrix(correlation_matrix, champions)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")

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
