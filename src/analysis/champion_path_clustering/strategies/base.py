"""Base protocol and class for clustering strategies."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from src.utils.logger import error, info
from src.utils.matrix_operations import create_distance_matrix


class ClusteringStrategy(Protocol):
    """Protocol defining the interface for clustering strategies."""

    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        ...

    def create_distance_matrix(
        self, correlation_matrix: dict[str, dict[str, float]], champions: list[str]
    ) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            champions: List of champion names to include in the matrix

        Returns:
            NumPy array containing the distance matrix
        """
        ...


class BaseClusteringStrategy(ABC):
    """Base class for clustering strategies with common functionality."""

    def create_distance_matrix(
        self, correlation_matrix: dict[str, dict[str, float]], champions: list[str]
    ) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix efficiently.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            champions: List of champion names to include in the matrix

        Returns:
            NumPy array containing the distance matrix
        """
        return create_distance_matrix(champions, correlation_matrix)

    def _organize_champions_by_cluster(
        self,
        cluster_labels: np.ndarray,
        champions: list[str],
        method_name: str = "clustering",
    ) -> dict[int, list[str]]:
        """
        Organize champions by cluster based on cluster labels.

        Args:
            cluster_labels: Array of cluster labels from clustering algorithm
            champions: List of champion names in the same order as cluster_labels
            method_name: Name of the clustering method for logging purposes

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        clusters = {}
        try:
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(champions[i])

            info(f"{method_name.capitalize()} grouped champions into {len(clusters)} clusters")

            return clusters

        except Exception as e:
            error(f"Error organizing champions by {method_name} cluster: {str(e)}")
            return {}

    @abstractmethod
    def cluster(
        self, correlation_matrix: dict[str, dict[str, float]], n_clusters: int
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        pass
