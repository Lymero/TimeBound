"""
Champion Path Clusterer.

This module provides the main class for clustering champions
based on their win rate vs game length graphs.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.analysis.graph_correlation import compare
from src.processing.svg_parser import SVGPathParser
from src.utils.logger import error, info, warning

from .clustering_types import ClusterStats
from .factory import ClusteringStrategyFactory
from .strategies.base import ClusteringStrategy

DEFAULT_SVG_PATHS_FILE = "data/champion_svg_paths.json"
DEFAULT_CORRELATION_THRESHOLD = 0.8
DEFAULT_CLUSTER_RATIO = 0.1


class ChampionPathClusterer:
    """A class to cluster champions based on the similarity of their win rate vs game length graphs.

    This class extracts points from champion SVG paths, computes a correlation matrix
    between all champions, and clusters them based on their graph similarities.
    """

    def __init__(
        self,
        svg_paths_file: str = DEFAULT_SVG_PATHS_FILE,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        limit: int | None = None,
        clustering_strategy: ClusteringStrategy | None = None,
    ) -> None:
        """
        Initialize the ChampionPathClusterer.

        Args:
            svg_paths_file: Path to the JSON file containing champion SVG paths
            correlation_threshold: Threshold for considering two champions as similar
            limit: Maximum number of champions to process (None for all)
            clustering_strategy: Strategy to use for clustering
        """
        self.svg_paths_file = Path(svg_paths_file)
        self.champion_points: dict[str, list[tuple[float, float]]] = {}
        self.correlation_matrix: dict[str, dict[str, float]] = {}
        self.clusters: dict[int, list[str]] = {}
        self.correlation_threshold = correlation_threshold
        self.limit = limit
        self.clustering_strategy = (
            clustering_strategy
            or ClusteringStrategyFactory.create_strategy("hierarchical")
        )

    def __repr__(self) -> str:
        """Return a string representation of the clusterer."""
        return (
            f"ChampionPathClusterer(champions={len(self.champion_points)}, "
            f"clusters={len(self.clusters)}, "
            f"strategy={self.clustering_strategy.__class__.__name__})"
        )

    def _load_svg_paths(self) -> dict[str, Any]:
        """
        Load champion SVG paths from the JSON file.

        Returns:
            Dictionary containing champion SVG path data

        Raises:
            FileNotFoundError: If the SVG paths file does not exist
            json.JSONDecodeError: If the SVG paths file is not valid JSON
        """
        try:
            with open(self.svg_paths_file, encoding="utf-8") as f:
                svg_paths = json.load(f)
            info(f"Loaded SVG paths for {len(svg_paths)} champions")
            return svg_paths
        except Exception as e:
            error(f"Error loading SVG paths: {e}")
            return {}

    def _extract_points(self) -> None:
        """
        Extract points from champion SVG paths.

        This method loads the SVG paths, parses them using SVGPathParser,
        and stores the resulting points for each champion.
        """
        svg_paths = self._load_svg_paths()

        if self.limit is not None:
            svg_paths = dict(list(svg_paths.items())[: self.limit])

        for champion, data in svg_paths.items():
            svg_path = data.get("svg_path")

            if not svg_path:
                warning(f"No SVG path found for {champion}")
                continue

            try:
                parser = SVGPathParser(svg_path)
                points = parser.parse_path()

                if points:
                    self.champion_points[champion] = points
                else:
                    warning(f"No points extracted for {champion}")
            except Exception as e:
                error(f"Error extracting points for {champion}: {e}")

        info(f"Extracted points for {len(self.champion_points)} champions")

    def _compute_correlation_matrix(self) -> None:
        """
        Compute the correlation matrix between all champions.

        This method computes the similarity score between each pair of champions.
        """
        if not self.champion_points:
            self._extract_points()

        champions = list(self.champion_points.keys())
        self.correlation_matrix = defaultdict(dict)

        # Set all diagonal values to 1.0 (champion compared to itself)
        for champion in champions:
            self.correlation_matrix[champion][champion] = 1.0

        pairs = [
            (champ1, champ2)
            for i, champ1 in enumerate(champions)
            for champ2 in champions[i + 1 :]
        ]

        for champ1, champ2 in tqdm(pairs, desc="Computing correlations"):
            correlation = compare(
                self.champion_points[champ1], self.champion_points[champ2]
            )

            self.correlation_matrix[champ1][champ2] = correlation
            self.correlation_matrix[champ2][champ1] = correlation

        info("Computed correlation matrix")

    def _determine_optimal_cluster_count(
        self, n_clusters: int, champions: list[str]
    ) -> int:
        """
        Determine the appropriate number of clusters based on input and data size.

        Args:
            n_clusters: Requested number of clusters (0 or None for automatic determination)
            champions: List of champion names to be clustered

        Returns:
            Appropriate number of clusters to use
        """
        # Determine number of clusters if not specified
        if n_clusters is None or n_clusters <= 0:
            n_clusters = max(3, int(len(champions) * DEFAULT_CLUSTER_RATIO))
            info(f"Using automatic n_clusters={n_clusters}")

        # Check if n_clusters is valid
        if n_clusters > len(champions):
            warning(
                f"n_clusters ({n_clusters}) is greater than the number of champions "
                f"({len(champions)}). Setting n_clusters to {len(champions)}"
            )
            n_clusters = len(champions)

        return n_clusters

    def cluster_champions(self, n_clusters: int = 0, evaluate: bool = False) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using the current strategy.

        Args:
            n_clusters: Number of clusters to create (0 or None for automatic determination)
            evaluate: Whether to compute and log the Silhouette Score

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        try:
            if not self.correlation_matrix:
                self._compute_correlation_matrix()

            champions = list(self.correlation_matrix.keys())
            n_clusters = self._determine_optimal_cluster_count(n_clusters, champions)
            
            self.clusters = self.clustering_strategy.cluster(
                self.correlation_matrix,
                n_clusters,
                correlation_threshold=self.correlation_threshold,
            )

            if evaluate:
                silhouette_avg = self.compute_silhouette_score()
                if silhouette_avg is not None:
                    info(f"Clustering Silhouette Score: {silhouette_avg:.4f}")

            return self.clusters

        except Exception as e:
            error(f"Unhandled exception in cluster_champions: {e}")
            return {}

    def compute_silhouette_score(self) -> float | None:
        """
        Compute the average Silhouette Score for the current clustering.

        Returns:
            The average Silhouette Score
        """
        champions = list(self.correlation_matrix.keys())
        distance_matrix = self.clustering_strategy.create_distance_matrix(
            self.correlation_matrix, champions
        )

        champion_to_id = {name: i for i, name in enumerate(champions)}
        labels = [0] * len(champions)

        for cluster_id, champions_in_cluster in self.clusters.items():
            for champion in champions_in_cluster:
                labels[champion_to_id[champion]] = cluster_id

        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            return score
        except ValueError as e:
             error(f"Could not compute Silhouette Score: {e}")
             return None
        except Exception as e:
             error(f"Unexpected error computing Silhouette Score: {e}")
             return None

    def get_similar_champions(
        self, champion: str, threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """
        Get champions similar to the specified champion.

        Args:
            champion: The champion to find similar champions for
            threshold: Similarity threshold (defaults to self.correlation_threshold)

        Returns:
            List of (champion, similarity) tuples sorted by similarity (highest first).
            Returns an empty list if the champion is not found or no similar champions exist.
        """
        if not self.correlation_matrix:
            self._compute_correlation_matrix()

        if champion not in self.correlation_matrix:
            warning(f"Champion {champion} not found in correlation matrix")
            return []

        threshold = threshold or self.correlation_threshold

        similar_champions = []
        for other_champion, similarity in self.correlation_matrix[champion].items():
            if other_champion != champion and similarity >= threshold:
                similar_champions.append((other_champion, similarity))

        return sorted(similar_champions, key=lambda x: x[1], reverse=True)

    def get_cluster_stats(
        self, clusters: dict[int, list[str]] | None = None
    ) -> ClusterStats:
        """
        Get basic statistics about the clusters.

        Args:
            clusters: Clusters to analyze (uses self.clusters if None)

        Returns:
            Dictionary with cluster statistics:
            - cluster_count: Number of clusters
            - champion_count: Total number of champions
            - avg_cluster_size: Average number of champions per cluster
            - min_cluster_size: Size of the smallest cluster
            - max_cluster_size: Size of the largest cluster
            - cluster_sizes: Dictionary mapping cluster IDs to their sizes
        """
        clusters = clusters or self.clusters

        if not clusters:
            return {
                "cluster_count": 0,
                "champion_count": 0,
                "avg_cluster_size": 0,
                "min_cluster_size": 0,
                "max_cluster_size": 0,
                "cluster_sizes": {},
            }

        cluster_sizes = {
            cluster_id: len(champions) for cluster_id, champions in clusters.items()
        }
        champion_count = sum(cluster_sizes.values())

        return {
            "cluster_count": len(clusters),
            "champion_count": champion_count,
            "avg_cluster_size": champion_count / len(clusters) if clusters else 0,
            "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "cluster_sizes": cluster_sizes,
        }
