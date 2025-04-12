"""
Threshold-based clustering strategy for champion path clustering.

This strategy ensures that champions in the same cluster have at least
a minimum correlation with each other.
"""

from collections import defaultdict

from src.utils.logger import info

from .base import BaseClusteringStrategy


class ThresholdClusteringStrategy(BaseClusteringStrategy): 
    """Strategy for threshold-based clustering of champions."""

    def cluster(
        self,
        correlation_matrix: dict[str, dict[str, float]],
        n_clusters: int,
        correlation_threshold: float = 0.7,
    ) -> dict[int, list[str]]:
        """
        Cluster champions based on their graph similarities using a threshold-based approach.

        This method ensures that all champions in a cluster have at least the specified
        minimum correlation with all other champions in the same cluster.

        Args:
            correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
            n_clusters: Target number of clusters (used as a reference, not guaranteed)
            correlation_threshold: Minimum correlation required between champions in the same cluster

        Returns:
            Dictionary mapping cluster IDs to lists of champions
        """
        champions = list(correlation_matrix.keys())
        info(
            f"Threshold clustering {len(champions)} champions with correlation_threshold={correlation_threshold}"
        )

        # Create a graph where edges represent correlations above the threshold
        graph = defaultdict(set)
        for champ1 in champions:
            for champ2 in champions:
                if (
                    champ1 != champ2
                    and correlation_matrix[champ1][champ2] >= correlation_threshold
                ):
                    graph[champ1].add(champ2)

        clusters = []
        remaining = set(champions)

        while remaining:
            # If no more connections above threshold, each remaining champion gets its own cluster
            if not graph:
                clusters.extend([[champ] for champ in remaining])
                break

            # Start with champion having most connections
            start = (
                max(graph.items(), key=lambda x: len(x[1]))[0]
                if graph
                else next(iter(remaining))
            )
            cluster = {start}
            candidates = graph.get(start, set()).copy()

            # Grow cluster while possible
            while candidates:
                # Find best candidate (highest avg correlation with current cluster)
                best = None

                # Pre-compute valid candidates to avoid repeated checks
                valid_candidates = {
                    candidate
                    for candidate in candidates
                    if all(
                        correlation_matrix[candidate][member] >= correlation_threshold
                        for member in cluster
                    )
                }

                if valid_candidates:
                    # Find candidate with highest average correlation
                    best = max(
                        valid_candidates,
                        key=lambda candidate: sum(
                            correlation_matrix[candidate][member] for member in cluster
                        ) / len(cluster),
                    )

                    cluster.add(best)
                    candidates.remove(best)
                    candidates.update(graph.get(best, set()) - cluster)
                else:
                    break

            clusters.append(list(cluster))
            remaining -= cluster

            # Optimize graph update by removing processed champions in batch
            for champ in cluster:
                if champ in graph:
                    del graph[champ]
            # Update connections in graph to remove references to champions in the cluster
            for graph_key in graph:
                graph[graph_key] -= cluster

        result = dict(enumerate(clusters))

        info(f"Threshold clustering completed with {len(result)} clusters")
        for cluster_id, champions_list in result.items():
            info(f"Threshold cluster {cluster_id}: {len(champions_list)} champions")

        return result
