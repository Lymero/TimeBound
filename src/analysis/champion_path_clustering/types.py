"""Type definitions for champion path clustering."""

from typing import TypedDict


class ClusterStats(TypedDict):
    """Type definition for cluster statistics."""

    cluster_count: int
    champion_count: int
    avg_cluster_size: float
    min_cluster_size: int
    max_cluster_size: int
    cluster_sizes: dict[int, int]
