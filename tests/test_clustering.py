"""Tests for champion path clustering functionality."""

from typing import Any

import numpy as np
import pytest

from src.analysis.champion_path_clustering.strategies.base import BaseClusteringStrategy
from src.analysis.champion_path_clustering.strategies.hierarchical import HierarchicalClusteringStrategy
from src.analysis.champion_path_clustering.strategies.spectral import SpectralClusteringStrategy


@pytest.fixture
def correlation_matrix() -> dict[str, dict[str, float]]:
    """Sample correlation matrix for testing."""
    champions = ["champ1", "champ2", "champ3", "champ4", "champ5", "champ6"]
    matrix = {}
    
    # Create a correlation matrix with clear clusters
    # Champions 1-3 and 4-6 form distinct clusters
    for i, champ1 in enumerate(champions):
        matrix[champ1] = {}
        for j, champ2 in enumerate(champions):
            if i == j:
                matrix[champ1][champ2] = 1.0
            # Champions in the same cluster have high correlation
            elif (i < 3 and j < 3) or (i >= 3 and j >= 3):
                matrix[champ1][champ2] = 0.9
            # Champions in different clusters have low correlation
            else:
                matrix[champ1][champ2] = 0.2
    
    return matrix


@pytest.fixture
def sample_svg_paths() -> dict[str, dict[str, Any]]:
    """Sample SVG paths for testing."""
    return {
        "champ1": {"svg_path": "M10,20 L30,40 L50,60"},
        "champ2": {"svg_path": "M10,20 L30,40 L50,60"},
        "champ3": {"svg_path": "M15,25 L35,45 L55,65"},
        "champ4": {"svg_path": "M50,60 L70,80 L90,100"},
        "champ5": {"svg_path": "M50,60 L70,80 L90,100"}
    }


class MockClusteringStrategy(BaseClusteringStrategy):
    """Mock implementation of BaseClusteringStrategy for testing."""
    
    def cluster(
        self, 
        correlation_matrix: dict[str, dict[str, float]], 
        n_clusters: int
    ) -> dict[int, list[str]]:
        """Implement abstract method."""
        champions = list(correlation_matrix.keys())
        return {i: [champ] for i, champ in enumerate(champions)}


def test_base_strategy_distance_matrix(correlation_matrix: dict[str, dict[str, float]]) -> None:
    """Test that the base strategy correctly creates a distance matrix."""
    strategy = MockClusteringStrategy()
    champions = list(correlation_matrix.keys())
    
    distance_matrix = strategy.create_distance_matrix(correlation_matrix, champions)
    
    # Check that the distance matrix is correctly created
    assert isinstance(distance_matrix, np.ndarray)
    assert distance_matrix.shape == (len(champions), len(champions))
    
    # Check diagonal values (distance to self should be 0)
    for i in range(len(champions)):
        assert distance_matrix[i, i] == 0.0
    
    # Check that distances are 1 - correlation
    for i, champ1 in enumerate(champions):
        for j, champ2 in enumerate(champions):
            assert distance_matrix[i, j] == pytest.approx(1.0 - correlation_matrix[champ1][champ2])


def test_organize_champions_by_cluster() -> None:
    """Test that champions are correctly organized by cluster."""
    strategy = MockClusteringStrategy()
    champions = ["champ1", "champ2", "champ3", "champ4", "champ5", "champ6"]
    
    # Create cluster labels (0, 0, 0, 1, 1, 1)
    cluster_labels = np.array([0, 0, 0, 1, 1, 1])
    
    clusters = strategy._organize_champions_by_cluster(cluster_labels, champions)
    
    # Check that clusters are correctly organized
    assert len(clusters) == 2
    assert set(clusters[0]) == {"champ1", "champ2", "champ3"}
    assert set(clusters[1]) == {"champ4", "champ5", "champ6"}


def test_hierarchical_clustering(correlation_matrix: dict[str, dict[str, float]]) -> None:
    """Test that hierarchical clustering produces expected results."""
    strategy = HierarchicalClusteringStrategy()
    n_clusters = 2
    
    clusters = strategy.cluster(correlation_matrix, n_clusters)
    
    # Check that we have the expected number of clusters
    assert len(clusters) == n_clusters
    
    # Check that all champions are assigned to a cluster
    all_champions = set(correlation_matrix.keys())
    clustered_champions = set()
    for cluster in clusters.values():
        clustered_champions.update(cluster)
    
    assert clustered_champions == all_champions


def test_spectral_clustering(correlation_matrix: dict[str, dict[str, float]]) -> None:
    """Test that spectral clustering produces expected results."""
    strategy = SpectralClusteringStrategy()
    n_clusters = 2
    
    clusters = strategy.cluster(correlation_matrix, n_clusters)
    
    # Check that we have the expected number of clusters
    assert len(clusters) == n_clusters
    
    # Check that all champions are assigned to a cluster
    all_champions = set(correlation_matrix.keys())
    clustered_champions = set()
    for cluster in clusters.values():
        clustered_champions.update(cluster)
    
    assert clustered_champions == all_champions