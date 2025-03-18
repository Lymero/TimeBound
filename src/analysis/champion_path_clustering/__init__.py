"""Champion path clustering package for analyzing champion relationships."""

__version__ = "0.1.0"

from .clusterer import ChampionPathClusterer
from .factory import ClusteringStrategyFactory, StrategyType
from .strategies.base import BaseClusteringStrategy, ClusteringStrategy
from .strategies.hierarchical import HierarchicalClusteringStrategy
from .strategies.spectral import SpectralClusteringStrategy
from .types import ClusterStats

__all__ = [
    "BaseClusteringStrategy",
    "ChampionPathClusterer",
    "ClusterStats",
    "ClusteringStrategy",
    "ClusteringStrategyFactory",
    "HierarchicalClusteringStrategy",
    "SpectralClusteringStrategy",
    "StrategyType",
    "__version__",
]
