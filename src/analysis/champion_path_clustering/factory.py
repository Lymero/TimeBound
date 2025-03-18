"""
Factory for creating clustering strategies.

This module provides a factory class for creating different clustering strategy.
"""

from enum import Enum

from .strategies.base import ClusteringStrategy
from .strategies.hierarchical import HierarchicalClusteringStrategy
from .strategies.spectral import SpectralClusteringStrategy
from .strategies.threshold import ThresholdClusteringStrategy


class StrategyType(str, Enum):
    """Enum of available clustering strategy types."""

    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    THRESHOLD = "threshold"

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of all strategy type values."""
        return [e.value for e in cls]


class ClusteringStrategyFactory:
    """Factory for creating clustering strategy instances."""

    _strategies: dict[str, type[ClusteringStrategy]] = {
        StrategyType.HIERARCHICAL.value: HierarchicalClusteringStrategy,
        StrategyType.SPECTRAL.value: SpectralClusteringStrategy,
        StrategyType.THRESHOLD.value: ThresholdClusteringStrategy,
    }

    @classmethod
    def create_strategy(cls, strategy_type: str) -> ClusteringStrategy:
        """
        Create a clustering strategy instance based on the strategy type.

        Args:
            strategy_type: The type of clustering strategy to create
                           (e.g., "hierarchical", "spectral", "threshold")

        Returns:
            An instance of the requested clustering strategy

        Raises:
            ValueError: If the strategy type is not recognized
        """
        if strategy_type not in cls._strategies:
            valid_strategies = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Valid strategies are: {valid_strategies}"
            )

        strategy_class = cls._strategies[strategy_type]
        return strategy_class()

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get a list of available strategy types.

        Returns:
            List of available strategy type names
        """
        return list(cls._strategies.keys())
