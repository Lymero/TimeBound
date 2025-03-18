"""
Matrix Operation Utilities.

This module provides utility functions for matrix operations used across the application.
"""

import numpy as np


def create_distance_matrix(
    champions: list[str], 
    correlation_matrix: dict[str, dict[str, float]]
) -> np.ndarray:
    """
    Create a distance matrix from a correlation matrix.
    
    Args:
        champions: List of champion names
        correlation_matrix: Dictionary mapping champion names to dictionaries of correlations
        
    Returns:
        NumPy array containing the distance matrix (1 - correlation)
    """
    return 1 - np.array(
        [
            [correlation_matrix.get(c1, {}).get(c2, 0) for c2 in champions]
            for c1 in champions
        ]
    ) 