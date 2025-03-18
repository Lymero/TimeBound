"""
Graph correlation utilities for comparing win rate curves.

This module provides functions to calculate and visualize the similarity
between different champion win rate curves, focusing on trend correlation
rather than exact matches.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.utils.logger import error, warning

EPSILON = 1e-10  # Small value to avoid division by zero and for float comparisons
PERFECT_CORR_THRESHOLD = 0.99  # Threshold for considering correlation as perfect
FULL_OVERLAP_THRESHOLD = 0.99  # Threshold for considering full x-range overlap
SIGNIFICANT_OVERLAP_THRESHOLD = 0.9  # Threshold for applying overlap penalty


def compare(
    points1: list[tuple[float, float]], points2: list[tuple[float, float]]
) -> float:
    """
    Compare two sets of points as graphs, focusing on correlation of trends.
    
    This method is:
    - Invariant to vertical shifts (y-axis shifts)
    - Sensitive to horizontal shifts (x-axis shifts)
    - Invariant to vertical scaling (y-axis scaling)

    For example:
    - [(0,0), (1,1)] and [(0,1), (1,2)] would be 100% similar (vertical shift)
    - [(0,0), (1,1)] and [(0,0), (1,2)] would be 100% similar (vertical scaling)
    - [(0,0), (1,1)] and [(1,0), (2,1)] would NOT be 100% similar (horizontal shift)

    Args:
        points1: First set of (x,y) points
        points2: Second set of (x,y) points

    Returns:
        Similarity score between -1 and 1, where:
        - 1 means perfectly positively correlated (similar scaling patterns)
        - -1 means perfectly negatively correlated (opposite scaling patterns)
        - 0 means no correlation
    """
    if not points1 or not points2:
        error("One or both point sets are empty")
        return 0.0

    # Sort both point sets by x-coordinate
    points1_sorted = sorted(points1, key=lambda p: p[0])
    points2_sorted = sorted(points2, key=lambda p: p[0])

    # Extract x and y values
    x1, y1 = (
        np.array([p[0] for p in points1_sorted]),
        np.array([p[1] for p in points1_sorted]),
    )
    x2, y2 = (
        np.array([p[0] for p in points2_sorted]),
        np.array([p[1] for p in points2_sorted]),
    )

    # Check if x ranges are compatible
    x1_min, x1_max = min(x1), max(x1)
    x2_min, x2_max = min(x2), max(x2)

    # Calculate overlap percentage of x ranges
    overlap_min = max(x1_min, x2_min)
    overlap_max = min(x1_max, x2_max)

    if overlap_max <= overlap_min:
        # No overlap in x ranges
        return 0.0

    overlap_percentage = (overlap_max - overlap_min) / max(
        x1_max - x1_min, x2_max - x2_min
    )

    # If x values are identical, we can do a direct comparison
    if np.array_equal(x1, x2):
        f1 = interp1d(x1, y1, bounds_error=False, fill_value="extrapolate")
        f2 = interp1d(x2, y2, bounds_error=False, fill_value="extrapolate")

        # Get y values at common x points
        y1_common = y1
        y2_common = y2
    else:
        f1 = interp1d(x1, y1, bounds_error=False, fill_value="extrapolate")
        f2 = interp1d(x2, y2, bounds_error=False, fill_value="extrapolate")

        # Create common x points in the overlapping range
        x_common = np.linspace(overlap_min, overlap_max, 100)

        # Get y values at common x points
        y1_common = f1(x_common)
        y2_common = f2(x_common)

    # Remove vertical shift by subtracting mean
    y1_centered = y1_common - np.mean(y1_common)
    y2_centered = y2_common - np.mean(y2_common)

    # Remove vertical scaling by dividing by standard deviation
    # (avoid division by zero)
    y1_std = max(np.std(y1_centered), EPSILON)
    y2_std = max(np.std(y2_centered), EPSILON)

    y1_normalized = y1_centered / y1_std
    y2_normalized = y2_centered / y2_std

    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(y1_normalized, y2_normalized)[0, 1]

    # Handle NaN (can happen with constant functions)
    if np.isnan(correlation):
        # Check if both are constant functions (flat lines)
        if np.std(y1_common) < EPSILON and np.std(y2_common) < EPSILON:
            return 1.0  # Two flat lines are perfectly correlated
        return 0.0

    # For perfect or near-perfect correlation, return 1.0 or -1.0
    if abs(correlation) > PERFECT_CORR_THRESHOLD and overlap_percentage > FULL_OVERLAP_THRESHOLD:
        return 1.0 if correlation > 0 else -1.0

    # Apply penalty for partial x-range overlap
    if overlap_percentage < SIGNIFICANT_OVERLAP_THRESHOLD:
        correlation *= overlap_percentage

    return correlation


def plot_comparison(
    points1: list[tuple[float, float]],
    points2: list[tuple[float, float]],
    title: str = "Graph Correlation Comparison",
) -> None:
    """
    Plot two sets of points with their correlation score.

    Args:
        points1: First set of (x,y) points
        points2: Second set of (x,y) points
        title: Plot title
    """
    if not points1 or not points2:
        warning("One or both point sets are empty")
        return

    correlation = compare(points1, points2)
    x1, y1 = zip(*points1, strict=True)
    x2, y2 = zip(*points2, strict=True)

    plt.figure(figsize=(12, 8))
    plt.plot(x1, y1, "b.-", label="Set 1")
    plt.plot(x2, y2, "r.-", label="Set 2")

    plt.grid(True)
    plt.title(f"{title} (Correlation: {correlation:.4f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()


def main() -> None:
    """Main entry point for the graph correlation module."""
    pass
