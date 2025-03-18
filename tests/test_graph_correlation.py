"""Tests for the Graph Correlation module using pytest."""

import numpy as np
import pytest

from src.analysis.graph_correlation import compare


# Test fixtures
@pytest.fixture
def linear_points() -> list[tuple[float, float]]:
    """Simple linear points [(0,0), (1,1), ..., (9,9)]."""
    return [(i, i) for i in range(10)]


@pytest.fixture
def vertical_shift(linear_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Linear points with vertical shift [(0,5), (1,6), ..., (9,14)]."""
    return [(x, y + 5) for x, y in linear_points]


@pytest.fixture
def vertical_scale(linear_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Linear points with vertical scaling [(0,0), (1,2), ..., (9,18)]."""
    return [(x, y * 2) for x, y in linear_points]


@pytest.fixture
def horizontal_shift(linear_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Linear points with horizontal shift [(5,0), (6,1), ..., (14,9)]."""
    return [(x + 5, y) for x, y in linear_points]


@pytest.fixture
def combined_vertical(linear_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Linear points with combined vertical shift and scaling [(0,2), (1,5), ..., (9,29)]."""
    return [(x, y * 3 + 2) for x, y in linear_points]


@pytest.fixture(scope="module")
def trig_x_values() -> np.ndarray:
    """X values for trigonometric functions."""
    return np.linspace(0, 2 * np.pi, 100)


@pytest.fixture
def sine_points(trig_x_values: np.ndarray) -> list[tuple[float, float]]:
    """Sine wave points [(x, sin(x)), ...]."""
    return [(float(x_val), float(np.sin(x_val))) for x_val in trig_x_values]


@pytest.fixture
def cosine_points(trig_x_values: np.ndarray) -> list[tuple[float, float]]:
    """Cosine wave points [(x, cos(x)), ...]."""
    return [(float(x_val), float(np.cos(x_val))) for x_val in trig_x_values]


@pytest.fixture
def noisy_sine(trig_x_values: np.ndarray) -> list[tuple[float, float]]:
    """Sine wave with added noise."""
    np.random.seed(42)  # For reproducibility
    return [
        (float(x_val), float(np.sin(x_val) + 0.2 * np.random.randn()))
        for x_val in trig_x_values
    ]


# Vertical transformation tests
@pytest.mark.parametrize(
    "transform_fixture,expected_correlation",
    [
        ("vertical_shift", pytest.approx(1.0, abs=1e-4)),
        ("vertical_scale", pytest.approx(1.0, abs=1e-4)),
        ("combined_vertical", pytest.approx(1.0, abs=1e-4)),
    ],
)
def test_vertical_transformations(
    request: pytest.FixtureRequest,
    linear_points: list[tuple[float, float]],
    transform_fixture: str,
    expected_correlation: float,
) -> None:
    """Test that vertical transformations don't affect correlation."""
    transformed_points = request.getfixturevalue(transform_fixture)
    correlation = compare(linear_points, transformed_points)
    assert correlation == expected_correlation


# Horizontal transformation tests
def test_horizontal_shift_correlation(
    linear_points: list[tuple[float, float]], horizontal_shift: list[tuple[float, float]]
) -> None:
    """Horizontal shifts should reduce correlation (unlike vertical shifts)."""
    correlation = compare(linear_points, horizontal_shift)
    assert correlation < 1.0


def test_non_overlapping_x_ranges(linear_points: list[tuple[float, float]]) -> None:
    """Non-overlapping x ranges should have low correlation."""
    far_shifted_points = [(x + 20, y) for x, y in linear_points]
    correlation = compare(linear_points, far_shifted_points)
    assert correlation < 0.5


# Different pattern tests
def test_different_trends_correlation(
    sine_points: list[tuple[float, float]], cosine_points: list[tuple[float, float]]
) -> None:
    """Different trends (sine vs cosine) should have low correlation."""
    correlation = compare(sine_points, cosine_points)
    assert correlation < 0.5


def test_noisy_correlation(
    sine_points: list[tuple[float, float]], noisy_sine: list[tuple[float, float]]
) -> None:
    """Adding noise should reduce but not destroy correlation with original."""
    correlation = compare(sine_points, noisy_sine)
    assert correlation > 0.8


# Identity and edge cases
def test_identical_points_correlation(linear_points: list[tuple[float, float]]) -> None:
    """Identical point sets should have perfect correlation."""
    correlation = compare(linear_points, linear_points)
    assert correlation == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "points1,points2,expected",
    [
        ([], [], 0.0),
        ("linear_points", [], 0.0),
        ([], "linear_points", 0.0),
    ],
)
def test_empty_points(
    request: pytest.FixtureRequest, 
    points1: object, 
    points2: object, 
    expected: float
) -> None:
    """Empty point sets should return zero correlation."""
    p1 = request.getfixturevalue(points1) if isinstance(points1, str) else points1
    p2 = request.getfixturevalue(points2) if isinstance(points2, str) else points2

    correlation = compare(p1, p2)
    assert correlation == expected
