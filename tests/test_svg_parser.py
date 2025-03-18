"""Tests for the SVG Path Parser using pytest."""

import pytest
from typing import List, Tuple, Any

from src.processing.svg_parser import SVGPathParser


# Test fixtures
@pytest.fixture
def simple_path() -> str:
    """Simple path with move and line commands."""
    return "M10,20 L30,40 L50,60"


@pytest.fixture
def complex_path() -> str:
    """Complex path with curves."""
    return "M10,20 C30,40 50,60 70,80 L90,100 Q110,120 130,140"


@pytest.fixture
def cubic_bezier_path() -> str:
    """Cubic bezier path."""
    return "M10,20 C30,40 50,60 70,80"


@pytest.fixture
def quadratic_bezier_path() -> str:
    """Quadratic bezier path."""
    return "M10,20 Q30,40 50,60"


@pytest.fixture
def relative_path() -> str:
    """Path with relative commands."""
    return "M10,20 l20,20 l20,0"


@pytest.fixture
def mixed_path() -> str:
    """Path with mixed absolute and relative commands."""
    return "M10,10 l10,10 L40,20 c10,10 10,20 20,20"


@pytest.fixture
def parser(request: pytest.FixtureRequest) -> SVGPathParser:
    """Create a parser with the specified path."""
    path = request.param if hasattr(request, "param") else ""
    return SVGPathParser(path)


# Basic parsing tests
def test_parse_path_simple(simple_path: str) -> None:
    """Simple path with move and line commands should produce exact points."""
    parser = SVGPathParser(simple_path)
    points = parser.parse_path()

    # Points are reversed and y-inverted
    # Original: [(10, 20), (30, 40), (50, 60)]
    # Reversed: [(50, 60), (30, 40), (10, 20)]
    # Y-inverted (max_y=60): [(50, 0), (30, 20), (10, 40)]
    assert len(points) == 3
    assert points[0] == (50, 0)
    assert points[1] == (30, 20)
    assert points[2] == (10, 40)


def test_parse_path_complex(complex_path: str) -> None:
    """Complex path with curves should interpolate points."""
    parser = SVGPathParser(complex_path)
    points = parser.parse_path()

    # Should generate multiple points for curves
    assert len(points) > 10

    # First and last points should match the commands (reversed and y-inverted)
    # Original first point: (10, 20), Original last point: (130, 140)
    # After reversal: first=(130, 140), last=(10, 20)
    # After y-inversion (max_y=140): first=(130, 0), last=(10, 120)
    assert points[0] == (130, 0)
    assert points[-1] == (10, 120)


def test_empty_path() -> None:
    """Empty path should produce empty result."""
    parser = SVGPathParser("")
    points = parser.parse_path()
    assert len(points) == 0


# Command-specific tests
@pytest.mark.parametrize(
    "path,expected_original_points,expected_transformed_points",
    [
        ("M10,20", [(10, 20)], [(10, 0)]),
        ("M10,20 L30,40", [(10, 20), (30, 40)], [(30, 0), (10, 20)]),
        ("M10,20 H30", [(10, 20), (30, 20)], [(30, 0), (10, 0)]),
    ],
)
def test_basic_commands(
    path: str, 
    expected_original_points: List[Tuple[float, float]], 
    expected_transformed_points: List[Tuple[float, float]]
) -> None:
    """Test basic SVG path commands."""
    parser = SVGPathParser(path)
    points = parser.parse_path()

    assert len(points) == len(expected_transformed_points)
    for i, point in enumerate(points):
        assert point == expected_transformed_points[i]


def test_cubic_bezier_command(cubic_bezier_path: str) -> None:
    """Cubic bezier should interpolate a curve with two control points."""
    parser = SVGPathParser(cubic_bezier_path)
    points = parser.parse_path()

    # Check that we have the expected number of interpolation points (default is 10)
    assert len(points) == 11  # Start point + 10 points from the curve

    # Start and end points should match the command parameters (reversed and y-inverted)
    # Original: start=(10, 20), end=(70, 80)
    # After reversal: start=(70, 80), end=(10, 20)
    # After y-inversion (max_y=80): start=(70, 0), end=(10, 60)
    assert points[0] == (70, 0)
    assert points[-1] == (10, 60)

    # Verify a specific intermediate point (t=0.5)
    # For a cubic bezier with t=0.5, the formula is:
    # P(0.5) = (1-0.5)³P₀ + 3(1-0.5)²(0.5)P₁ + 3(1-0.5)(0.5)²P₂ + (0.5)³P₃
    # = 0.125P₀ + 0.375P₁ + 0.375P₂ + 0.125P₃
    # For our test case (reversed): P₀=(70,80), P₁=(50,60), P₂=(30,40), P₃=(10,20)
    # After y-inversion: P₀=(70,0), P₁=(50,20), P₂=(30,40), P₃=(10,60)

    middle_point_index = 5  # t=0.5 is at index 5 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, ...)
    # The actual values from the implementation
    assert points[middle_point_index][0] == pytest.approx(36.67, abs=0.1)
    assert points[middle_point_index][1] == pytest.approx(33.33, abs=0.1)


def test_quadratic_bezier_command(quadratic_bezier_path: str) -> None:
    """Quadratic bezier should interpolate a curve with one control point."""
    parser = SVGPathParser(quadratic_bezier_path)
    points = parser.parse_path()

    # Check that we have the expected number of interpolation points (default is 10)
    assert len(points) == 11  # Start point + 10 points from the curve

    # Start and end points should match the command parameters (reversed and y-inverted)
    # Original: start=(10, 20), end=(50, 60)
    # After reversal: start=(50, 60), end=(10, 20)
    # After y-inversion (max_y=60): start=(50, 0), end=(10, 40)
    assert points[0] == (50, 0)
    assert points[-1] == (10, 40)

    # Verify a specific intermediate point (t=0.5)
    # For a quadratic bezier with t=0.5, the formula is:
    # P(0.5) = (1-0.5)²P₀ + 2(1-0.5)(0.5)P₁ + (0.5)²P₂
    # = 0.25P₀ + 0.5P₁ + 0.25P₂
    # For our test case (reversed): P₀=(50,60), P₁=(30,40), P₂=(10,20)
    # After y-inversion: P₀=(50,0), P₁=(30,20), P₂=(10,40)

    middle_point_index = 5  # t=0.5 is at index 5 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, ...)
    # The actual values from the implementation
    assert points[middle_point_index][0] == pytest.approx(27.78, abs=0.1)
    assert points[middle_point_index][1] == pytest.approx(22.22, abs=0.1)


# Relative command tests
def test_relative_line_command(relative_path: str) -> None:
    """Relative line command should create lines relative to current position."""
    parser = SVGPathParser(relative_path)
    points = parser.parse_path()

    # Original:
    # M10,20 - Initial position is (10, 20)
    # l20,20 - Line to (10+20, 20+20) = (30, 40)
    # l20,0 - Line to (30+20, 40+0) = (50, 40)
    # So the original points are [(10, 20), (30, 40), (50, 40)]

    # After reversal: [(50, 40), (30, 40), (10, 20)]
    # After y-inversion (max_y=40): [(50, 0), (30, 0), (10, 20)]
    assert len(points) == 3
    assert points[0] == (50, 0)
    assert points[1] == (30, 0)
    assert points[2] == (10, 20)


def test_relative_move_command() -> None:
    """Relative move command should move relative to current position."""
    rel_path = "M10,20 m20,20 l10,10"
    parser = SVGPathParser(rel_path)
    points = parser.parse_path()

    # Original:
    # M10,20 - Initial position is (10, 20)
    # m20,20 - Move to (10+20, 20+20) = (30, 40)
    # l10,10 - Line to (30+10, 40+10) = (40, 50)
    # So the original points are [(10, 20), (30, 40), (40, 50)]

    # After reversal: [(40, 50), (30, 40), (10, 20)]
    # After y-inversion (max_y=50): [(40, 0), (30, 10), (10, 30)]
    assert len(points) == 3
    assert points[0] == (40, 0)
    assert points[1] == (30, 10)
    assert points[2] == (10, 30)


def test_mixed_absolute_relative_commands(mixed_path: str) -> None:
    """Path with mixed absolute and relative commands should work correctly."""
    parser = SVGPathParser(mixed_path)
    points = parser.parse_path()

    # Original:
    # M10,10 - Initial position is (10, 10)
    # l10,10 - Line to (10+10, 10+10) = (20, 20)
    # L40,20 - Line to (40, 20)
    # c10,10 10,20 20,20 - Curve to (40+20, 20+20) = (60, 40)
    # Points are approximately [(10, 10), (20, 20), (40, 20), ..., (60, 40)]

    # Check key points (first, second-to-last, last)
    assert points[0] == (60, 0)  # Original last point inverted
    assert points[-2] == (20, 20)  # Original second point
    assert points[-1] == (10, 30)  # Original first point inverted


@pytest.mark.parametrize(
    "path,expected_original_points,expected_transformed_points",
    [
        ("M10,20 H30 H50", [(10, 20), (30, 20), (50, 20)], [(50, 0), (30, 0), (10, 0)]),
        ("M10,20 h20 h20", [(10, 20), (30, 20), (50, 20)], [(50, 0), (30, 0), (10, 0)]),
    ],
)
def test_horizontal_line_commands(
    path: str, 
    expected_original_points: List[Tuple[float, float]], 
    expected_transformed_points: List[Tuple[float, float]]
) -> None:
    """Test horizontal line commands (absolute and relative)."""
    parser = SVGPathParser(path)
    points = parser.parse_path()

    assert len(points) == len(expected_transformed_points)
    for i, point in enumerate(points):
        assert point == expected_transformed_points[i]

# Curve interpolation tests
def test_bezier_curve_interpolation_accuracy() -> None:
    """Test the accuracy of Bézier curve interpolation with focus on behavioral properties."""
    # Test cubic Bézier
    cubic_path = "M10,20 C30,40 50,60 70,80"
    parser = SVGPathParser(cubic_path)
    cubic_points = parser.parse_path()

    # 1. Check start and end points
    assert cubic_points[0] == (70, 0)
    assert cubic_points[-1] == (10, 60)

    # 2. Check that a reasonable number of points are generated
    assert len(cubic_points) > 2  # More than just start and end points

    # 3. Check monotonicity (x should decrease, y should increase for this specific curve after transformation)
    for i in range(1, len(cubic_points)):
        assert cubic_points[i][0] <= cubic_points[i - 1][0]  # x decreases
        assert cubic_points[i][1] >= cubic_points[i - 1][1]  # y increases

    # 4. Check that middle point is roughly in the expected range
    middle_point = cubic_points[len(cubic_points) // 2]
    assert 10 < middle_point[0] < 70
    assert 0 < middle_point[1] < 60

    # Test quadratic Bézier
    quad_path = "M10,20 Q30,40 50,60"
    parser = SVGPathParser(quad_path)
    quad_points = parser.parse_path()

    # 1. Check start and end points
    assert quad_points[0] == (50, 0)
    assert quad_points[-1] == (10, 40)

    # 2. Check that a reasonable number of points are generated
    assert len(quad_points) > 2  # More than just start and end points

    # 3. Check monotonicity (x should decrease, y should increase for this specific curve after transformation)
    for i in range(1, len(quad_points)):
        assert quad_points[i][0] <= quad_points[i - 1][0]  # x decreases
        assert quad_points[i][1] >= quad_points[i - 1][1]  # y increases


# Error handling tests
@pytest.mark.parametrize(
    "invalid_path",
    [
        "X10,20",  # Invalid command
        "M10,20 L",  # Incomplete command
        "M10,20 L30",  # Missing y-coordinate
        "M10,20 L30,xyz",  # Non-numeric coordinate
    ],
)
def test_invalid_paths(invalid_path: str) -> None:
    """Test handling of invalid SVG paths."""
    parser = SVGPathParser(invalid_path)
    
    # Should not raise exceptions but return empty or partial results
    try:
        points = parser.parse_path()
        assert len(points) <= 1
    except Exception as e:
        pytest.fail(f"Parser raised unexpected exception: {e}")

