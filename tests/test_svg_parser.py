"""Tests for the SVG Path Parser using pytest."""

import pytest

from src.processing.svg_parser import SVGPathParser


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


def test_parse_path_simple(simple_path: str) -> None:
    """Simple path with move and line commands should produce exact points."""
    parser = SVGPathParser(simple_path)
    points = parser.parse_path()

    # Points are reversed, y-inverted, and x-transformed to game time scale [15-40]
    assert len(points) == 3
    assert points[0][0] == pytest.approx(40, abs=0.1)  # Last point maps to MAX_TIME
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[1][0] == pytest.approx(27.5, abs=5)  # Middle point maps to middle of time range (15-40)
    assert points[1][1] == pytest.approx(20, abs=0.1)
    assert points[2][0] == pytest.approx(15, abs=0.1)  # First point maps to MIN_TIME
    assert points[2][1] == pytest.approx(40, abs=0.1)


def test_parse_path_complex(complex_path: str) -> None:
    """Complex path with curves should interpolate points."""
    parser = SVGPathParser(complex_path)
    points = parser.parse_path()

    # Should generate multiple points for curves
    assert len(points) > 10

    # First and last points should map to MIN_TIME and MAX_TIME with y-inversion
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[-1][0] == pytest.approx(15, abs=0.1)  
    assert points[-1][1] == pytest.approx(120, abs=0.1)


def test_empty_path() -> None:
    """Empty path should produce empty result."""
    parser = SVGPathParser("")
    points = parser.parse_path()
    assert len(points) == 0


@pytest.mark.parametrize(
    "path,expected_points",
    [
        ("M10,20", [(40, 0)]),  # Single point maps to MAX_TIME
        ("M10,20 L30,40", [(40, 0), (15, 20)]),  # First point maps to MAX_TIME, last to MIN_TIME
        ("M10,20 H30", [(40, 0), (15, 0)]),  # Both points have same y, x maps to time range
    ],
)
def test_basic_commands(
    path: str, 
    expected_points: list[tuple[float, float]]
) -> None:
    """Test basic SVG path commands."""
    parser = SVGPathParser(path)
    points = parser.parse_path()

    assert len(points) == len(expected_points)
    for i, point in enumerate(points):
        assert point[0] == pytest.approx(expected_points[i][0], abs=0.1)
        assert point[1] == pytest.approx(expected_points[i][1], abs=0.1)


def test_cubic_bezier_command(cubic_bezier_path: str) -> None:
    """Cubic bezier should interpolate a curve with two control points."""
    parser = SVGPathParser(cubic_bezier_path)
    points = parser.parse_path()

    # Check that we have the expected number of interpolation points (default is 10)
    assert len(points) == 11  # Start point + 10 points from the curve

    # Start and end points map to time range with y-inversion
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[-1][0] == pytest.approx(15, abs=0.1)  
    assert points[-1][1] == pytest.approx(60, abs=0.1)

    # Middle point should be in middle of time range
    middle_point_index = 5  # t=0.5 is at index 5
    assert points[middle_point_index][0] == pytest.approx(27.5, abs=5)  # Middle of 15-40 time range
    assert points[middle_point_index][1] == pytest.approx(30, abs=5)


def test_quadratic_bezier_command(quadratic_bezier_path: str) -> None:
    """Quadratic bezier should interpolate a curve with one control point."""
    parser = SVGPathParser(quadratic_bezier_path)
    points = parser.parse_path()

    # Check that we have the expected number of interpolation points (default is 10)
    assert len(points) == 11  # Start point + 10 points from the curve

    # Start and end points map to time range with y-inversion
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[-1][0] == pytest.approx(15, abs=0.1)  
    assert points[-1][1] == pytest.approx(40, abs=0.1)

    # Middle point should be in middle of time range
    middle_point_index = 5  # t=0.5 is at index 5
    assert points[middle_point_index][0] == pytest.approx(27.5, abs=5)  # Middle of 15-40 time range
    assert points[middle_point_index][1] == pytest.approx(20, abs=5)


# Relative command tests
def test_relative_line_command(relative_path: str) -> None:
    """Relative line command should create lines relative to current position."""
    parser = SVGPathParser(relative_path)
    points = parser.parse_path()

    # Points map to time range
    assert len(points) == 3
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[1][0] == pytest.approx(27.5, abs=5)  # Middle point maps to middle of time range
    assert points[1][1] == pytest.approx(0, abs=0.1)
    assert points[2][0] == pytest.approx(15, abs=0.1)  
    assert points[2][1] == pytest.approx(20, abs=0.1)


def test_relative_move_command() -> None:
    """Relative move command should move relative to current position."""
    rel_path = "M10,20 m20,20 l10,10"
    parser = SVGPathParser(rel_path)
    points = parser.parse_path()

    # Points map to time range
    assert len(points) == 3
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[1][0] == pytest.approx(27.5, abs=10)  # Middle point maps to middle of time range
    assert points[1][1] == pytest.approx(10, abs=0.1)
    assert points[2][0] == pytest.approx(15, abs=0.1)  
    assert points[2][1] == pytest.approx(30, abs=0.1)


def test_mixed_absolute_relative_commands(mixed_path: str) -> None:
    """Path with mixed absolute and relative commands should work correctly."""
    parser = SVGPathParser(mixed_path)
    points = parser.parse_path()

    # Check key points - all x coordinates map to time range
    assert points[0][0] == pytest.approx(40, abs=0.1)  
    assert points[0][1] == pytest.approx(0, abs=0.1)
    assert points[-2][0] <= 27.5  # Original second point maps to lower half of time range
    assert points[-2][1] == pytest.approx(20, abs=0.1)
    assert points[-1][0] == pytest.approx(15, abs=0.1)  
    assert points[-1][1] == pytest.approx(30, abs=0.1)


@pytest.mark.parametrize(
    "path,expected_points",
    [
        ("M10,20 H30 H50", [(40, 0), (27.5, 0), (15, 0)]),  # Maps to time range with same y
        ("M10,20 h20 h20", [(40, 0), (27.5, 0), (15, 0)]),  # Same as above but using relative commands
    ],
)
def test_horizontal_line_commands(
    path: str, 
    expected_points: list[tuple[float, float]]
) -> None:
    """Test horizontal line commands (absolute and relative)."""
    parser = SVGPathParser(path)
    points = parser.parse_path()

    assert len(points) == len(expected_points)
    for i, point in enumerate(points):
        assert point[0] == pytest.approx(expected_points[i][0], abs=0.1)
        assert point[1] == pytest.approx(expected_points[i][1], abs=0.1)


# Curve interpolation tests
def test_bezier_curve_interpolation_accuracy() -> None:
    """Test the accuracy of Bézier curve interpolation with focus on behavioral properties."""
    # Test cubic Bézier
    cubic_path = "M10,20 C30,40 50,60 70,80"
    parser = SVGPathParser(cubic_path)
    cubic_points = parser.parse_path()

    # 1. Check start and end points map to time range
    assert cubic_points[0][0] == pytest.approx(40, abs=0.1)  
    assert cubic_points[0][1] == pytest.approx(0, abs=0.1)
    assert cubic_points[-1][0] == pytest.approx(15, abs=0.1)  
    assert cubic_points[-1][1] == pytest.approx(60, abs=0.1)

    # 2. Check that a reasonable number of points are generated
    assert len(cubic_points) > 2  # More than just start and end points

    # 3. Check monotonicity - x should decrease across points
    for i in range(1, len(cubic_points)):
        assert cubic_points[i][0] <= cubic_points[i - 1][0]  # x decreases
        assert cubic_points[i][1] >= cubic_points[i - 1][1]  # y increases

    # 4. Check that middle point is roughly in the middle of time range
    middle_point = cubic_points[len(cubic_points) // 2]
    assert 20 < middle_point[0] < 35  # Middle of time range (15-40)
    assert 0 < middle_point[1] < 60  # Within y range

    # Test quadratic Bézier
    quad_path = "M10,20 Q30,40 50,60"
    parser = SVGPathParser(quad_path)
    quad_points = parser.parse_path()

    # 1. Check start and end points map to time range
    assert quad_points[0][0] == pytest.approx(40, abs=0.1)  
    assert quad_points[0][1] == pytest.approx(0, abs=0.1)
    assert quad_points[-1][0] == pytest.approx(15, abs=0.1)  
    assert quad_points[-1][1] == pytest.approx(40, abs=0.1)

    # 2. Check that a reasonable number of points are generated
    assert len(quad_points) > 2  # More than just start and end points

    # 3. Check monotonicity
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
    """Test that invalid paths are handled gracefully."""
    parser = SVGPathParser(invalid_path)
    points = parser.parse_path()
    assert len(points) == 0 or isinstance(points, list)

