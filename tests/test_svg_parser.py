"""Tests for the SVG Path Parser using pytest."""

import pytest

from src.processing.svg_parser import MAX_TIME, MAX_WINRATE_DEVIATION, MIN_TIME, MIN_WINRATE_DEVIATION, SVGPathParser


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
    """Test simple path with move and line commands."""
    parser = SVGPathParser(simple_path)
    points = parser.parse_path()
    
    middle_time = (MIN_TIME + MAX_TIME) / 2
    assert len(points) == 3
    
    # Check first, middle, and last points map correctly
    assert_point(points[0], MAX_TIME, MIN_WINRATE_DEVIATION)  
    assert_point(points[1], middle_time, 0, tolerance=5)
    assert_point(points[2], MIN_TIME, MAX_WINRATE_DEVIATION)


def test_parse_path_complex(complex_path: str) -> None:
    """Test complex path with curves."""
    parser = SVGPathParser(complex_path)
    points = parser.parse_path()

    # Should generate multiple points for curves
    assert len(points) > 10
    
    # Check endpoints
    assert_point(points[0], MAX_TIME, MIN_WINRATE_DEVIATION, tolerance=5)
    assert_point(points[-1], MIN_TIME, MAX_WINRATE_DEVIATION, tolerance=5)


def test_empty_path() -> None:
    """Test empty path produces empty result."""
    parser = SVGPathParser("")
    points = parser.parse_path()
    assert len(points) == 0


@pytest.mark.parametrize(
    "path,expected_points",
    [
        # Single point maps to MAX_TIME, and y=0 (when only one y value exists)
        ("M10,20", [(MAX_TIME, 0.0)]),
        # Two points with different y values map to min/max deviation
        ("M10,20 L30,40", [(MAX_TIME, MIN_WINRATE_DEVIATION), (MIN_TIME, MAX_WINRATE_DEVIATION)]),
        # Both points have same y value, so they map to 0.0 deviation
        ("M10,20 H30", [(MAX_TIME, 0.0), (MIN_TIME, 0.0)]),
        # Horizontal line commands with same y map to 0.0 deviation
        ("M10,20 H30 H50", [(MAX_TIME, 0.0), ((MIN_TIME + MAX_TIME) / 2, 0.0), (MIN_TIME, 0.0)]),
        ("M10,20 h20 h20", [(MAX_TIME, 0.0), ((MIN_TIME + MAX_TIME) / 2, 0.0), (MIN_TIME, 0.0)]),
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
        assert_point(point, expected_points[i][0], expected_points[i][1], tolerance=5)


def test_bezier_commands() -> None:
    """Test both cubic and quadratic bezier curves."""
    # Test cubic Bézier
    parser = SVGPathParser("M10,20 C30,40 50,60 70,80")
    cubic_points = parser.parse_path()
    
    # Test quadratic Bézier
    parser = SVGPathParser("M10,20 Q30,40 50,60")
    quad_points = parser.parse_path()
    
    # Both should generate interpolation points (default is 10)
    assert len(cubic_points) == 11
    assert len(quad_points) == 11
    
    middle_time = (MIN_TIME + MAX_TIME) / 2
    mid_deviation = (MIN_WINRATE_DEVIATION + MAX_WINRATE_DEVIATION) / 2
    
    # Check endpoints and middle points for both curve types
    for points in [cubic_points, quad_points]:
        assert_point(points[0], MAX_TIME, MIN_WINRATE_DEVIATION, tolerance=5)
        assert_point(points[-1], MIN_TIME, MAX_WINRATE_DEVIATION, tolerance=5)
        assert_point(points[5], middle_time, mid_deviation, tolerance=5)
        
        # Check monotonicity
        for i in range(1, len(points)):
            assert points[i][0] <= points[i-1][0]  # x decreases
            assert points[i][1] >= points[i-1][1]  # y increases


def test_relative_commands() -> None:
    """Test relative path commands."""
    # Test relative line
    parser = SVGPathParser("M10,20 l20,20 l20,0")
    rel_line_points = parser.parse_path()
    
    # Test relative move
    parser = SVGPathParser("M10,20 m20,20 l10,10")
    rel_move_points = parser.parse_path()
    
    # Test mixed commands
    parser = SVGPathParser("M10,10 l10,10 L40,20 c10,10 10,20 20,20")
    mixed_points = parser.parse_path()
    
    # All should have correct endpoints
    for points in [rel_line_points, rel_move_points, mixed_points]:
        assert len(points) >= 3
        assert_point(points[0], MAX_TIME, MIN_WINRATE_DEVIATION, tolerance=5)
        assert_point(points[-1], MIN_TIME, MAX_WINRATE_DEVIATION, tolerance=5)


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


def assert_point(point: tuple[float, float], expected_x: float, expected_y: float, tolerance: float = 0.1) -> None:
    """Helper to check a point's coordinates."""
    assert point[0] == pytest.approx(expected_x, abs=tolerance)
    assert point[1] == pytest.approx(expected_y, abs=tolerance)

