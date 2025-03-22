"""
SVG Path Parser Module.

This module provides functionality for parsing SVG path data into point sequences
for function graph analysis and visualization.
"""

import math
import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import error, info, warning

COORD_PAIR_SIZE = 2
MIN_TIME = 15
MAX_TIME = 40
BASELINE_WINRATE = 50      
MIN_WINRATE_DEVIATION = -10
MAX_WINRATE_DEVIATION = 10 

class SVGPathParser:
    """
    A class to parse SVG path data into point sequences for function graph analysis.

    This parser is optimized for extracting coordinate points from SVG paths that
    represent mathematical function graphs. It supports the essential SVG path commands
    for function representation: move, line, horizontal line, cubic bezier, and
    quadratic bezier curves.

    The parser converts SVG path commands into a sequence of (x,y) coordinate points
    that can be used for correlation analysis between different function graphs.
    """

    def __init__(self, path_data: str) -> None:
        """
        Initialize the parser with SVG path data.

        Args:
            path_data: The SVG path data string containing commands and coordinates
        """
        self.path_data = path_data
        self.command_pattern = re.compile(r"[MLHCSQmlhcsq]")
        self.number_pattern = re.compile(r"-?\d*\.?\d+(?:[eE][+-]?\d+)?")
        self.current_position = (0.0, 0.0)

        # Command handlers dictionary
        self._command_handlers: dict[str, Callable] = {
            "M": self._handle_move_to,
            "m": self._handle_move_to_relative,
            "L": self._handle_line_to,
            "l": self._handle_line_to_relative,
            "H": self._handle_horizontal_line,
            "h": self._handle_horizontal_line_relative,
            "C": self._handle_cubic_bezier,
            "c": self._handle_cubic_bezier_relative,
            "Q": self._handle_quadratic_bezier,
            "q": self._handle_quadratic_bezier_relative,
        }

    def parse_path(self) -> list[tuple[float, float]]:
        """
        Parse the SVG path data into a list of coordinate points.

        This method processes the SVG path data string, extracting commands and their
        coordinates, and converts them into a sequence of (x,y) points that represent
        the function graph. It handles all supported commands (move, line, curves)
        and maintains the continuity of the path.

        Note:
        1. The points are reversed to correct the order (SVG paths are often defined from right to left)
        2. The y-coordinates are inverted to correct the orientation
           (in SVG, y increases downward, but we want y to increase upward for analysis)
        3. X-coordinates are transformed to game time minutes (15-40)
        4. Y-coordinates are transformed to win rate deviations from baseline (typically ±10%)

        Returns:
            List of (x,y) coordinate tuples representing the function graph
        """
        if not self.path_data:
            return []

        points = []
        self.current_position = (0.0, 0.0)

        try:
            # Split the path data into command and coordinate segments
            segments = self.command_pattern.split(self.path_data)
            commands = self.command_pattern.findall(self.path_data)

            # Process each command and its coordinates
            for command, segment in zip(commands, segments[1:], strict=False):
                handler = self._command_handlers.get(command)
                if handler:
                    coordinates = self.number_pattern.findall(segment)
                    new_points = handler(coordinates)
                    if new_points:
                        points.extend(new_points)
                else:
                    warning(f"Unsupported command '{command}' - skipping")
        except ValueError as e:
            error(f"Error parsing SVG path coordinates: {e}")
        except IndexError as e:
            error(f"Error processing SVG path segments: {e}")
        except Exception as e:
            error(f"Unexpected error parsing SVG path: {e}")

        # Reverse the points to fix the order
        reversed_points = list(reversed(points))

        # Invert the y-coordinates
        if reversed_points:
            max_y = max(y for _, y in reversed_points)
            # Invert y-coordinates (y = max_y - y)
            inverted_points = [(x, max_y - y) for x, y in reversed_points]
            
            time_points = self._transform_to_game_time(inverted_points)
            return self._transform_to_winrate_deviation(time_points)

        return []

    def _handle_move_to(self, coordinates: list[str]) -> list[tuple[float, float]]:
        """Handle the Move To (M) command."""
        points = []

        if len(coordinates) >= COORD_PAIR_SIZE:
            x, y = float(coordinates[0]), float(coordinates[1])
            self.current_position = (x, y)
            points.append(self.current_position)

            # If there are more than one pair of coordinates, treat them as line_to commands
            # e.g. a path like M10,20 30,40 50,60 is equivalent to M10,20 L30,40 L50,60
            if len(coordinates) > COORD_PAIR_SIZE:
                line_coordinates = coordinates[COORD_PAIR_SIZE:]
                line_points = self._handle_line_to(line_coordinates)
                if line_points:
                    points.extend(line_points)

        return points

    def _convert_relative_to_absolute(self, coordinates: list[str]) -> list[str]:
        """
        Convert relative coordinates to absolute coordinates.

        Args:
            coordinates: List of relative coordinate strings

        Returns:
            List of absolute coordinate strings
        """
        absolute_coordinates = []

        for i in range(0, len(coordinates), 2):
            if i + 1 < len(coordinates):
                rel_x = float(coordinates[i])
                abs_x = self.current_position[0] + rel_x
                absolute_coordinates.append(str(abs_x))
                rel_y = float(coordinates[i + 1])
                abs_y = self.current_position[1] + rel_y
                absolute_coordinates.append(str(abs_y))

        return absolute_coordinates

    def _handle_move_to_relative(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the relative Move To (m) command."""
        absolute_coordinates = self._convert_relative_to_absolute(coordinates)
        return self._handle_move_to(absolute_coordinates)

    def _handle_line_to(self, coordinates: list[str]) -> list[tuple[float, float]]:
        """Handle the Line To (L) command."""
        points = []

        for i in range(0, len(coordinates), 2):
            if i + 1 < len(coordinates):
                x, y = float(coordinates[i]), float(coordinates[i + 1])
                self.current_position = (x, y)
                points.append(self.current_position)

        return points

    def _handle_line_to_relative(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the relative Line To (l) command."""
        absolute_coordinates = self._convert_relative_to_absolute(coordinates)
        return self._handle_line_to(absolute_coordinates)

    def _handle_horizontal_line(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the Horizontal Line (H) command."""
        points = []

        for param in coordinates:
            x = float(param)
            y = self.current_position[1]
            self.current_position = (x, y)
            points.append(self.current_position)

        return points

    def _handle_horizontal_line_relative(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the relative Horizontal Line (h) command."""
        absolute_coordinates = []

        for param in coordinates:
            # For horizontal lines, only the x coordinate is relative
            rel_x = float(param)
            abs_x = self.current_position[0] + rel_x
            absolute_coordinates.append(str(abs_x))

        return self._handle_horizontal_line(absolute_coordinates)

    def _handle_cubic_bezier(self, coordinates: list[str]) -> list[tuple[float, float]]:
        """Handle the Cubic Bezier Curve (C) command."""
        points = []

        for i in range(0, len(coordinates), 6):
            if i + 5 < len(coordinates):
                x1, y1 = float(coordinates[i]), float(coordinates[i + 1])
                x2, y2 = float(coordinates[i + 2]), float(coordinates[i + 3])
                x, y = float(coordinates[i + 4]), float(coordinates[i + 5])

                # Generate points along the bezier curve
                curve_points = self._generate_bezier_points(
                    [self.current_position, (x1, y1), (x2, y2), (x, y)], num_points=10
                )

                # Skip the first point (start point) to avoid duplication
                # except for the first curve segment
                if i == 0:
                    points.extend(curve_points)
                else:
                    points.extend(curve_points[1:])

                self.current_position = (x, y)

        return points

    def _handle_cubic_bezier_relative(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the relative Cubic Bezier Curve (c) command."""
        absolute_coordinates = self._convert_relative_to_absolute(coordinates)
        return self._handle_cubic_bezier(absolute_coordinates)

    def _handle_quadratic_bezier(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the Quadratic Bezier Curve (Q) command."""
        points = []

        for i in range(0, len(coordinates), 4):
            if i + 3 < len(coordinates):
                x1, y1 = float(coordinates[i]), float(coordinates[i + 1])
                x, y = float(coordinates[i + 2]), float(coordinates[i + 3])
                curve_points = self._generate_bezier_points(
                    [self.current_position, (x1, y1), (x, y)], num_points=10
                )

                # Skip the first point (start point) to avoid duplication
                # except for the first curve segment
                if i == 0:
                    points.extend(curve_points)
                else:
                    points.extend(curve_points[1:])

                self.current_position = (x, y)

        return points

    def _handle_quadratic_bezier_relative(
        self, coordinates: list[str]
    ) -> list[tuple[float, float]]:
        """Handle the relative Quadratic Bezier Curve (q) command."""
        absolute_coordinates = self._convert_relative_to_absolute(coordinates)
        return self._handle_quadratic_bezier(absolute_coordinates)

    def _generate_bezier_points(
        self, control_points: list[tuple[float, float]], num_points: int = 10
    ) -> list[tuple[float, float]]:
        """
        Generate points along a Bezier curve of any degree.

        Args:
            control_points: List of control points [P0, P1, ..., Pn] where:
                            P0 is the start point
                            Pn is the end point
                            P1...P(n-1) are the control points
            num_points: Number of points to generate

        Returns:
            List of points along the curve

        Note:
            Points are generated at evenly spaced t-values from 0 to 1.
            This means the first point is at t=0, the last point is at t=1, and intermediate points are
            at t-values of 1/(num_points-1), 2/(num_points-1), etc.

            This method implements the general form of the Bezier curve using the Bernstein basis polynomials.
        """
        points = []
        n = len(control_points) - 1  # Degree of the Bezier curve

        for t in np.linspace(0, 1, num_points):
            x = 0.0
            y = 0.0

            for i in range(n + 1):
                binomial = math.comb(n, i)
                bernstein = binomial * (1 - t) ** (n - i) * t**i

                # Add weighted contribution of this control point
                x += bernstein * control_points[i][0]
                y += bernstein * control_points[i][1]

            points.append((x, y))

        return points

    def plot_path(
        self,
        figsize: tuple[int, int] = (10, 6),
        color: str = "b",
        show_points: bool = True,
    ) -> None:
        """
        Visualize the SVG path using matplotlib.

        Args:
            figsize: Tuple of (width, height) for the plot
            color: Color for the path line
            show_points: Whether to show points on the path
        """
        points = self.parse_path()

        if not points:
            info("No points to plot")
            return

        x_coords, y_coords = zip(*points, strict=False)

        plt.figure(figsize=figsize)

        # Plot the path
        if show_points:
            plt.plot(x_coords, y_coords, f"{color}.-", label="Path Points")
        else:
            plt.plot(x_coords, y_coords, f"{color}-", label="Path")

        plt.grid(True)
        plt.title("SVG Path Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        # Flip y-axis to match SVG coordinate system
        plt.gca().invert_yaxis()
        plt.show()

    def save_to_file(self, filename: str, format: str = "png", dpi: int = 300) -> None:
        """
        Save the plot to a file.

        Args:
            filename: Name of the file to save to
            format: File format (png, jpg, svg, pdf)
            dpi: Resolution for raster formats
        """
        points = self.parse_path()

        if not points:
            info("No points to save")
            return

        x_coords, y_coords = zip(*points, strict=False)

        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, y_coords, "b.-", label="Path Points")
        plt.grid(True)
        plt.title("SVG Path Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        # Flip y-axis to match SVG coordinate system
        plt.gca().invert_yaxis()

        # Save to file
        plt.savefig(filename, format=format, dpi=dpi)
        plt.close()
        info(f"Plot saved to {filename}")

    def _transform_to_game_time(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Transform SVG x-coordinates to actual game time minutes using linear mapping.
        
        The SVG x-coordinates are linearly mapped from their original range to the game time
        range defined by MIN_TIME and MAX_TIME constants (15-40 minutes).
        
        Args:
            points: List of (x,y) coordinate tuples
            
        Returns:
            List of transformed (x,y) coordinate tuples with x representing actual game time minutes
        """
        if not points:
            return []
            
        x_coords, y_coords = np.array([x for x, _ in points]), np.array([y for _, y in points])
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        if min_x == max_x:
            return [(MAX_TIME, y) for y in y_coords]

        # Linear transformation from original x range to game time range (15-40 minutes)
        game_times = MIN_TIME + (MAX_TIME - MIN_TIME) * (x_coords - min_x) / (max_x - min_x)

        return list(zip(game_times, y_coords, strict=True))
    
    def _transform_to_winrate_deviation(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Transform SVG y-coordinates to normalized win rate deviations.
        
        The SVG y-coordinates are linearly mapped to represent deviations from 
        the baseline win rate (typically 50%), with the range defined by 
        MIN_WINRATE_DEVIATION and MAX_WINRATE_DEVIATION constants.
        
        This produces values that show relative performance differences between
        champions rather than absolute win rate percentages. The final values can be
        interpreted as: BASELINE_WINRATE + deviation (e.g., 50% ± 10%).
        
        Args:
            points: List of (x,y) coordinate tuples
            
        Returns:
            List of transformed (x,y) coordinate tuples with y representing win rate deviations
        """
        if not points:
            return []
            
        x_coords, y_coords = np.array([x for x, _ in points]), np.array([y for _, y in points])
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        if min_y == max_y:
            return [(x, 0.0) for x in x_coords]
            
        # Linear transformation from original y range to win rate deviation range
        win_rate_devs = MIN_WINRATE_DEVIATION + (MAX_WINRATE_DEVIATION - MIN_WINRATE_DEVIATION) * (y_coords - min_y) / (max_y - min_y)
        
        return list(zip(x_coords, win_rate_devs, strict=True))
