#!/usr/bin/env python
"""Check the correlation between two champions."""

import argparse
import json

from src.analysis.graph_correlation import compare
from src.processing.svg_parser import SVGPathParser
from src.utils.logger import error, info


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check the correlation between two champions"
    )

    parser.add_argument("champion1", type=str, help="First champion name")

    parser.add_argument("champion2", type=str, help="Second champion name")

    parser.add_argument(
        "--svg-paths-file",
        type=str,
        default="data/champion_svg_paths.json",
        help="Path to the JSON file containing champion SVG paths",
    )

    return parser.parse_args()


def load_champion_svg_paths(file_path: str) -> dict:
    """Load champion SVG paths from a JSON file."""
    with open(file_path) as f:
        return json.load(f)


def extract_points(svg_path: str) -> list:
    """Extract points from an SVG path."""
    if not svg_path:
        return []

    parser = SVGPathParser(path_data=svg_path)
    return parser.parse_path()


def main() -> None:
    """Main function."""
    args = parse_arguments()

    info(f"Loading champion SVG paths from {args.svg_paths_file}...")
    svg_paths = load_champion_svg_paths(args.svg_paths_file)

    champion1 = args.champion1.lower()
    champion2 = args.champion2.lower()
    if champion1 not in svg_paths:
        error(f"Champion {champion1} not found")
        return
    if champion2 not in svg_paths:
        error(f"Champion {champion2} not found")
        return

    info(f"Extracting points for {champion1}...")
    points1 = extract_points(svg_paths[champion1]["svg_path"])

    info(f"Extracting points for {champion2}...")
    points2 = extract_points(svg_paths[champion2]["svg_path"])

    info(f"Computing correlation between {champion1} and {champion2}...")
    correlation = compare(points1, points2)
    info(f"Correlation between {champion1} and {champion2}: {correlation:.4f}")


if __name__ == "__main__":
    main()
