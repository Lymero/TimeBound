"""
Main application entry point for the winrates analysis tool.

This module provides the main entry point for the application, tying together
the data acquisition, processing, analysis, and visualization components.
"""

import argparse
import asyncio
import json
import traceback
from pathlib import Path

import numpy as np
from rich.console import Console

from src.acquisition.svg_extractor import extract_champions, get_all_champion_names
from src.analysis.champion_path_clustering.clusterer import ChampionPathClusterer
from src.analysis.champion_path_clustering.factory import (
    ClusteringStrategyFactory,
    StrategyType,
)
from src.utils.logger import error, info, success, warning
from src.utils.matrix_operations import create_distance_matrix
from src.visualization.cluster_visualizer import ClusterVisualizer


def lowercase_str(arg_value: str) -> str:
    """Custom argparse type for case-insensitive string handling."""
    return arg_value.lower()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="League of Legends Winrate Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data acquisition parser
    acquire_parser = subparsers.add_parser(
        "acquire", 
        help="Acquire champion winrate SVG data from LoLalytics"
    )
    acquire_parser.add_argument(
        "--patch",
        type=lowercase_str,
        default="30",
        help="Patch to scrape data for (e.g., '30', 'current')"
    )
    acquire_parser.add_argument(
        "--tier",
        type=lowercase_str,
        default="all",
        help="Rank tier to scrape data for (e.g., 'all', 'platinum_plus')"
    )
    acquire_parser.add_argument(
        "--region",
        type=lowercase_str,
        default="all",
        help="Region to scrape data for (e.g., 'all', 'na')"
    )
    acquire_parser.add_argument(
        "--output",
        type=str,
        default="data/champion_svg_paths.json",
        help="Path to save the extracted SVG paths"
    )
    acquire_parser.add_argument(
        "--champions",
        nargs="+",
        type=lowercase_str,
        help="List of champions to process (space-separated, case-insensitive)"
    )
    acquire_parser.add_argument(
        "--all",
        action="store_true",
        help="Process all champions"
    )
    acquire_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        choices=range(1, 11),
        help="Number of concurrent extractions"
    )
    
    # Clustering parser
    cluster_parser = subparsers.add_parser(
        "cluster", 
        help="Cluster champions based on winrate vs game length"
    )
    cluster_parser.add_argument(
        "--svg-paths-file",
        type=str,
        default="data/champion_svg_paths.json",
        help="Path to the JSON file containing champion SVG paths",
    )
    cluster_parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Threshold for considering two champions as similar",
    )
    cluster_parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of clusters to create (0 for automatic determination)",
    )
    cluster_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory to save output visualizations",
    )
    cluster_parser.add_argument(
        "--champion", 
        type=lowercase_str, 
        help="Find champions similar to this champion (case-insensitive)"
    )
    cluster_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of champions to process (0 for all)",
    )
    cluster_parser.add_argument(
        "--clustering-method",
        type=str,
        choices=[*ClusteringStrategyFactory.get_available_strategies()],
        default=StrategyType.THRESHOLD.value,
        help="Clustering method to use",
    )
    cluster_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML visualizations",
    )
    
    # General options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


async def acquire_data(args: argparse.Namespace) -> None:
    """
    Acquire champion winrate SVG data from LoLalytics.
    
    Args:
        args: Command line arguments
    """
    info(f"Starting data acquisition for patch={args.patch}, tier={args.tier}, region={args.region}")
    
    if args.all:
        champions = get_all_champion_names()
        info(f"Processing all champions ({len(champions)} champions)")
    elif args.champions:
        champions = args.champions
        info(f"Processing specified champions: {', '.join(champions)}")
    else:
        champions = ["kogmaw"]
        info(f"No champions specified, defaulting to: {champions[0]}")
    
    result = await extract_champions(
        champions=champions,
        concurrency=args.concurrency,
        patch=args.patch,
        tier=args.tier,
        region=args.region,
        output_file=args.output,
    )
    
    success(f"Successfully acquired SVG path data for {len(result)} champions")
    success(f"Data saved to {args.output}")


def process_similar_champions(
    clusterer: ChampionPathClusterer, 
    champion: str, 
    output_dir: Path
) -> None:
    """
    Find and save champions similar to the given champion.
    
    Args:
        clusterer: The champion path clusterer
        champion: The champion to find similar champions for (in lowercase)
        output_dir: Directory to save output files
    """
    # Extract points if they haven't been extracted yet
    if not clusterer.champion_points:
        clusterer._extract_points()
    
    # Since champion is already lowercase, find matching champion with correct capitalization
    champion_key = None
    for champ in clusterer.champion_points:
        if champ.lower() == champion:
            champion_key = champ
            break
    
    if not champion_key:
        error(f"Champion '{champion}' not found. Please check the name and try again.")
        return
    
    info(f"Finding champions similar to {champion_key}...")
    similar_champions = clusterer.get_similar_champions(champion_key)

    if similar_champions:
        info(f"Champions similar to {champion_key}:")
        for champ, similarity in similar_champions:
            info(f"  {champ}: {similarity:.4f}")

        similar_champions_file = output_dir / f"similar_to_{champion}.json"
        with open(similar_champions_file, "w", encoding="utf-8") as f:
            json.dump(dict(similar_champions), f, indent=2)
        info(f"Saved similar champions to {similar_champions_file}")
    else:
        warning(f"No champions similar to {champion_key} found")


def save_clusters_to_json(clusters: dict[int, list[str]], output_path: Path) -> None:
    """
    Save clusters to a JSON file.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of champions
        output_path: Path where to save the JSON file
    """
    try:
        serializable_clusters = {int(k): v for k, v in clusters.items()}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_clusters, f, indent=2)
        info(f"Saved clusters to {output_path}")
    except Exception as e:
        error(f"Error saving clusters to {output_path}: {e}")


def create_visualizations(
    clusterer: ChampionPathClusterer,
    clusters: dict[int, list[str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """
    Create various visualizations based on clustering results.
    
    Args:
        clusterer: The champion path clusterer
        clusters: Dictionary mapping cluster IDs to lists of champions
        args: Command line arguments
        output_dir: Directory to save visualizations
        
    Returns:
        A tuple containing champions list, distance matrix, and champion-to-cluster mapping
    """
    champions = list(clusterer.champion_points.keys())
    
    visualizer = ClusterVisualizer()
    
    distance_matrix = create_distance_matrix(
        champions=champions,
        correlation_matrix=clusterer.correlation_matrix
    )
    
    champion_to_cluster = {
        champion: cluster_id
        for cluster_id, champions_in_cluster in clusters.items()
        for champion in champions_in_cluster
    }

    method = args.clustering_method

    # Basic cluster visualization
    clusters_viz_file = output_dir / f"{method}_clusters.png"
    visualizer.visualize_clusters(
        champions=champions,
        distance_matrix=distance_matrix,
        champion_to_cluster=champion_to_cluster,
        output_path=str(clusters_viz_file),
    )
    info(f"Created cluster visualization at {clusters_viz_file}")

    # Interactive visualization if requested
    if args.interactive:
        interactive_viz_file = output_dir / f"{method}_clusters_interactive.html"
        visualizer.visualize_clusters_interactive(
            champions=champions,
            distance_matrix=distance_matrix,
            champion_to_cluster=champion_to_cluster,
            output_path=str(interactive_viz_file),
        )
        info(f"Created interactive cluster visualization at {interactive_viz_file}")

    # Correlation network visualization
    network_viz_file = output_dir / f"{method}_correlation_network.png"
    visualizer.visualize_correlation_network(
        champions=champions,
        correlation_matrix=clusterer.correlation_matrix,
        correlation_threshold=clusterer.correlation_threshold,
        output_path=str(network_viz_file),
    )
    info(f"Created correlation network visualization at {network_viz_file}")

    # Cluster profiles visualization
    profiles_dir = output_dir / f"{method}_cluster_profiles"
    profiles_dir.mkdir(exist_ok=True)
    visualizer.visualize_cluster_profiles(
        champion_points=clusterer.champion_points,
        clusters=clusters,
        output_dir=str(profiles_dir),
        prefix=f"{method}_cluster",
    )
    info(f"Created cluster profiles in {profiles_dir}")

    return champions, distance_matrix, champion_to_cluster


def cluster_champions(args: argparse.Namespace) -> None:
    """
    Cluster champions based on winrate vs game length.
    
    Args:
        args: Command line arguments
    """
    info("Starting champion clustering analysis")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clustering_strategy = ClusteringStrategyFactory.create_strategy(
        args.clustering_method
    )

    clusterer = ChampionPathClusterer(
        svg_paths_file=args.svg_paths_file,
        correlation_threshold=args.correlation_threshold,
        limit=args.limit,
        clustering_strategy=clustering_strategy,
    )
    
    # If a specific champion is provided, find similar champions
    if args.champion:
        process_similar_champions(clusterer, args.champion, output_dir)
        return
        
    # Otherwise, perform clustering
    info(f"Clustering champions using {args.clustering_method} method...")
    n_clusters = args.n_clusters if args.n_clusters > 0 else None
    clusters = clusterer.cluster_champions(n_clusters)
    
    # Print cluster results
    info(f"Found {len(clusters)} clusters:")
    for cluster_id, champions in clusters.items():
        info(f"Cluster {cluster_id}: {len(champions)} champions")
        info(f"  {', '.join(champions)}")
    
    # Save clusters to JSON
    clusters_file = output_dir / f"{args.clustering_method}_clusters.json"
    save_clusters_to_json(clusters, clusters_file)
    
    # Create visualizations
    info("Creating visualizations...")
    champions, distance_matrix, champion_to_cluster = create_visualizations(
        clusterer, clusters, args, output_dir
    )
    
    success("Champion clustering analysis completed successfully")


def main() -> None:
    """Main entry point for the application."""
    args = parse_arguments()
    
    console = Console()
    
    try:
        if args.command == "acquire":
            asyncio.run(acquire_data(args))
        elif args.command == "cluster":
            cluster_champions(args)
        else:
            console.print("[bold yellow]Please specify a command: acquire or cluster[/bold yellow]")
            console.print("Run with --help for more information")
    except KeyboardInterrupt:
        console.print("[bold red]Operation cancelled by user[/bold red]")
    except Exception as e:
        error(f"An unexpected error occurred: {e}")
        if args.debug:
            error(traceback.format_exc())


if __name__ == "__main__":
    main()
