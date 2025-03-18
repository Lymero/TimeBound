"""
Champion Path Clustering Example.

This script demonstrates how to use the ChampionPathClusterer to group champions
based on the similarity of their win rate vs game length graphs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.champion_path_clustering.clusterer import ChampionPathClusterer
from src.analysis.champion_path_clustering.factory import (
    ClusteringStrategyFactory,
    StrategyType,
)
from src.utils.logger import error, info
from src.utils.matrix_operations import create_distance_matrix
from src.visualization.cluster_visualizer import ClusterVisualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster champions based on win rate vs game length graphs"
    )

    parser.add_argument(
        "--svg-paths-file",
        type=str,
        default="data/champion_svg_paths.json",
        help="Path to the JSON file containing champion SVG paths",
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Threshold for considering two champions as similar",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of clusters to create (0 for automatic determination)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory to save output visualizations",
    )

    parser.add_argument(
        "--champion", type=str, help="Find champions similar to this champion"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of champions to process (0 for all)",
    )

    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=[*ClusteringStrategyFactory.get_available_strategies(), "compare"],
        default=StrategyType.THRESHOLD.value,
        help="Clustering method to use",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML visualizations",
    )

    parser.add_argument(
        "--evaluate-metrics",
        action="store_true",
        help="Evaluate and visualize clustering metrics",
    )

    return parser.parse_args()


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


def process_similar_champions(
    clusterer: ChampionPathClusterer, champion: str, output_dir: Path
) -> None:
    """Find and save champions similar to the given champion."""
    info(f"Finding champions similar to {champion}...")
    similar_champions = clusterer.get_similar_champions(champion)

    if similar_champions:
        info(f"Champions similar to {champion}:")
        for champ, similarity in similar_champions:
            info(f"  {champ}: {similarity:.4f}")

        # Save similar champions to JSON
        similar_champions_file = output_dir / f"similar_to_{champion}.json"
        with open(similar_champions_file, "w", encoding="utf-8") as f:
            json.dump(dict(similar_champions), f, indent=2)
        info(f"Saved similar champions to {similar_champions_file}")
    else:
        info(f"No champions similar to {champion} found")


def create_visualizations(
    clusterer: ChampionPathClusterer,
    clusters: dict[int, list[str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Create various visualizations based on clustering results."""
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

    return champions, distance_matrix, champion_to_cluster


def evaluate_clustering_metrics(
    champions: list[str],
    distance_matrix: np.ndarray,
    champion_to_cluster: dict[str, int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Evaluate and visualize clustering metrics."""
    if not args.evaluate_metrics:
        return
        
    visualizer = ClusterVisualizer()
    method = args.clustering_method
    
    cluster_mappings = {method.capitalize(): champion_to_cluster}
    metrics_file = output_dir / f"{method}_metrics.png"
    
    silhouette_scores = visualizer.visualize_cluster_metrics(
        champions=champions,
        distance_matrix=distance_matrix,
        cluster_mappings=cluster_mappings,
        output_path=str(metrics_file),
    )
    info(f"Created clustering metrics visualization at {metrics_file}")
    info(f"Silhouette scores: {silhouette_scores}")

    # Save metrics to JSON
    metrics_json_file = output_dir / f"{method}_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump({"silhouette_scores": silhouette_scores}, f, indent=2)
    info(f"Saved clustering metrics to {metrics_json_file}")


def main() -> None:
    """Main entry point for the champion path clustering example."""
    args = parse_arguments()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the clusterer with the appropriate strategy
    clustering_strategy = ClusteringStrategyFactory.create_strategy(
        args.clustering_method
    )

    clusterer = ChampionPathClusterer(
        svg_paths_file=args.svg_paths_file,
        correlation_threshold=args.correlation_threshold,
        limit=args.limit if args.limit > 0 else None,
        clustering_strategy=clustering_strategy,
    )

    # If a specific champion is provided, find similar champions
    if args.champion:
        process_similar_champions(clusterer, args.champion, output_dir)

    info(f"Clustering champions using {args.clustering_method} method...")
    clusters = clusterer.cluster_champions(args.n_clusters)

    if clusters:
        clusters_file = output_dir / f"{args.clustering_method}_clusters.json"
        save_clusters_to_json(clusters, clusters_file)

        champions, distance_matrix, champion_to_cluster = create_visualizations(
            clusterer, clusters, args, output_dir
        )
        
        # Evaluate metrics if requested
        if args.evaluate_metrics:
            evaluate_clustering_metrics(
                champions, distance_matrix, champion_to_cluster, args, output_dir
            )

        stats = clusterer.get_cluster_stats(clusters)
        info(f"Cluster stats: {stats}")
    else:
        error("Clustering failed")


if __name__ == "__main__":
    main()
