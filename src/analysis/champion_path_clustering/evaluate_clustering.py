"""
Script to evaluate clustering performance with different parameters.

Runs clustering with various methods and parameters, measuring Silhouette Scores
to find optimal configurations.
"""

import argparse
from typing import Any

from rich.console import Console
from rich.table import Table

from src.analysis.champion_path_clustering.clusterer import ChampionPathClusterer
from src.analysis.champion_path_clustering.factory import ClusteringStrategyFactory

CLUSTERING_METHODS = ["hierarchical", "spectral"]
CLUSTER_COUNT_RANGE = range(10, 25)
CORRELATION_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
DEFAULT_FIXED_CLUSTER_COUNT = 22


console = Console()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for clustering evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate clustering performance with different parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=CLUSTERING_METHODS,
        nargs="+",
        default=CLUSTERING_METHODS,
        help="Only evaluate specific clustering methods"
    )
    
    parser.add_argument(
        "--evaluate-cluster-counts",
        action="store_true",
        default=False,
        help="Evaluate clustering with different cluster count values (default: True)"
    )
    
    parser.add_argument(
        "--evaluate-correlation-thresholds",
        action="store_true",
        default=False,
        help="Evaluate clustering with different correlation threshold values"
    )
    
    parser.add_argument(
        "--fixed-cluster-count",
        type=int,
        default=DEFAULT_FIXED_CLUSTER_COUNT,
        help="Fixed cluster count to use when evaluating correlation thresholds"
    )
    
    return parser.parse_args()


def evaluate_clustering(
    method: str, 
    n_clusters: int | None = None,
    correlation_threshold: float | None = None
) -> tuple[float, int]:
    """
    Evaluate a clustering configuration and return its silhouette score.
    
    Args:
        method: Clustering method to use
        n_clusters: Number of clusters for hierarchical/spectral methods
        correlation_threshold: Correlation threshold for considering champions as similar
        
    Returns:
        Tuple of (silhouette_score, num_clusters)
    """
    clustering_strategy = ClusteringStrategyFactory.create_strategy(method)
    clusterer = ChampionPathClusterer(
        clustering_strategy=clustering_strategy,
        correlation_threshold=correlation_threshold if correlation_threshold is not None else 0.8
    )
    
    clusters = clusterer.cluster_champions(n_clusters=n_clusters, evaluate=True)
    silhouette_score = clusterer.compute_silhouette_score() or -1.0
    
    return silhouette_score, len(clusters)


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print formatted table of clustering results."""
    if not results:
        console.print("[yellow]No valid Silhouette Scores obtained.[/]", style="yellow")
        return

    table = Table(title="Clustering Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim", width=12)
    table.add_column("k", justify="right")
    table.add_column("Correlation Threshold", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for i, res in enumerate(results):
        score_str = f"{res['score']:.4f}"
        style = "green" if i == 0 else ""
        
        table.add_row(
            res["method"],
            str(res["k"] if res["k"] is not None else "-"),
            str(res["correlation_threshold"] if "correlation_threshold" in res else "-"),
            str(res["clusters_found"]),
            score_str,
            style=style
        )
    console.print(table)


def print_best_result(result: dict[str, Any]) -> None:
    """Print details of the best clustering result."""
    console.print("\n[bold green]Best Result:[/]", style="green")
    console.print(f"  Method: {result['method']}")
    console.print(f"  Target k: {result['k'] if result['k'] is not None else '-'}")
    
    if "correlation_threshold" in result:
        console.print(f"  Correlation Threshold: {result['correlation_threshold']}")
        
    console.print(f"  Clusters Found: {result['clusters_found']}")
    console.print(f"  Silhouette Score: {result['score']:.4f}")


def evaluate_with_cluster_counts(method: str) -> list[dict[str, Any]]:
    """Evaluate a clustering method with different cluster count values."""
    results = []
    for cluster_count in CLUSTER_COUNT_RANGE:
        console.print(f"Evaluating {method} with cluster_count={cluster_count}...", style="dim")
        score, clusters_found = evaluate_clustering(method=method, n_clusters=cluster_count)
        
        if score > -1:
            results.append({
                "method": method, 
                "k": cluster_count, 
                "clusters_found": clusters_found, 
                "score": score
            })
    return results


def evaluate_with_correlation_thresholds(method: str, fixed_cluster_count: int) -> list[dict[str, Any]]:
    """Evaluate a clustering method with different correlation threshold values."""
    results = []
    
    for correlation_threshold in CORRELATION_THRESHOLDS:
        console.print(
            f"Evaluating {method} with cluster_count={fixed_cluster_count}, "
            f"correlation_threshold={correlation_threshold}...",
            style="dim"
        )
        
        score, clusters_found = evaluate_clustering(
            method=method,
            n_clusters=fixed_cluster_count,
            correlation_threshold=correlation_threshold
        )
        
        if score > -1:
            results.append({
                "method": method,
                "k": fixed_cluster_count,
                "correlation_threshold": correlation_threshold,
                "clusters_found": clusters_found,
                "score": score
            })
    
    return results


def main() -> None:
    """Evaluate clustering with different parameters and find optimal configuration."""
    args = parse_arguments()

    results = []
    console.print("[bold blue]Starting Clustering Evaluation...[/]", style="blue")
    
    for method in args.method:
        console.print(f"\n--- Method: [bold cyan]{method}[/] ---", style="cyan")
        
        if args.evaluate_cluster_counts:
            console.print("Evaluating with different cluster counts:", style="dim")
            results.extend(evaluate_with_cluster_counts(method))
        
        if args.evaluate_correlation_thresholds:
            console.print(f"Evaluating with different correlation thresholds (fixed k={args.fixed_cluster_count}):", style="dim")
            results.extend(evaluate_with_correlation_thresholds(method, args.fixed_cluster_count))

    console.print("\n--- [bold blue]Evaluation Summary[/] ---", style="blue")
    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        print_results_table(results)
        print_best_result(results[0])
    else:
        console.print("[bold red]No valid results obtained. Please check your data.[/]", style="red")


if __name__ == "__main__":
    main()
