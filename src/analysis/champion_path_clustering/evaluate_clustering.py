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
CLUSTER_COUNT_RANGE = range(5, 30)


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
    
    return parser.parse_args()


def evaluate_clustering(
    method: str, 
    n_clusters: int | None = None
) -> tuple[float, int]:
    """
    Evaluate a clustering configuration and return its silhouette score.
    
    Args:
        method: Clustering method to use
        n_clusters: Number of clusters for hierarchical/spectral methods
        
    Returns:
        Tuple of (silhouette_score, num_clusters)
    """
    clustering_strategy = ClusteringStrategyFactory.create_strategy(method)
    clusterer = ChampionPathClusterer(
        clustering_strategy=clustering_strategy
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
    table.add_column("Clusters", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for i, res in enumerate(results):
        score_str = f"{res['score']:.4f}"
        style = "green" if i == 0 else ""
        
        table.add_row(
            res["method"],
            str(res["k"] if res["k"] is not None else "-"),
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


def main() -> None:
    """Evaluate clustering with different parameters and find optimal configuration."""
    args = parse_arguments()

    results = []
    console.print("[bold blue]Starting Clustering Evaluation...[/]", style="blue")
    
    for method in args.method:
        console.print(f"\n--- Method: [bold cyan]{method}[/] ---", style="cyan")
        results.extend(evaluate_with_cluster_counts(method))

    console.print("\n--- [bold blue]Evaluation Summary[/] ---", style="blue")
    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        print_results_table(results)
        print_best_result(results[0])
    else:
        console.print("[bold red]No valid results obtained. Please check your data.[/]", style="red")


if __name__ == "__main__":
    main()
