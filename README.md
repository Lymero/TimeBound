# LoL Winrate vs. Game Length Analysis Tool

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Python tool for analyzing champion winrate trends over game length in League of Legends, identifying patterns and correlations between different champions.

## Overview

This tool extracts champion winrate vs. game length data from LoLalytics, processes the SVG path data into point coordinates, analyzes correlations between these game-length-dependent winrate trends, and generates visualizations of the results. The analysis helps identify which champions perform better in early, mid, or late game scenarios, and which champions have similar performance patterns.

## Features

- **Data Acquisition**: 
  - Extract SVG path data from LoLalytics champion pages using Playwright
  - Collect champion metadata including names, roles, and win rates
- **Data Processing**: 
  - Parse SVG path data into point coordinates representing game time vs. winrate
  - Pre-process data for correlation analysis and clustering
- **Analysis**: 
  - Multiple clustering strategies (Hierarchical, Threshold-based)
  - Calculate correlations between champion winrate trends
  - Group champions into clusters based on similar winrate patterns
  - Find champions with similar performance patterns to a target champion
- **Visualization**: 
  - Generate static and interactive cluster visualizations
  - Create correlation network graphs
  - Plot winrate curves for individual champions and cluster profiles
  - Compare multiple champions' performance patterns

## Project Structure

```
winrates/
├── data/                      # Data files
│   ├── champion_svg_paths.json  # SVG paths for champion win rate curves
│   ├── champions.json         # Champion metadata
│   └── outputs                # Generated visualizations and analysis results
├── src/                       # Source code
│   ├── acquisition/           # Data acquisition modules
│   │   └── svg_extractor.py   # SVG path extraction from LoLalytics
│   ├── processing/            # Data processing modules
│   │   └── svg_parser.py      # SVG path parsing utilities
│   ├── analysis/              # Analysis modules
│   │   ├── champion_path_clustering/  # Champion clustering 
│   │   │   ├── clusterer.py           # Core clustering implementation
│   │   │   ├── factory.py             # Strategy factory pattern
│   │   │   ├── strategies/            # Different clustering algorithms
│   │   │   └── types.py               # Type definitions
│   │   └── graph_correlation.py       # Correlation utilities
│   ├── visualization/         # Visualization modules
│   │   └── cluster_visualizer.py      # Visualization for results
│   ├── utils/                 # Utility functions
│   │   └── logger.py          # Logging utilities
│   ├── examples/              # Example scripts
│   └── main.py                # Main entry point
├── tests/                     # Test files
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration and dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/winrates.git
   cd winrates
   ```

2. This project uses [Rye](https://github.com/astral-sh/rye) for dependency management. Install Rye if you don't have it:
   ```bash
   curl -sSf https://rye-up.com/get | bash
   ```

3. Set up the project with Rye:
   ```bash
   rye sync
   ```

4. Install Playwright browsers:
   ```bash
   rye run install-playwright
   ```

## Usage

### Data Acquisition

To scrape champion data from LoLalytics:

```bash
# Scrape all champions
rye run dev acquire --all

# Scrape specific champions
rye run dev acquire --champions Aatrox Ahri Akali

# Customize patch, tier, or region
rye run dev acquire --all --patch 30 --tier platinum_plus --region na
```

### Clustering Analysis

To analyze champion clusters using different strategies:

```bash
# Using threshold-based clustering (default)
rye run dev cluster --clustering-method threshold

# Using hierarchical clustering
rye run dev cluster --clustering-method hierarchical

# Customize clustering parameters
rye run dev cluster --clustering-method threshold --correlation-threshold 0.85 --interactive
```

### Finding Similar Champions

To find champions with similar win rate patterns:

```bash
rye run dev cluster --champion Aatrox
```

### Visualization

The tool automatically generates visualizations during analysis. Output files are saved to the `data/outputs/` directory and include:
- Static cluster visualizations (PNG)
- Interactive cluster visualizations (HTML, when using the --interactive flag)
- Correlation network graphs
- Individual cluster profile visualizations

## Development

The project includes several useful commands configured in the `pyproject.toml` file:

- Run tests: `rye run test`
- Run tests with coverage: `rye run test-cov`
- Check code quality: `rye run lint`
- Format code: `rye run format`
- Type checking: `rye run typecheck`
- Run the application: `rye run dev`

## Requirements

Python 3.12 or higher is required. Key dependencies include:
- numpy, scipy, pandas for data processing
- matplotlib, seaborn, plotly for visualization
- networkx for graph analysis
- scikit-learn for clustering algorithms
- playwright for web automation
- rich for console output

## License

This project is licensed under the MIT License. 