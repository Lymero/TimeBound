"""
Extract champion winrate vs game length SVG data using Playwright.

This module uses Playwright to navigate to LoLalytics champion pages,
wait for the SVG graph to render, and extract the path data for analysis.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright

DEFAULT_PATCH = "30"
DEFAULT_TIER = "all"
DEFAULT_REGION = "all"
DEFAULT_OUTPUT = "data/champion_svg_paths.json"
SCROLL_WAIT_TIME = 500  # milliseconds
PAGE_LOAD_WAIT = 1500  # milliseconds
VIEWPORT_SIZE = {"width": 1920, "height": 1080}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


async def extract_svg_path(page: Page) -> str | None:
    """Extract the SVG path data for winrate by game length from the current page."""
    win_rate_text = await page.query_selector(
        "text:has-text('Win Rate vs Game Length')"
    )

    if not win_rate_text:
        for scroll_position in range(500, 5000, 500):
            await page.evaluate(f"window.scrollTo(0, {scroll_position})")
            await page.wait_for_timeout(SCROLL_WAIT_TIME)
            win_rate_text = await page.query_selector(
                "text:has-text('Win Rate vs Game Length')"
            )
            if win_rate_text:
                break

    if win_rate_text:
        await page.evaluate("window.scrollBy(0, 200)")
        await page.wait_for_timeout(SCROLL_WAIT_TIME)
    else:
        print("Could not find 'Win Rate vs Game Length' text on the page")

    try:
        path_data = await page.evaluate(
            """
            () => {
                const path = document.querySelector('path._line_wrlchart_kz3qr_1');
                return path ? path.getAttribute('d') : null;
            }
        """
        )
        return path_data
    except Exception as e:
        print(f"Error extracting SVG path: {e}")
        return None


async def extract_champion_data(
    champion: str,
    patch: str = DEFAULT_PATCH,
    tier: str = DEFAULT_TIER,
    region: str = DEFAULT_REGION,
) -> dict[str, Any]:
    """Extract data for a champion using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT_SIZE, user_agent=USER_AGENT
        )

        page = await context.new_page()
        url = f"https://lolalytics.com/lol/{champion}/build/?patch={patch}&tier={tier}&region={region}"

        result = {
            "champion": champion,
            "patch": patch,
            "tier": tier,
            "region": region,
            "url": url,
        }

        try:
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(PAGE_LOAD_WAIT)
            svg_path = await extract_svg_path(page)
            result["svg_path"] = svg_path

            if svg_path:
                print(f"✓ Successfully extracted SVG path for {champion}")
            else:
                print(f"✗ Failed to extract SVG path for {champion}")
        except Exception as e:
            result["error"] = str(e)
            print(f"✗ Error processing {champion}: {e}")
        finally:
            await browser.close()
            
        return result


async def load_existing_results(output_path: Path) -> dict[str, Any]:
    """Load existing results from file if available."""
    results = {}
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results")
        except json.JSONDecodeError:
            print("Warning: Could not parse existing results file, starting fresh")
    return results


async def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


async def extract_champions(
    champions: list[str],
    concurrency: int,
    patch: str,
    tier: str,
    region: str,
    output_file: str,
) -> dict[str, Any]:
    """Extract data for multiple champions concurrently."""
    print(f"Using concurrency: {concurrency}")
    output_path = Path(output_file)
    results = await load_existing_results(output_path)

    champions_to_process = [
        c for c in champions if c not in results or not results[c].get("svg_path")
    ]
    if not champions_to_process:
        print("All champions already processed. Nothing to do.")
        return results

    print(f"Processing {len(champions_to_process)} champions")

    semaphore = asyncio.Semaphore(concurrency)

    async def extract_champion_with_rate_limit(champion: str) -> dict[str, Any]:
        async with semaphore:
            return await extract_champion_data(champion, patch, tier, region)

    tasks = [
        extract_champion_with_rate_limit(champion) for champion in champions_to_process
    ]
    champion_results = await asyncio.gather(*tasks)

    for i, champion in enumerate(champions_to_process):
        results[champion] = champion_results[i]

    await save_results(results, output_path)
    print(f"Extraction complete. Results saved to {output_file}")
    return results


def get_all_champion_names() -> list[str]:
    """Get a list of all champion names from the champions data file."""
    champions_file = Path("data/champions.json")

    try:
        with open(champions_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading champions file: {e}")
        return []


async def main_async() -> None:
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Extract champion winrate vs game length SVG data"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--champions", nargs="+", help='Champion names (e.g., "kogmaw nasus")'
    )
    group.add_argument(
        "--all", action="store_true", help="Extract data for all champions"
    )

    parser.add_argument(
        "--patch",
        default=DEFAULT_PATCH,
        help=f'Game patch (default: "{DEFAULT_PATCH}")',
    )
    parser.add_argument(
        "--tier", default=DEFAULT_TIER, help=f'Rank tier (default: "{DEFAULT_TIER}")'
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f'Game region (default: "{DEFAULT_REGION}")',
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f'Output file path (default: "{DEFAULT_OUTPUT}")',
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        choices=range(1, 21),
        default=1,
        help="Number of concurrent extractions (default: 1)",
    )

    args = parser.parse_args()

    if args.all:
        champions = get_all_champion_names()
    elif args.champions:
        champions = args.champions
    else:
        champions = ["kogmaw"]

    await extract_champions(
        champions=champions,
        concurrency=args.concurrency,
        patch=args.patch,
        tier=args.tier,
        region=args.region,
        output_file=args.output,
    )


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
        print("Extraction completed successfully")
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
