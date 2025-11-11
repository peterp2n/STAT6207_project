"""Main entry point for TorScraper v1"""

from pathlib import Path
import time
import traceback

# Absolute imports work when main.py is at project root
from src.stat6207_project.scraper.scraper_v1 import TorScraper


def main():
    """Run the TorScraper with specified arguments"""
    arguments = {
        "db_path": Path("data") / "Topic1_dataset.sqlite",
        "headless": False,  # Set to True for headless mode
        "json_folder": Path("data") / "scrapes_v1",
    }

    scraper = TorScraper(arguments)
    scraper.load_all_tables()

    # Load queries from database
    que = (scraper.table_holder
           .get("products")
           .select("barcode2")
           .unique(maintain_order=True)
           .collect()
           .to_series()
           .to_list())

    scraper.load_queries(queries=que[:5])

    try:
        # Create browser
        if not scraper.create_driver():
            print("Failed to create browser")
            return

        # Handle Tor connection (only once at the start)
        scraper.handle_tor_connection("initial")

        # Scrape all queries - results are saved incrementally
        start = time.time()
        results = scraper.scrape_all()
        elapsed = time.time() - start

        print(f"\nTotal execution time: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user - partial results have been saved")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()
