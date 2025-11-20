import polars as pl
from pathlib import Path
from src.stat6207_project.scraper.read_html import Extractor


if __name__ == "__main__":

    ext = Extractor()
    html_folder = Path("data")

    # All HTML files recursively
    all_paths = list(html_folder.rglob("*.html"))

    # Keep only files where the filename (without .html) is exactly 13 characters
    # OR starts with "978" (your original logic)
    html_paths = [
        p for p in all_paths
        if len(p.stem) == 13 or p.stem.startswith("product_978")
    ]

    print(f"Processing {len(html_paths)} files...")

    dup_results = []
    for html_path in html_paths:
        content = Extractor.read_html(html_path)
        if content:
            result = ext.parse(content)
            dup_results.append(result)

    df = ext.to_dataframe()
    df.write_csv(html_folder / "amazon_cleaned.csv")

    html_folder = Path("data")
    df = pl.scan_csv(html_folder / "amazon_cleaned.csv", schema_overrides={
        "isbn": pl.Utf8,
        "isbn_10": pl.Utf8,
        "isbn_13": pl.Utf8,
        "best_sellers_rank": pl.Int32
    }).collect()
    api = (pl.scan_csv(html_folder / "books_api_cleaned.csv", schema_overrides={"isbn": pl.Utf8})
           .select(["page_count", "isbn"])).collect()
    merged = df.join(api, left_on='isbn_13', right_on='isbn', how='left', suffix='_api').with_columns(
        # Use original print_length, if present; else use page_count from API
        print_length=
        pl.when(pl.col("print_length").is_not_null()).then(pl.col("print_length"))
        .when(pl.col("page_count").is_not_null() & (pl.col("page_count") != 0)).then(pl.col("page_count"))
        .otherwise(None)
        # Drop the page_count column after merging
    ).drop("page_count")

    merged.write_csv(html_folder / "merged.csv")
    print(df)
    pass