import polars as pl
from pathlib import Path


if __name__ == "__main__":
    html_folder = Path("data")
    df = pl.scan_csv(html_folder / "amazon_cleaned.csv")
    api = (pl.scan_csv(html_folder / "books_api_cleaned.csv", schema_overrides={"isbn": pl.Utf8})
           .select(["page_count", "isbn"]).collect())
    merged = df.join(api, left_on='isbn_13', right_on='isbn', how='left', suffix='_api').with_columns(
        # Use original print_length, if present; else use page_count from API
        print_length=
        pl.when(pl.col("print_length").is_not_null()).then(pl.col("print_length"))
        .when(pl.col("page_count").is_not_null() & (pl.col("page_count") != 0)).then(pl.col("page_count"))
        .otherwise(None)
        # Drop the page_count column after merging
    ).drop("page_count")

    # merged.write_csv(html_folder / "merged.csv")


    pass