import polars as pl
import polars.selectors as cs
from pathlib import Path

def load_df(csv_path: str|Path) -> pl.DataFrame:
    input = pl.scan_csv(csv_path, schema_overrides={
        "isbn": pl.Utf8,
        "isbn_10": pl.Utf8,
        "isbn_13": pl.Utf8,
        "best_sellers_rank": pl.Int32
    }).filter(pl.col("scrape_status") == "success")
    return input

def standardize_columns(lazyframe: pl.LazyFrame) -> pl.LazyFrame:
    """
    Standardize all numeric columns using z-score normalization.

    Parameters
    ----------
    lazyframe : pl.LazyFrame
        Input LazyFrame to standardize

    Returns
    -------
    pl.LazyFrame
        LazyFrame with standardized numeric columns
    """
    df_copy = lazyframe.clone()

    # Standardize all numeric columns: (x - mean) / std
    df_standardized = df_copy.with_columns(
        ((cs.numeric() - cs.numeric().mean()) / cs.numeric().std())
    )

    return df_standardized

if __name__ == "__main__":
    csv = Path("data") / "merged.csv"
    df = load_df(csv)
    df = df.with_columns(
        pl.col("publication_date").str.to_date(format="%Y-%m-%d", strict=False)
    )

    numerics = df.select(cs.numeric()).collect()
    numeric_df = standardize_columns(numerics)
    pass