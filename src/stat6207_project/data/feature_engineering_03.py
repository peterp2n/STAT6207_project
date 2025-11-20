import polars as pl
import polars.selectors as cs
from pathlib import Path


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
    csv = Path("data") / "merged2.csv"
    merge2 = (
        pl.scan_csv(csv)
        .with_columns(
            pl.col("isbn").cast(pl.Utf8)
        )

    ).collect()

    publisher_counts = (
        merge2
        .group_by("publisher")
        .agg(pl.col("publisher").count().alias("count"))
        .filter(pl.col("count") >= 10)
        .select("publisher")
    )

    merge3 = (
        merge2
        .with_columns(
            pl.when()
            .alias("publisher")
        )
    )
    pass