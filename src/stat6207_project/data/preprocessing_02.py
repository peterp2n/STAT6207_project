import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from pathlib import Path

def load_df(csv_path: str|Path) -> pl.DataFrame:
    input = pl.scan_csv(csv_path,
        schema_overrides={
            "isbn": pl.Utf8,
            "title": pl.Utf8,
            "publisher": pl.Utf8,
            "publication_date": pl.Utf8,
            "book_format": pl.Utf8,
            "language": pl.Utf8,
            "isbn_10": pl.Utf8,
            "isbn_13": pl.Utf8,
            "asin": pl.Utf8,
            "description": pl.Utf8,
            "product_url": pl.Utf8,
            "scrape_status": pl.Enum(["success", "fail"]),
    }
    ).filter(pl.col("scrape_status") == "success")
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
    merge = load_df(csv)
    # Columns:['isbn', 'title', 'publisher', 'publication_date', 'series_name', 'book_format', 'language',
    # 'print_length', 'isbn_10', 'isbn_13', 'item_weight', 'length', 'width', 'height', 'rating',
    # 'number_of_reviews', 'availability', 'price', 'best_sellers_rank', 'customer_reviews', 'reading_age',
    # 'edition', 'author', 'asin', 'description', 'product_url', 'scrape_status']

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        for i in range(0, merge.collect_schema().len(), 5):
            print(f"\nColumns {i} to {min(i + 5, merge.collect_schema().len())}:")
            print(merge.select(cs.by_index(range(i, min(i + 5, merge.collect_schema().len())))).describe())

    merge = merge.drop([
        "asin", "edition", "series_name",
        "availability",  "product_url",
        "scrape_status", 'isbn_10', 'isbn_13'])
    merge2 = merge.select(
            ['isbn', 'title', 'publisher', 'publication_date',
             'book_format', "reading_age", 'language', 'print_length',
             'item_weight', 'length', 'width', 'height',
             'rating', 'number_of_reviews', 'price',
             'best_sellers_rank', 'customer_reviews', 'author',
             'description']
    ).with_columns(
            pl.coalesce([
                pl.col("publication_date").str.to_date(format="%Y-%m-%d", strict=False),
                pl.col("publication_date").str.to_date(format="%-d/%-m/%Y", strict=False),
                pl.col("publication_date").str.to_date(format="%d/%m/%Y", strict=False),
            ]).alias("publication_date"),
            pl.col("publisher").str.to_lowercase().str.replace_many(
                [r"\s?ltd\s?", r"[,\s]?inc.?\s?"],
                ["", ""]
            ),
            pl.col("book_format").str.to_lowercase().str.replace_all(r".*kindle.*", "kindle"),
            pl.col("language").str.to_lowercase(),
            pl.col("description")
            .str.replace("\xa0", " ", literal=True)  # literal replacement
            .str.replace(r"\s?Read more", "", literal=False)
    ).collect()
    # Merge2 columns: ['isbn', 'title', 'publisher', 'publication_date', 'book_format', 'reading_age', 'language',
    # 'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews', 'price',
    # 'best_sellers_rank', 'customer_reviews', 'author', 'description']

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        for i in range(0, merge2.collect_schema().len(), 5):
            print(f"\nColumns {i} to {min(i + 5, merge2.collect_schema().len())}:")
            print(merge2.select(cs.by_index(range(i, min(i + 5, merge2.collect_schema().len())))).describe())

    # merge2.write_csv(Path("data") / "merged2.csv", include_bom=True)

    pass