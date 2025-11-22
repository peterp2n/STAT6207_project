import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def clean_publishers(lf_input: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
    """
    Clean and normalize publisher names, adding book counts.

    Args:
        lf_input: LazyFrame containing a 'publisher' column

    Returns:
        Tuple of (cleaned LazyFrame with book_count, unique publishers as list)
    """
    # Define publisher normalization mapping
    publisher_mappings = {
        "egmont": "egmont",
        "parragon": "parragon",
        "paragon": "parragon",
        "scholastic": "scholastic",
        "ladybird": "ladybird",
        "random house": "random_house",
        "kumon": "kumon",
        "igloo": "igloo",
        "hinkler": "hinkler",
        "alligator": "alligator",
        "autumn": "autumn",
        "rh": "penguin_random_house",
        "creative teaching": "creative_teaching",
        "edc pub": "edc_publishing",
        "national geographic": "national_geographic",
        "makebelieveideas": "make_believe_ideas",
        "make believe": "make_believe_ideas",
        "macmillan": "macmillan_childrens",
        "harper": "harper_collins",
        "henry holt": "henry_holt",
        "guinness": "guinness_world_records",
        "five mile": "five_mile_press",
        "penguin": "penguin_random_house",
        "phidal": "phidal",
        "pi kids": "phoenix_international",
        "pi_kids": "phoenix_international",
        "phoenix": "phoenix_international",
        "publications international": "publications_international",
        "puffin": "puffin",
        "qed": "qed_publishing",
        "shree book": "shree_book_centre",
        "simon spotlight": "simon_spotlight",
        "turtleback": "turtleback",
        "usborne": "usborne",
        "workman": "workman",
    }

    # Build normalization expression dynamically
    normalization_expr = pl.col("publisher")
    for pattern, normalized_name in publisher_mappings.items():
        normalization_expr = pl.when(
            pl.col("publisher").str.contains(pattern)
        ).then(pl.lit(normalized_name)).otherwise(normalization_expr)

    # Apply normalization
    cleaned = lf_input.with_columns(
        normalization_expr.alias("publisher")
    )

    # Calculate publisher counts and join back
    publisher_counts = (
        cleaned
        .group_by("publisher")
        .agg(pl.col("publisher").count().alias("book_count"))
        .sort("book_count", descending=True)
            .filter(pl.col("book_count") >= 5)
    )

    cleaned = cleaned.join(publisher_counts, on="publisher", how="inner")

    # Get unique publishers as a Python list
    unique_publishers = (
        cleaned
        .select("publisher")
        .unique()
        .sort("publisher")
        ["publisher"]
    )

    return cleaned, unique_publishers

def add_age_bins(reading_age_series: pl.Series) -> tuple[pl.Series, pl.DataFrame]:
    bins = [0, 2, 5, 7, 10, 1000]
    labels = ["baby", "toddler", "preschool", "preadolescence", "adolescence or above"]

    pd_reading_age = reading_age_series.to_pandas()
    pd_categories = pd.cut(
        pd_reading_age,
        bins=bins,
        labels=labels,
        right=False,       # intervals are closed on the left, open on the right
        include_lowest=True
    )
    pl_categories = pl.from_pandas(pd_categories).cast(pl.Utf8)
    df = pl.DataFrame({"reading_age": pl_categories})
    return pl_categories, df

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
    merge2 = (merge.select(
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
            pl.col("publisher").str.to_lowercase().str.replace_all(" ", "_"),
            pl.col("author").str.to_lowercase(),
            pl.col("book_format").str.to_lowercase()
            .str.replace_all(r".*kindle.*", "kindle")
            .str.replace_all(r".*card.*", "cards")
            .str.replace_all(r".*paperback.*", "paperback")
            .str.replace_all(" ", "_"),
            pl.col("language").str.to_lowercase(),
            pl.col("description")
            .str.replace("\xa0", " ", literal=True)  # literal replacement
            .str.replace(r"\s?Read more", "", literal=False)
    )
    .with_columns(
        pl.when(pl.col("description").str.len_chars() < 100)
        .then(pl.lit(""))
        .otherwise(pl.col("description"))
        .alias("description"),
        pl.when(pl.col("book_format").is_in(["board_book", "cards", "paperback", "hardcover", "library_binding"]))
        .then(pl.col("book_format"))
        .otherwise(pl.lit(None))
        .alias("book_format")
    )
    ).collect()

    merge2, unique_pubs = clean_publishers(merge2)

    # Group the reading_age into five bins
    unclean_ages = merge2.select("reading_age")["reading_age"]
    _, pl_bins_reading_age_df = add_age_bins(unclean_ages)

    useful_cols = ('isbn', 'title', 'publisher', 'publication_date', 'book_format', "reading_age",
                   'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews',
                   'price', 'best_sellers_rank', 'customer_reviews')

    merge2 = (
        pl.concat(
            [merge2.drop("reading_age"), pl_bins_reading_age_df],
            how="horizontal"
        ).select(useful_cols)
    )

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        for i in range(0, merge2.collect_schema().len(), 5):
            print(f"\nColumns {i} to {min(i + 5, merge2.collect_schema().len())}:")
            print(merge2.select(cs.by_index(range(i, min(i + 5, merge2.collect_schema().len())))).describe())

    # merge2.write_csv(Path("data") / "merged2.csv", include_bom=True)

    pass