import polars as pl
import polars.selectors as cs
from pathlib import Path


def standardize_columns(lf_input: pl.LazyFrame) -> pl.LazyFrame:
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
    df_copy = lf_input.clone()

    # Standardize all numeric columns: (x - mean) / std
    df_standardized = df_copy.with_columns(
        ((cs.numeric() - cs.numeric().mean()) / cs.numeric().std())
    )

    return df_standardized


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
        "disney": "disney",
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
    cleaned = lf_input.with_columns(normalization_expr.alias("publisher"))

    # Calculate publisher counts and join back
    publisher_counts = (
        cleaned
        .group_by("publisher")
        .agg(pl.col("publisher").count().alias("book_count"))
        .sort("book_count", descending=True)
        .filter(pl.col("book_count") >= 5)
    )

    cleaned = lf_input.join(publisher_counts, on="publisher", how="inner")

    # Get unique publishers as a Python list
    unique_publishers = (
        cleaned
        .select("publisher")
        .unique()
        .sort("publisher")
        .collect()
        .get_column("publisher")
    )

    return cleaned, unique_publishers


if __name__ == "__main__":
    csv = Path("data") / "merged2.csv"
    merge2 = (
        pl.scan_csv(csv)
        .with_columns(
            pl.col("isbn").cast(pl.Utf8)
        )
    )

    merge3 = standardize_columns(merge2)

    useful_cols = ('isbn', 'title', 'publisher', 'publication_date', 'book_format', "reading_age",
                   'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews',
                   'price', 'best_sellers_rank', 'customer_reviews', 'author', 'description')
    cleaned_publishers, _ = clean_publishers(merge3)


    merge3 = (
        merge3
        .select(useful_cols).collect()
        .to_dummies(columns=["book_format", "publisher"])
    )


    pass