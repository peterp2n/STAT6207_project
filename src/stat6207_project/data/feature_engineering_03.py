import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from pathlib import Path


def standardize_columns(lf_input: pl.LazyFrame) -> pl.LazyFrame:
    numeric_cols = lf_input.select(pl.col(pl.NUMERIC_DTYPES)).columns
    return lf_input.with_columns(
        [((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col) for col in numeric_cols]
    )



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

def add_age_bins(reading_age_series: pl.Series) -> pl.Series:
    bins = [0, 2, 5, 7, 10, 1000]
    labels = ["baby", "toddler", "preschool", "preadolescence", "others"]

    pd_reading_age = reading_age_series.to_pandas()
    pd_categories = pd.cut(
        pd_reading_age,
        bins=bins,
        labels=labels,
        right=False,       # intervals are closed on the left, open on the right
        include_lowest=True
    )
    pl_categories = pl.from_pandas(pd_categories).cast(pl.Utf8)
    return pl_categories


if __name__ == "__main__":
    csv = Path("data") / "merged2.csv"
    merge2 = (
        pl.scan_csv(csv)
        .with_columns(
            pl.col("isbn").cast(pl.Utf8)
        )
    )



    # Group the reading_age into five bins
    # pd_reading_age = merge2.select("reading_age").collect()["reading_age"].to_pandas()
    # bins_reading_age = pd.cut(x=pd_reading_age, bins=[0, 2, 5, 7, 10, 20], labels=["baby", "toddler", "preschool", "preadolescence", "others"])
    # pl_bins_reading_age = pl.from_pandas(bins_reading_age).cast(pl.Utf8)
    unclean_ages = merge2.select("reading_age").collect()["reading_age"]
    pl_bins_reading_age = add_age_bins(unclean_ages)
    pl_bins_reading_age_df = pl.DataFrame({"reading_age": pl_bins_reading_age})
    merge2 = pl.concat(
        [merge2.drop("reading_age").collect(), pl_bins_reading_age_df],
        how="horizontal"
    )

    merge3 = merge2
    # merge3 = standardize_columns(merge2)

    useful_cols = ('isbn', 'title', 'publisher', 'publication_date', 'book_format', "reading_age",
                   'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews',
                   'price', 'best_sellers_rank', 'customer_reviews', 'description')
    merge3, _ = clean_publishers(merge3)


    merge3 = (
        merge3
        .select(useful_cols).collect()
        .to_dummies(columns=["book_format", "publisher"])
    )



    # age_groups = (
    #     merge3
    #     .select("reading_age")
    #     .filter(pl.col("reading_age").is_not_null())
    #     .collect()
    #     .to_series()
    #     .to_list()
    # )
    # plt.hist(age_groups, bins=30)
    # plt.show()

    # merge3.write_csv(Path("data") / "merged3.csv")

    # arguments = {
    #     "db_path": Path("data") / "Topic1_dataset.sqlite",
    #     "headless": False,  # Set to True for headless mode
    #     "json_folder": Path("data") / "scrapes",
    # }

    # data = Data(arguments)
    # data.load_all_tables()
    # purchase = data.table_holder.get("purchase").collect()
    # products = data.table_holder.get("products").collect()
    # sales = data.table_holder.get("sales").collect()
    # shops = data.table_holder.get("shops").collect()

    pass