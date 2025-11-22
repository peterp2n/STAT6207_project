import pandas as pd
import polars as pl
import polars.selectors as cs
from pathlib import Path


def standardize_columns(lf_input: pl.LazyFrame) -> pl.LazyFrame:
    return lf_input.with_columns(
        (cs.numeric() - cs.numeric().mean()) / cs.numeric().std()
    )


def add_age_bins(reading_age_series: pl.Series) -> tuple[pl.Series, pl.DataFrame]:
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
    df = pl.DataFrame({"reading_age": pl_categories})
    return pl_categories, df


if __name__ == "__main__":
    csv = Path("data") / "merged2.csv"
    merge2 = (
        pl.scan_csv(csv)
        .with_columns(
            pl.col("isbn").cast(pl.Utf8)
        )
    )

    # Group the reading_age into five bins
    unclean_ages = merge2.select("reading_age").collect()["reading_age"]
    _, pl_bins_reading_age_df = add_age_bins(unclean_ages)

    useful_cols = ('isbn', 'title', 'publisher', 'publication_date', 'book_format', "reading_age",
                   'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews',
                   'price', 'best_sellers_rank', 'customer_reviews')

    merge3 = (
        pl.concat(
            [merge2.drop("reading_age").collect(), pl_bins_reading_age_df],
            how="horizontal"
        ).select(useful_cols)
    )

    # merge3.write_csv(Path("data") / "merged3.csv")

    # Standardize numeric columns
    merge3_std = standardize_columns(merge3)
    # merge3_std.write_csv(Path("data") / "merged3_std.csv")

    # Standardized with dummies
    merge3_std_dummy = (
        merge3
        .select(useful_cols)
        .to_dummies(columns=["book_format", "publisher"])
    )

    # merge3_std_dummy.write_csv(Path("data") / "merged3_std_dummy.csv")

    # # Check associations
    # reading_age_by_publisher = (
    #     merge3
    #     .group_by('publisher')
    #     .agg(pl.col('reading_age').count().alias('ra_count'))
    # )
    # reading_age_by_format = merge3.group_by('book_format')['reading_age'].value_counts(normalize=True)
    #
    #
    # merge3.group_by(['publisher', 'book_format']).agg(
    #     pl.col('reading_age').mode().first()
    # ).filter(pl.col('reading_age').is_not_null())

    pass