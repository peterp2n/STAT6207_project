import polars as pl
import polars.selectors as cs
from pathlib import Path

def impute_book_format(df_input: pl.DataFrame) -> pl.DataFrame:

    empty_reading_age = df_input.select("reading_age")["reading_age"].is_null()

    book_formats = (
        df_input
        .group_by("publisher", maintain_order=True)
        .agg(
            pl.col("book_format")
            .drop_nulls()
            .mode()
            .first()
            .alias("impute_book_format")
        )
    )
    joined = (
        df_input
        .join(book_formats, on="publisher", how="left")
        .with_columns(
            pl.when(pl.col("book_format").is_null() & empty_reading_age)
            .then(pl.col("impute_book_format"))
            .otherwise(pl.col("book_format"))
        )
        .with_columns(
            pl.when(pl.col("publisher").str.contains("cornerstone"))
            .then(pl.lit("paperback"))
            .when(pl.col("publisher").str.contains("random_house"))
            .then(pl.lit("paperback"))
            .otherwise(pl.col("book_format"))
            .alias("book_format")
        )
        .drop("impute_book_format")
    )
    return joined

def impute_reading_age(df_input: pl.DataFrame) -> pl.DataFrame:
    TITLE_KEYWORDS = {
        "toddler": [
            "peppa pig", "busy book", "my busy books", "lift-the-flap", "lift the flap",
            "that's not my", "touch and feel", "finger puppet", "my first library",
            "touchy-feely", "usborne lift", "usborne flap", "baby einstein", "board book",
            "thomas", "frozen"
        ],
        "baby": ["baby einstein", "my first library", "touch and feel", "baby's first"],
        "preschool": [
            "mr. men", "little miss", "barney", "disney princess", "sofia the first",
            "mickey mouse clubhouse", "pj masks", "step into reading", "kumon", "brain quest"
        ],
        "preadolescence": [
            "dog man", "captain underpants", "diary of a wimpy kid", "geronimo stilton",
            "thea stilton", "treehouse", "big nate", "guinness world records"
        ],
    }

    PUBLISHER_KEYWORDS = {
        "toddler": ["phidal", "make_believe_ideas", "ladybird", "usborne", "parragon", "hinkler"],
        "preschool": ["kumon"],
        "preadolescence": ["graphix", "scholastic"],
    }

    # Get the mode reading_age for each book_format
    # merge3.filter(~pl.col("reading_age").is_null()).group_by("book_format").agg(pl.col("reading_age").mode())

    empty_reading_age = df_input.select("reading_age")["reading_age"].is_null()
    imputed = (
        df_input
        .with_columns(
            pl.col("title").str.to_lowercase().alias("title_lower")
        )
        # Impute reading_age based on title keywords
        .with_columns(
            pl.when(pl.col("title_lower").str.contains("|".join(TITLE_KEYWORDS.get("toddler"))) & empty_reading_age)
            .then(pl.lit("toddler"))
            .when(pl.col("title_lower").str.contains("|".join(TITLE_KEYWORDS.get("baby"))) & empty_reading_age)
            .then(pl.lit("baby"))
            .when(pl.col("title_lower").str.contains("|".join(TITLE_KEYWORDS.get("preschool"))) & empty_reading_age)
            .then(pl.lit("preschool"))
            .when(pl.col("title_lower").str.contains("|".join(TITLE_KEYWORDS.get("preadolescence"))) & empty_reading_age)
            .then(pl.lit("preadolescence"))
            .otherwise(pl.col("reading_age"))
            .alias("reading_age")
        )
        # Impute reading_age based on book format
        .with_columns(
            pl.when(pl.col("book_format").str.contains("board book|hardcover") & empty_reading_age)
            .then(pl.lit("toddler"))
            .when(pl.col("book_format").str.contains("paperback|cards|library binding") & empty_reading_age)
            .then(pl.lit("preschool"))
            .otherwise(pl.col("reading_age"))
            .alias("reading_age")
        )
        # Impute reading_age based on publishers
        .with_columns(
            pl.when(pl.col("publisher").is_in(PUBLISHER_KEYWORDS.get("toddler")) & empty_reading_age)
            .then(pl.lit("toddler"))
            .when(pl.col("publisher").is_in(PUBLISHER_KEYWORDS.get("baby")) & empty_reading_age)
            .then(pl.lit("baby"))
            .when(pl.col("publisher").is_in(PUBLISHER_KEYWORDS.get("preschool")) & empty_reading_age)
            .then(pl.lit("preschool"))
            .when(pl.col("publisher").is_in(PUBLISHER_KEYWORDS.get("preadolescence")) & empty_reading_age)
            .then(pl.lit("preadolescence"))
            .otherwise(pl.col("reading_age"))
            .alias("reading_age")
        )
        .drop("title_lower")
    )

    return imputed

def standardize_columns(lf_input: pl.LazyFrame) -> pl.LazyFrame:
    return lf_input.with_columns(
        (cs.numeric() - cs.numeric().mean()) / cs.numeric().std()
    )

if __name__ == "__main__":

    useful_cols = ('isbn', 'title', 'publisher', 'publication_date', 'book_format', "reading_age",
                   'print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'number_of_reviews',
                   'price', 'best_sellers_rank', 'customer_reviews')

    merge3 = (
        pl.read_csv(Path("data") / "merged2.csv")
        .select(useful_cols)
        .with_columns(
            pl.col("isbn").cast(pl.Utf8)
        )
    )

    merge3 = impute_book_format(merge3)
    merge3 = impute_reading_age(merge3)
    # merge3.write_csv(Path("data") / "merged3.csv")

    sales_data = (
        pl.scan_csv(
            "expand_quarterly_sales_retail.csv",
            schema_overrides={
                "barcode2": pl.Utf8,
                "Quarter_num": pl.Enum(["1", "2", "3", "4"]),
            }
        )
        .collect()
    )

    merge3_sales = (
        merge3
        .join(
            sales_data,
            left_on="isbn",
            right_on="barcode2",
            how="left"
        )
    )

    # merge3_sales.write_csv(Path("data") / "merged3_sales.csv", include_bom=True)