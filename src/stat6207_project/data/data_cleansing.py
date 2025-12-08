import sqlite3
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Setup paths
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)
    db_path = data_folder / "Topic1_dataset.sqlite"

    # Establish connection once (Polars can read directly from SQLite)
    with sqlite3.connect(db_path) as conn:
        sales = pl.read_database("SELECT * FROM sales", connection=conn)
        products = pl.read_database("SELECT * FROM products", connection=conn)

    sales = (
       sales
        .rename({"product": "isbn", "price": "selling_price"})
        .with_columns([
           pl.col("isbn").cast(pl.Utf8),
           pl.col("selling_price").cast(pl.Utf8).str.replace("$", "").cast(pl.Float32).alias("selling_price")
       ])
        .with_columns([
           pl.col("trandate")
           .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f").dt.date()
           .alias("trandate"),
           ((pl.col("selling_price") * pl.col("quantity") - pl.col("amount"))
            / (pl.col("quantity") * pl.col("selling_price")) * 100)
           .round(0)
           .clip(lower_bound=0.0)
           .alias("discount_rate"),
           pl.col("channel").str.to_lowercase().alias("channel")
       ])
        .filter(pl.col("isbn").str.starts_with("978"))
    )

    # products = (
    #     products
    #     .drop(["isbn", "barcode", "product"])
    #     .rename({"barcode2": "isbn", "price": "marked_price"})
    #     .with_columns([
    #         pl.col("isbn").cast(pl.Utf8),
    #         pl.col("book_original_price").cast(pl.Utf8),
    #     ])
    #     .with_columns([
    #         pl.col("book_original_price").str.replace("$", "").alias("book_original_price")
    #     ])
    #     .filter(pl.col("isbn").str.starts_with("978"))
    # )

    # dog_mans = products["title"].str.to_lowercase().str.contains("dog man")
    # cat_kid = products["title"].str.to_lowercase().str.contains("cat kid")
    # captain = products["title"].str.to_lowercase().str.contains("captain underpants")
    # andy = products["title"].str.to_lowercase().str.contains("andy griffiths")
    # guinness = products["title"].str.to_lowercase().str.contains("guinness")
    #
    # target_products = products.filter(
    #     pl.any_horizontal(
    #         dog_mans,
    #         cat_kid,
    #         captain,
    #         andy,
    #         guinness
    #     )
    # )
    # target_products.write_csv("target_products.csv", include_bom=True)

    target_products = pl.read_csv("target_products.csv", schema_overrides={"isbn": pl.Utf8})

    all_cols = [
        "isbn",
        "title",
        "format",
        "marked_price",
        "channel",
        "location",
        "quantity",
        "selling_price",
        "discount",
        "disc_amt",
        "amount",
        "discount_rate",
        "invoice",
        "trandate",
        "book_original_price",
        "clients"
    ]

    cols_use = [
        'isbn',
        "title",
        'format',
        "trandate",
        'channel',
        'discount_rate',
        'quantity'
    ]


    # ['title', 'marked_price', 'book_original_price', 'isbn', 'format', 'channel', 'location', 'invoice', 'clients', 'quantity', 'trandate', 'selling_price', 'discount', 'disc_amt', 'amount', 'discount_rate']
    # Perform the inner join on isbn
    join = target_products.join(sales, on="isbn", how="inner")

    # Example filter: titles containing "Dog Man" (case-sensitive by default)
    # Adjust the pattern as needed; empty string "" matches everything
    dog_man_mask = join["title"].str.to_lowercase().str.contains("dog man")
    cat_kid_mask = join["title"].str.to_lowercase().str.contains("cat kid")
    captain_mask = join["title"].str.to_lowercase().str.contains("captain underpants")
    andy_mask = join["title"].str.to_lowercase().str.contains("andy griffiths")
    guinness_mask = join["title"].str.to_lowercase().str.contains("guinness")

    target_mask = pl.any_horizontal(
        dog_man_mask,
        cat_kid_mask,
        captain_mask,
        andy_mask,
        guinness_mask
    )

    join = (
        join
        .select(cols_use)
        .with_columns([
            pl.when(pl.col("trandate").dt.quarter() == 1).then(pl.lit("1"))
            .when(pl.col("trandate").dt.quarter() == 2).then(pl.lit("2"))
            .when(pl.col("trandate").dt.quarter() == 3).then(pl.lit("3"))
            .otherwise(pl.lit("4")).alias("q_num")
        ])
    )

    first_trandate_by_isbn = join.group_by("isbn").agg([
        pl.col("trandate").min().alias("first_trandate")
    ])

    join = join.join(first_trandate_by_isbn, on="isbn", how="left")

    join = (
        join.with_columns([
            ( (pl.col("trandate").dt.year() - pl.col("first_trandate").dt.year()) * 4
              + (pl.col("trandate").dt.quarter() - pl.col("first_trandate").dt.quarter())
            ).alias("q_since_first")
        ])
    )

    join = (
        join.with_columns([
            (pl.col("trandate").dt.year().cast(pl.Utf8) + "Q" + pl.col("trandate").dt.quarter().cast(pl.Utf8))
            .alias("year_quarter"),
        ])
    )

    groupby_cols = ["isbn", "title", "year_quarter", "format", "channel", "q_num", "q_since_first"]
    sales_groupby = (
        join
        .group_by(groupby_cols)
        .agg([
            pl.col("quantity").sum().alias("quantity"),
            pl.col("discount_rate").mean().alias("avg_discount_rate")
        ])
        .sort(["quantity"], descending=True)
    )

    select_cols = [
            "isbn",
            "print_length",
            "item_weight",
            "length",
            "width",
            "height",
            "rating",
            "number_of_reviews",
            "price",
    ]

    train = (
        pl.read_csv(data_folder / "train_all_cols_v3.csv", schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float32})
        .unique("isbn", keep="first")
        .select(select_cols)
        .drop("number_of_reviews")
    )
    test = (
        pl.read_csv(data_folder / "test_all_cols_v3.csv", schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float32})
        .unique("isbn", keep="first")
        .select(select_cols)
        .drop("number_of_reviews")
    )
    train_test = pl.concat([train, test])

    sales_features = (
        sales_groupby.join(train_test, on="isbn", how="left")
    )

    # sales_features.write_csv(data_folder / "target_series_new_with_features2.csv", include_bom=True)
    # sales_features.write_excel(data_folder / "target_series_new_with_features2.xlsx")
    # Combine features with sales_groupby
    sales_features = (
        pl.read_csv(data_folder / "target_series_new_with_features2.csv",
                    schema_overrides={"isbn": pl.Utf8,
                                      "print_length": pl.Float32,
                                      "number_of_reviews": pl.Float32,
                                      "length": pl.Float32,
                                      "item_weight": pl.Float32,
                                      "width": pl.Float32,
                                      "height": pl.Float32,
                                      "rating": pl.Float32,
                                      "price": pl.Float32,
                                      "avg_discount_rate": pl.Float32
                    })
    )

    # groupby_series = (
    #     sales_features.group_by("series").agg([
    #         pl.col("print_length").median().alias("series_print_length"),
    #         pl.col("length").median().alias("series_length"),
    #         pl.col("width").median().alias("series_width"),
    #         pl.col("height").median().alias("series_height"),
    #         pl.col("rating").median().alias("series_rating"),
    #         pl.col("item_weight").median().alias("series_item_weight"),
    #         pl.col("price").median().alias("series_price"),
    #     ])
    # )
    # sales_features = (
    #     sales_features.join(groupby_series, on="series", how="left")
    #     .with_columns([
    #         pl.when(pl.col("print_length").is_null())
    #         .then(pl.col("series_print_length"))
    #         .otherwise(pl.col("print_length"))
    #         .alias("print_length"),
    #         pl.when(pl.col("length").is_null())
    #         .then(pl.col("series_length"))
    #         .otherwise(pl.col("length"))
    #         .alias("length"),
    #         pl.when(pl.col("width").is_null())
    #         .then(pl.col("series_width"))
    #         .otherwise(pl.col("width"))
    #         .alias("width"),
    #         pl.when(pl.col("height").is_null())
    #         .then(pl.col("series_height"))
    #         .otherwise(pl.col("height"))
    #         .alias("height"),
    #         pl.when(pl.col("rating").is_null())
    #         .then(pl.col("series_rating"))
    #         .otherwise(pl.col("rating"))
    #         .alias("rating"),
    #         pl.when(pl.col("item_weight").is_null())
    #         .then(pl.col("series_item_weight"))
    #         .otherwise(pl.col("item_weight"))
    #         .alias("item_weight"),
    #         pl.when(pl.col("price").is_null())
    #         .then(pl.col("series_price"))
    #         .otherwise(pl.col("price"))
    #         .alias("price")
    #     ])
    #     .drop([
    #         "series_print_length",
    #         "series_length",
    #         "series_width",
    #         "series_height",
    #         "series_rating",
    #         "series_item_weight",
    #         "series_price"
    #     ])
    # )



    sales_groupby = (
        sales_groupby.to_dummies(columns=["format", "channel", "q_num"], drop_first=True)
    )

    cols_order = [col for col in sales_groupby.columns if col != "quantity"] + ["quantity"]
    sales_groupby = sales_groupby.select(cols_order)

    # sales_features.write_csv(data_folder / "target_series_new_with_features.csv", include_bom=True)
    # sales_features.write_excel(data_folder / "target_series_new_with_features.xlsx")


    print("end")