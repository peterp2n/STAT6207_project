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
        purchase = pl.read_database("SELECT * FROM purchase", connection=conn)

    purchase = (
        purchase
        .with_columns([
            pl.col("trandate")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f").dt.date()
            .alias("trandate")
        ])
        .with_columns([
            pl.col("trandate").dt.year().alias("year"),
            pl.col("trandate").dt.month().alias("month"),
            pl.col("trandate").min().alias("first_trandate"),
            pl.when(pl.col("trandate").dt.quarter() == 1).then(pl.lit("1"))
            .when(pl.col("trandate").dt.quarter() == 2).then(pl.lit("2"))
            .when(pl.col("trandate").dt.quarter() == 3).then(pl.lit("3"))
            .otherwise(pl.lit("4")).alias("q_num")
        ])
        .with_columns([
            ((pl.col("trandate").dt.year() - pl.col("first_trandate").dt.year()) * 4
             + (pl.col("trandate").dt.quarter() - pl.col("first_trandate").dt.quarter())
             ).alias("q_since_first"),
            pl.concat_str(pl.col("trandate").dt.year().cast(pl.Utf8) + pl.lit("Q") + pl.col("trandate").dt.quarter().cast(pl.Utf8))
            .alias("year_quarter"),
        ])
        .rename({"product": "isbn"})
    )

    purchase_groupby_cols = [
        "isbn",
        "q_since_first",
        "year_quarter",
        "q_num"
    ]
    groupby_purchase = (
        purchase
        .group_by(purchase_groupby_cols)
        .agg(pl.col("quantity").sum())
    )

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

    products = (
        products
        .drop(["isbn", "barcode", "product"])
        .rename({"barcode2": "isbn", "price": "marked_price"})
        .with_columns([
            pl.col("isbn").cast(pl.Utf8),
            pl.col("book_original_price").cast(pl.Utf8),
        ])
        .with_columns([
            pl.col("book_original_price").str.replace("$", "").alias("book_original_price")
        ])
        .filter(pl.col("isbn").str.starts_with("978"))
    )

    dog_man_mask = products["title"].str.to_lowercase().str.contains("dog man")
    cat_kid_mask = products["title"].str.to_lowercase().str.contains("cat kid")
    captain_mask = products["title"].str.to_lowercase().str.contains("captain underpants")
    andy_mask = products["title"].str.to_lowercase().str.contains("andy griffiths")
    guinness_mask = products["title"].str.to_lowercase().str.contains("guinness")

    target_products = products.filter(
        pl.any_horizontal(
            dog_man_mask,
            cat_kid_mask,
            captain_mask,
            andy_mask,
            guinness_mask
        )
    )
    # target_products.write_csv("target_products.csv", include_bom=True)



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
        "trandate",
        'channel',
        'discount_rate',
        'quantity'
    ]

    join_purchase = target_products.join(purchase, on="isbn", how="inner")
    join_purchase = (
        join_purchase
        .with_columns([
            pl.when(join_purchase["title"].str.to_lowercase().str.contains("dog man")).then(pl.lit("dog_man"))
            .when(join_purchase["title"].str.to_lowercase().str.contains("cat kid")).then(pl.lit("cat_kid"))
            .when(join_purchase["title"].str.to_lowercase().str.contains("captain underpants")).then(pl.lit("captain_underpants"))
            .when(join_purchase["title"].str.to_lowercase().str.contains("andy griffiths")).then(pl.lit("andy_griffiths"))
            .when(join_purchase["title"].str.to_lowercase().str.contains("guinness")).then(pl.lit("guinness"))
            .otherwise(pl.lit("other"))
            .alias("series")
        ])
    )

    reorder_purchase = (
        ["isbn", "title", "series", "year_quarter", "q_num", "q_since_first", "trandate"] +
        [col for col in join_purchase.columns if col not in ["isbn", "title", "series", "year_quarter", "q_num", "q_since_first", "trandate"]]
    )
    join_purchase = join_purchase.select(*reorder_purchase)
    join_purchase.write_csv(data_folder / "purchase.csv", include_bom=True)

    # ['title', 'marked_price', 'book_original_price', 'isbn', 'format', 'channel', 'location', 'invoice', 'clients', 'quantity', 'trandate', 'selling_price', 'discount', 'disc_amt', 'amount', 'discount_rate']
    # Perform the inner join on isbn
    join = target_products.join(sales, on="isbn", how="inner")


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
        join
        .with_columns([
            ((pl.col("trandate").dt.year() - pl.col("first_trandate").dt.year()) * 4
              + (pl.col("trandate").dt.quarter() - pl.col("first_trandate").dt.quarter())
            ).alias("q_since_first"),
            # create a year column for me
            pl.col("trandate").dt.year().alias("year"),
            # create a month column for me
            pl.col("trandate").dt.month().alias("month"),
        ])
    )

    join = (
        join.with_columns([
            (pl.col("trandate").dt.year().cast(pl.Utf8) + "Q" + pl.col("trandate").dt.quarter().cast(pl.Utf8))
            .alias("year_quarter"),
        ])
    )

    # groupby_cols = ["isbn", "title", "year_quarter", "format", "channel", "q_num", "q_since_first"]
    # sales_groupby = (
    #     join
    #     .group_by(groupby_cols)
    #     .agg([
    #         pl.col("quantity").sum().alias("quantity"),
    #         pl.col("discount_rate").mean().alias("avg_discount_rate")
    #     ])
    #     .sort(["quantity"], descending=True)
    # )

    # select_cols = [
    #         "isbn",
    #         "print_length",
    #         "item_weight",
    #         "length",
    #         "width",
    #         "height",
    #         "rating",
    #         "number_of_reviews",
    #         "price",
    # ]

    train = (
        pl.read_csv(data_folder / "train_all_cols_v3.csv", schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float32})
        .unique("isbn", keep="first")
        # .select(select_cols)
        .drop("number_of_reviews")
    )
    test = (
        pl.read_csv(data_folder / "test_all_cols_v3.csv", schema_overrides={"isbn": pl.Utf8, "number_of_reviews": pl.Float32})
        .unique("isbn", keep="first")
        # .select(select_cols)
        .drop("number_of_reviews")
    )


    train_test = pl.concat([train, test])

    target_products = \
    pl.read_csv(data_folder / "target_series_new_with_features2.csv", schema_overrides={"isbn": pl.Utf8})[
        "isbn", "title", "series", "format"]
    join = target_products.join(train_test, on="isbn", how="left")

    join = (
        join.join(train_test, on="isbn", how="left")
    )


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

    cols_order = [
        "isbn",
        "title",
        "series",
        "format",
        "trandate",
        "first_trandate",
        "year_quarter",
        "month",
        "year",
        "q_num",
        "channel",
        "print_length",
        "item_weight",
        "length",
        "width",
        "height",
        "rating",
        "q_since_first",
        "discount_rate",
        "quantity"
    ]

    # sales_raw = join.filter(target_mask).select(cols_order)



    # join.write_csv(data_folder / "target_series_new_with_features.csv", include_bom=True)
    # join.write_excel(data_folder / "target_series_new_with_features.xlsx")


    print("end")