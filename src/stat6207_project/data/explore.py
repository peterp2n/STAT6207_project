import sqlite3
from pathlib import Path
import polars as pl

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

    groupby_isbn = join.group_by("isbn").agg([
        pl.col("trandate").min().alias("first_trandate")
    ])
    join = join.join(groupby_isbn, on="isbn", how="left")

    join = (
        join.with_columns([
            ( (pl.col("trandate").dt.year() - pl.col("first_trandate").dt.year()) * 4
              + (pl.col("trandate").dt.quarter() - pl.col("first_trandate").dt.quarter())
            ).alias("q_since_first")
        ])
    )

    sales_groupby = (
        join
        .group_by("q_since_first").agg([pl.col("quantity").sum()])
        .sort(["quantity"], descending=True)
    )

    join = join.drop(["trandate", "first_trandate"])


    join = (
        join.to_dummies(columns=["format", "channel", "q_num"], drop_first=True)
    )

    join = join.select([col for col in join.columns if col != "quantity"] + ["quantity"])

    # If you want to see the filtered results:
    dog_man_books = join.filter(dog_man_mask)
    cat_kid_books = join.filter(cat_kid_mask)
    captain_books = join.filter(captain_mask)
    andy_books = join.filter(andy_mask)
    guinness_books = join.filter(guinness_mask)
    target_books = join.filter(target_mask)

    target_books.write_csv(data_folder / "target_books_new.csv", include_bom=True)
    target_books.write_excel(data_folder / "target_books_new.xlsx")


    print("end")