import numpy as np
import polars as pl
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':

    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    target = pl.read_excel(data_folder / "target_books.xlsx", schema_overrides={"isbn": pl.Utf8})
    test = (
        pl.read_csv(data_folder / "test_all_cols_unstd_v2.csv",
                    schema_overrides={"isbn": pl.Utf8, "best_sellers_rank": pl.Utf8})
    )



    categorical_variables = ["book_format", "reading_age"]

    target = target.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("MISSING") for c in categorical_variables])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int8)

    # 1. Encode the selected columns
    target_encoded = ohe.fit_transform(target.select(categorical_variables))

    # 2. Convert feature names from numpy array to list
    new_cat_names = ohe.get_feature_names_out(input_features=categorical_variables).tolist()

    # 3. Drop old columns and hstack the new encoded ones
    target = (
        target
        .drop(categorical_variables)
        .hstack(pl.DataFrame(target_encoded, schema=new_cat_names))
        .select(test.columns)
    )

    target_cols = set(target.columns)
    test_cols = set(test.columns)

    difference = test_cols.difference(target_cols)

    target.write_excel(data_folder / "target_books_cleaned.xlsx")

    print("end")