import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config

# 1. Magic Switch: Use Polars engine (Fixes the 'concatenate' crash too!)
set_config(transform_output="polars")


def decode_pl(df, enc, cols):
    # 1. Extract the imputed float values
    vals = df.select(cols).to_numpy()

    decoded_series = []
    for i, col_name in enumerate(cols):
        # Access the category list for this specific column
        cats = enc.categories_[i]
        max_idx = len(cats) - 1

        # 2. Round, Fill, and Clip
        # This ensures every single value is a valid integer index (0 to N)
        col_idx = vals[:, i]
        col_idx = np.round(col_idx)
        col_idx = np.nan_to_num(col_idx, nan=0)  # Fallback to category 0 if NaN remains
        col_idx = np.clip(col_idx, 0, max_idx).astype(int)

        # 3. Direct Lookup (Manual Inverse Transform)
        # We use numpy indexing to pull the strings directly.
        # This cannot produce NaNs unless the category list itself contains them.
        predicted_strings = cats[col_idx]

        decoded_series.append(pl.Series(col_name, predicted_strings))

    # 4. Return dataframe with the float columns replaced by string columns
    return df.with_columns(decoded_series)




if __name__ == "__main__":
    # --- Load & Prep ---
    data_folder = Path("data")
    df = pl.read_csv(data_folder / "merged3.csv", schema_overrides={"isbn": pl.Utf8})
    df_feat = df.drop(["title", "publication_date"])

    # --- Split ---
    train_full, test_full = train_test_split(df_feat, test_size=0.2, random_state=42, shuffle=True)

    # Separate ISBN (Leakage Prevention)
    train_isbn, test_isbn = train_full.select("isbn"), test_full.select("isbn")
    X_train, X_test = train_full.drop("isbn"), test_full.drop("isbn")

    # --- Pipeline ---
    cat_cols = ["publisher", "book_format", "reading_age"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', Pipeline([
                ('encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan
                ))
            ]), cat_cols),
            ('num', 'passthrough', num_cols)
        ], verbose_feature_names_out=False)),

        # Smart Impute
        ('imputer', IterativeImputer(max_iter=10, random_state=0, initial_strategy='most_frequent')),

        # SAFETY NET: If IterativeImputer gives up (leaving NaNs), fill with Mode.
        # This prevents the "NaN in decode" crash.
        ('safety_net', SimpleImputer(strategy='most_frequent'))
    ])

    # --- Run ---
    print("Running Imputation...")
    X_train_imp = pipeline.fit_transform(X_train)
    X_test_imp = pipeline.transform(X_test)

    # --- Decode & Finalize ---
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']

    train_ready = train_isbn.hstack(decode_pl(X_train_imp, encoder, cat_cols))
    test_ready = test_isbn.hstack(decode_pl(X_test_imp, encoder, cat_cols))

    print(f"Final Train Shape: {train_ready.shape}")
    print(f"Sample:\n{train_ready.head(1)}")

    mode_format = (
        train_ready
        .group_by("publisher")
        .agg(
            pl.col("book_format").drop_nulls()
            .mode()
            .first()
            .alias("mode_book_format")
        )
    )
    mode_age = (
        train_ready
        .group_by("publisher")
        .agg(
            pl.col("reading_age").drop_nulls()
            .mode()
            .first()
            .alias("mode_reading_age")
        )
    )

    train_ready = train_ready.join(mode_age, on="publisher", how="left")
    train_ready = train_ready.join(mode_format, on="publisher", how="left")
    train_ready = (
        train_ready
        .with_columns(
            pl.col("book_format").fill_null(pl.col("mode_book_format")),
            pl.col("reading_age").fill_null(pl.col("mode_reading_age"))
        )
        .drop(["mode_book_format", "mode_reading_age"])
    )

    pass