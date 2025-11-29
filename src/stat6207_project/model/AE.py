import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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


def impute_by_group_mode(train_df, test_df, group_col, target_cols):
    """
    Imputes missing values in target_cols using the mode of the group_col.
    Fits on train_df, transforms both train_df and test_df.

    Strategy:
    1. Try filling with the Mode of the specific group (e.g., Publisher).
    2. If Group Mode is missing (unknown group or group has no data), fill with Global Mode from train set.
    """

    # --- 1. Fit (Calculate Statistics on Train) ---

    # A. Global Modes (Fallback)
    # resulting dict: {'book_format': 'Hardcover', 'reading_age': '8-12'}
    global_modes = {
        col: train_df[col].drop_nulls().mode().first()
        for col in target_cols
    }

    # B. Group Modes
    # We build one agg expression per target column to do this in a single groupby
    agg_exprs = [
        pl.col(col).drop_nulls().mode().first().alias(f"mode_{col}")
        for col in target_cols
    ]

    group_modes_df = (
        train_df
        .group_by(group_col)
        .agg(agg_exprs)
    )

    # --- 2. Transform (Apply to Dataframes) ---

    def apply_imputation(df):
        # Join the group modes once
        df_joined = df.join(group_modes_df, on=group_col, how="left")

        # Define fill expressions for all columns
        fill_exprs = []
        for col in target_cols:
            fill_exprs.append(
                pl.col(col)
                .fill_null(pl.col(f"mode_{col}"))  # Priority 1: Group Mode
                .fill_null(global_modes[col])  # Priority 2: Global Mode
            )

        # Apply fills and drop the temp mode columns
        return df_joined.with_columns(fill_exprs).drop([f"mode_{c}" for c in target_cols])

    # Apply to both
    return apply_imputation(train_df), apply_imputation(test_df)


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

    target_cols = ["book_format", "reading_age"]

    X_train, X_test = impute_by_group_mode(
        train_df=X_train,
        test_df=X_test,
        group_col="publisher",
        target_cols=target_cols
    )

    print("Imputation Complete")
    print(X_train.select(target_cols).null_count())

    pass