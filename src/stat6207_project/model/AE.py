import polars as pl
import polars.selectors as cs
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def impute_by_group(
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        group_col: str,
        target_cols: list[str],
        strategy: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Imputes missing values in target_cols using the chosen statistic (mode/median/mean)
    calculated within each group defined by group_col.

    Fits on train_df and transforms both train_df and test_df.

    Fallback order:
      1. Group-level statistic (mode/median/mean)
      2. Global statistic from the training set

    Parameters
    ----------
    train_df, test_df : pl.DataFrame
        Input dataframes (test_df may contain unseen groups).
    group_col : str
        Column name used for grouping.
    target_cols : list[str]
        Columns containing missing values to impute.
    strategy : {"mode", "median", "mean"}
        Statistic to use for imputation.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Imputed train_df and test_df.
    """
    strategy = strategy.lower()
    if strategy not in {"mode", "median", "mean"}:
        raise ValueError("strategy must be one of 'mode', 'median', or 'mean'")

    # --- 1. Fit: Compute global and group-level statistics on train_df ---

    # Global statistics (fallback)
    if strategy == "mode":
        global_stats = {
            col: train_df[col].drop_nulls().mode().first()
            for col in target_cols
        }
    elif strategy == "median":
        global_stats = {
            col: train_df[col].drop_nulls().median()
            for col in target_cols
        }
    else:  # mean
        global_stats = {
            col: train_df[col].drop_nulls().mean()
            for col in target_cols
        }

    # Group-level statistics
    agg_exprs = []
    for col in target_cols:
        base = pl.col(col).drop_nulls()
        if strategy == "mode":
            expr = base.mode().first().alias(f"{strategy}_{col}")
        elif strategy == "median":
            expr = base.median().alias(f"{strategy}_{col}")
        else:  # mean
            expr = base.mean().alias(f"{strategy}_{col}")
        agg_exprs.append(expr)

    group_stats_df = (
        train_df
        .group_by(group_col)
        .agg(agg_exprs)
    )

    # --- 2. Transform: Apply imputation to any dataframe ---
    def apply_imputation(df: pl.DataFrame) -> pl.DataFrame:
        # Left join group statistics
        df_joined = df.join(group_stats_df, on=group_col, how="left")

        # Build sequential fill expressions
        fill_exprs = []
        for col in target_cols:
            group_col_name = f"{strategy}_{col}"
            expr = pl.col(col).fill_null(pl.col(group_col_name))  # Group statistic
            expr = expr.fill_null(global_stats[col])  # Global fallback
            fill_exprs.append(expr)

        # Apply and clean up temporary columns
        result = df_joined.with_columns(fill_exprs)
        temp_cols = [f"{strategy}_{c}" for c in target_cols]
        return result.drop(temp_cols)

    # Apply to both datasets
    train_imputed = apply_imputation(train_df)
    test_imputed = apply_imputation(test_df)

    return train_imputed, test_imputed


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

    # Impute categorical book_format and reading_age by mode group by publisher
    X_train, X_test = impute_by_group(
        train_df=X_train,
        test_df=X_test,
        group_col="publisher",
        target_cols=["book_format", "reading_age"],
        strategy="mode"
    )

    # Impute the numeric columns by median group by publisher (before scaling)
    X_train, X_test = impute_by_group(
        train_df=X_train,
        test_df=X_test,
        group_col="publisher",
        target_cols=[
            "print_length", "item_weight", "length", "width", "height", "rating",
            "number_of_reviews", "price", "best_sellers_rank", "customer_reviews"
        ],
        strategy="median"
    )

    # Standardize the numeric columns (now on imputed data)
    numeric_cols = X_train.select(cs.numeric()).columns
    scaler = StandardScaler()
    # Fit and transform in one step, then replace directly
    X_train = X_train.with_columns(
        scaler.fit_transform(X_train.select(numeric_cols))
        .rename(dict(zip(scaler.feature_names_in_, numeric_cols)))  # safe name restore
    )

    X_test = X_test.with_columns(
        scaler.transform(X_test.select(numeric_cols))
        .rename(dict(zip(scaler.feature_names_in_, numeric_cols)))
    )

    pass