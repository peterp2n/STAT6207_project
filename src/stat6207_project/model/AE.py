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
    train_isbns, test_isbns = train_full.select("isbn"), test_full.select("isbn")
    X_train, X_test = train_full.drop("isbn"), test_full.drop("isbn")

    # Impute categorical book_format and reading_age by mode group by publisher
    X_train, X_test = impute_by_group(
        train_df=X_train,
        test_df=X_test,
        group_col="publisher",
        target_cols=["book_format", "reading_age"],
        strategy="mode"
    )

    # Impute the numeric columns by median group by publisher
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

    # ==================== LOG1P + BEST_SELLERS_RANK FIX ====================
    log1p_cols = [
        "item_weight", "length", "width", "height",
        "number_of_reviews", "customer_reviews", "price"
    ]
    for col in log1p_cols:
        X_train = X_train.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))
        X_test  = X_test.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))

    # Invert + log best_sellers_rank
    X_train = X_train.with_columns(
        (1.0 / (pl.col("best_sellers_rank") + 1)).log1p().alias("bsr_score_log1p")
    )
    X_test = X_test.with_columns(
        (1.0 / (pl.col("best_sellers_rank") + 1)).log1p().alias("bsr_score_log1p")
    )

    # ==================== FINAL SCALING ====================
    final_numeric_cols = [
        "print_length",           # no log
        "rating",                 # no log
        "item_weight_log1p", "length_log1p", "width_log1p", "height_log1p",
        "number_of_reviews_log1p", "customer_reviews_log1p",
        "price_log1p",
        "bsr_score_log1p"
    ]

    scaler_books = StandardScaler()
    X_train = X_train.with_columns(
        scaler_books.fit_transform(X_train.select(final_numeric_cols))
        .rename(dict(zip(scaler_books.feature_names_in_, final_numeric_cols)))
    )
    X_test = X_test.with_columns(
        scaler_books.transform(X_test.select(final_numeric_cols))
        .rename(dict(zip(scaler_books.feature_names_in_, final_numeric_cols)))
    )

    # Re-attach ISBNs
    X_train = train_isbns.hstack(X_train)
    X_test  = test_isbns.hstack(X_test)

    print("merged3 preprocessing complete with proper log1p!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Final scaled columns:", final_numeric_cols)

    sales_data = (
        pl.read_csv(
            "expand_quarterly_sales_retail.csv",
            schema_overrides={
                "barcode2": pl.Utf8,
                "Quarter_num": pl.Enum(["1", "2", "3", "4"]),
            }
        )
        .rename({"barcode2": "isbn"})
    )

    # --- 1. Split sales_data according to the original train/test ISBN split ---

    train_sales = sales_data.filter(pl.col("isbn").is_in(train_isbns["isbn"]))
    test_sales = sales_data.filter(pl.col("isbn").is_in(test_isbns["isbn"]))

    # --- 2. Join book features (already clean & scaled) to sales records ---
    train_sales = train_isbns.join(train_sales, on="isbn", how="left")
    test_sales = test_isbns.join(test_sales, on="isbn", how="left")

    # X_train_full = train_isbns.hstack(X_train)
    # X_test_full = test_isbns.hstack(X_test)

    train_sales = (
        train_sales
        .with_columns([
            pl.col("Avg_discount")
            .replace({float("-inf"): 0.0, float("inf"): 0.0})
            .fill_nan(0.0)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .alias("Avg_discount")
        ])
    )

    numeric_cols_sales = sales_data.select(cs.numeric()).columns
    scaler_sales = StandardScaler()
    # Fit and transform in one step, then replace directly
    train_sales = train_sales.with_columns(
        scaler_sales.fit_transform(train_sales.select(numeric_cols_sales))
        .rename(dict(zip(scaler_sales.feature_names_in_, numeric_cols_sales)))  # safe name restore
    )
    test_sales = test_sales.with_columns(
        scaler_sales.transform(test_sales.select(numeric_cols_sales))
        .rename(dict(zip(scaler_sales.feature_names_in_, numeric_cols_sales)))
    )



    pass