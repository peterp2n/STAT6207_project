import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config

# 1. Magic Switch: Use Polars engine (Fixes the 'concatenate' crash too!)
set_config(transform_output="polars")


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


def preprocess_data(
        books_df: pl.DataFrame,
        sales_df: pl.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
):
    """
    Preprocesses raw dataframes, handles leaks, scaling, and encoding.

    Returns:
    --------
        train_df, test_df (Polars DataFrames - Ready for model input selection)
        scaler_books (Sklearn Scaler for book features)
        scaler_sales (Sklearn Scaler for sales targets/features)
    """
    print("--- Starting Data Preprocessing ---")

    # --- Books Prep ---
    df_feat = books_df.with_columns(pl.col("isbn").cast(pl.Utf8)).drop(["title", "publication_date"], strict=False)

    # --- Split ---
    train_full, test_full = train_test_split(df_feat, test_size=test_size, random_state=random_state, shuffle=shuffle)

    # Separate ISBN (Leakage Prevention)
    train_isbns, test_isbns = train_full.select("isbn"), test_full.select("isbn")
    train_books, test_books = train_full.drop("isbn"), test_full.drop("isbn")

    # --- Imputation ---
    train_books, test_books = impute_by_group(
        train_books, test_books, "publisher", ["book_format", "reading_age"], "mode"
    )
    train_books, test_books = impute_by_group(
        train_books, test_books, "publisher",
        ["print_length", "item_weight", "length", "width", "height", "rating",
         "number_of_reviews", "price", "best_sellers_rank", "customer_reviews"],
        "median"
    )

    # --- Feature Engineering (Books) ---
    log1p_cols_books = ["item_weight", "length", "width", "height", "number_of_reviews",
                        "rating", "customer_reviews", "price", "print_length"]
    for col in log1p_cols_books:
        train_books = train_books.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))
        test_books = test_books.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))

    train_books = train_books.with_columns((1.0 / (pl.col("best_sellers_rank") + 1)).log1p().alias("bsr_inv_log1p"))
    test_books = test_books.with_columns((1.0 / (pl.col("best_sellers_rank") + 1)).log1p().alias("bsr_inv_log1p"))

    final_numeric_cols_books = [
        "print_length_log1p", "rating_log1p", "item_weight_log1p", "length_log1p", "width_log1p",
        "height_log1p", "number_of_reviews_log1p", "customer_reviews_log1p", "price_log1p", "bsr_inv_log1p"
    ]

    scaler_books = StandardScaler()
    train_books = train_books.with_columns(
        scaler_books.fit_transform(train_books.select(final_numeric_cols_books))
        .rename(dict(zip(scaler_books.feature_names_in_, final_numeric_cols_books)))
    )
    test_books = test_books.with_columns(
        scaler_books.transform(test_books.select(final_numeric_cols_books))
        .rename(dict(zip(scaler_books.feature_names_in_, final_numeric_cols_books)))
    )

    # Re-attach ISBNs
    train_books = train_isbns.hstack(train_books)
    test_books = test_isbns.hstack(test_books)

    # --- Sales Prep ---
    sales_data = sales_df.with_columns(pl.col("isbn").cast(pl.Utf8))

    train_sales = sales_data.filter(pl.col("isbn").is_in(train_isbns["isbn"]))
    test_sales = sales_data.filter(pl.col("isbn").is_in(test_isbns["isbn"]))

    # --- Sales Cleaning ---
    cols_to_clean = [
        pl.col("Avg_discount").fill_nan(0.0).fill_null(0.0).clip(lower_bound=0.0).alias("Avg_discount_cleaned"),
        pl.col("Next_Q1").fill_null(0.0), pl.col("Next_Q2").fill_null(0.0),
        pl.col("Next_Q3").fill_null(0.0), pl.col("Next_Q4").fill_null(0.0)
    ]
    train_sales = train_sales.with_columns(cols_to_clean)
    test_sales = test_sales.with_columns(cols_to_clean)

    log1p_cols_sales = ['Previous_quarter_qty', 'Current_quarter_qty', 'Avg_discount_cleaned',
                        'Next_Q1', 'Next_Q2', 'Next_Q3', 'Next_Q4']

    for col in log1p_cols_sales:
        train_sales = train_sales.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))
        test_sales = test_sales.with_columns(pl.col(col).log1p().alias(f"{col}_log1p"))

    final_numeric_cols_sales = [
        'Previous_quarter_qty_log1p', 'Current_quarter_qty_log1p', 'Avg_discount_cleaned_log1p',
        'Next_Q1_log1p', 'Next_Q2_log1p', 'Next_Q3_log1p', 'Next_Q4_log1p'
    ]

    scaler_sales = StandardScaler()
    train_sales = train_sales.with_columns(
        scaler_sales.fit_transform(train_sales.select(final_numeric_cols_sales))
        .rename(dict(zip(scaler_sales.feature_names_in_, final_numeric_cols_sales)))
    )
    test_sales = test_sales.with_columns(
        scaler_sales.transform(test_sales.select(final_numeric_cols_sales))
        .rename(dict(zip(scaler_sales.feature_names_in_, final_numeric_cols_sales)))
    )

    # --- Merge ---
    train_sales = train_isbns.join(train_sales, on="isbn", how="inner")
    test_sales = test_isbns.join(test_sales, on="isbn", how="inner")

    X_train_full = train_books.join(train_sales, on="isbn", how="inner")
    X_test_full = test_books.join(test_sales, on="isbn", how="inner")

    # --- One-Hot Encoding ---
    categorical_variables = ["book_format", "reading_age", "publisher", "Quarter_num"]

    X_train_full = X_train_full.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("MISSING") for c in categorical_variables])
    X_test_full = X_test_full.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("MISSING") for c in categorical_variables])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int8)
    ohe.fit(X_train_full.select(categorical_variables))

    train_encoded = ohe.transform(X_train_full.select(categorical_variables))
    test_encoded = ohe.transform(X_test_full.select(categorical_variables))

    X_train_final = X_train_full.drop(categorical_variables).hstack(train_encoded)
    X_test_final = X_test_full.drop(categorical_variables).hstack(test_encoded)

    print(f"Preprocessing Done! Train Cols: {X_train_final.width}, Test Cols: {X_test_final.width}")

    return X_train_final, X_test_final, scaler_books, scaler_sales


def to_tensors(df: pl.DataFrame, target_col: str, drop_cols: list[str]):
    """Converts a specific dataframe to X and y tensors."""
    # 1. Identify features (exclude target and dropped cols)
    feature_cols = [c for c in df.columns if c != target_col and c not in drop_cols]

    # 2. Cast to Float32 (Critical for PyTorch)
    X_np = df.select(feature_cols).cast(pl.Float32).to_numpy()
    y_np = df.select(target_col).cast(pl.Float32).to_numpy()

    return torch.tensor(X_np), torch.tensor(y_np), feature_cols