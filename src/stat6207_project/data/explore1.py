from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

if __name__ == "__main__":

    data_folder = Path("data")

    # Load the sales features data
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

    # # --- IMPUTATION LOGIC ---
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
    #
    # # We perform imputation on 'sales_features' here.
    # # Note: For the "Before" heatmap, we technically want the RAW data with NaNs.
    # # But since you overwrite 'sales_features' with the imputed version,
    # # we should capture the RAW state first if we want a true "Pre-Imputation" check.
    #
    # # Capture RAW data (for "Before" Heatmap) BEFORE overwriting
    continuous_cols = [
        "q_since_first", "avg_discount_rate", "print_length", "item_weight",
        "length", "width", "height", "rating", "price", "quantity"
    ]
    raw_sales_features_pdf = sales_features.select(continuous_cols).to_pandas()

    sns.set_theme(style="whitegrid")

    # --- PART 1: q_num Plot ---
    col = "q_num"
    format_counts = (
        sales_features.group_by(col)
        .agg(pl.col("quantity").sum().alias("total_quantity"))
        .to_pandas()
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=format_counts,
        x=col,
        y="total_quantity",
        hue=col,
        legend=False
    )
    plt.title(f"Total Quantity Sold by {col.capitalize()} for Target Books")
    plt.xlabel(col.capitalize())
    plt.ylabel("Total Quantity Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- PART 2: Stacked Bar Charts ---
    categorical_cols = ["format", "channel"]
    for cat_col in categorical_cols:
        stacked_data = (
            sales_features
            .group_by(["q_num", cat_col])
            .agg(pl.col("quantity").sum().alias("total_quantity"))
            .sort(cat_col)
            .to_pandas()
        )
        pivot_df = stacked_data.pivot(index="q_num", columns=cat_col, values="total_quantity").fillna(0)
        ax = pivot_df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        plt.title(f"Total Quantity Sold by Quartile (Stacked by {cat_col.capitalize()})")
        plt.xlabel("Quartile (q_num)")
        plt.ylabel("Total Quantity Sold")
        plt.xticks(rotation=0)
        plt.legend(title=cat_col.capitalize())
        plt.tight_layout()
        plt.show()

    # --- PART 3: Distribution Analysis & Data Collection ---
    transformed_data_collection = {}

    for col in continuous_cols:
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
        # axes = axes.flatten()

        # Extract data (This is now the IMPUTED data)
        original_data = sales_features[col].to_pandas()

        # Transformations
        log_data = np.log1p(original_data)

        l_mean = log_data.mean()
        l_std = log_data.std()
        lower_bound = l_mean - 7 * l_std
        upper_bound = l_mean + 7 * l_std
        clipped_data = np.clip(log_data, lower_bound, upper_bound)

        c_mean = clipped_data.mean()
        c_std = clipped_data.std()
        if c_std != 0:
            std_clipped_data = (clipped_data - c_mean) / c_std
        else:
            std_clipped_data = clipped_data - c_mean

        # Collect transformed data for "After" heatmap
        transformed_data_collection[col] = std_clipped_data

        # # Plotting
        # sns.boxplot(y=original_data, ax=axes[0], color="skyblue")
        # axes[0].set_title(f"1. Original (Imputed): {col}")
        # axes[0].set_ylabel(col)
        #
        # sns.boxplot(y=log_data, ax=axes[1], color="orange")
        # axes[1].set_title("2. Log1p Transformed")
        # axes[1].set_ylabel("Log Value")
        #
        # sns.boxplot(y=clipped_data, ax=axes[2], color="green")
        # axes[2].set_title("3. Clipped (±3 std of Log)")
        # axes[2].set_ylabel("Clipped Log Value")
        #
        # sns.boxplot(y=std_clipped_data, ax=axes[3], color="purple")
        # axes[3].set_title("4. Standardized Clipped")
        # axes[3].set_ylabel("Z-Score")
        #
        # fig.suptitle(f"Distribution Analysis: {col}", fontsize=16)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

    # --- PART 4: Separate Heatmaps (Before vs. After) ---

    # 1. Calculate "Before" Matrix (Raw Data with NaNs)
    corr_before = raw_sales_features_pdf.corr()

    # 2. Calculate "After" Matrix (Transformed Data)
    transformed_df = pd.DataFrame(transformed_data_collection)
    corr_after = transformed_df.dropna().corr()

    # Common styling for heatmaps
    heatmap_args = {
        "annot": True,
        "fmt": ".2f",
        "cmap": "coolwarm",
        "center": 0,
        "square": True,
        "linewidths": 1,
        "cbar_kws": {"shrink": 0.7},
        "annot_kws": {"size": 11, "weight": "bold"},
    }

    # Heatmap 1: Before Imputation
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_before, **heatmap_args)
    plt.title("Correlation: Raw Data (Before Imputation)\n(Pair-wise Deletion)", fontsize=18, pad=20)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

    # Heatmap 2: After Imputation + Transformation
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_after, **heatmap_args)
    plt.title("Correlation: Transformed Data (After)\n(Log1p + Clipped + Std)", fontsize=18, pad=20)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- PART 5a: Scatterplots with best-fit line (unscaled) ---

    scatter_x_cols = [c for c in continuous_cols if c != "quantity"]
    sales_pdf = sales_features.select(scatter_x_cols + ["quantity"]).to_pandas()

    for col in scatter_x_cols:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=sales_pdf,
            x=col,
            y="quantity",
            scatter_kws={"alpha": 0.4, "s": 20},
            line_kws={"color": "red", "linewidth": 2},
            ci=None,
        )
        plt.title(f"quantity vs {col} (unscaled) with best-fit line")
        plt.xlabel(col)
        plt.ylabel("quantity")
        plt.tight_layout()
        plt.show()

    # --- PART 5b: Scatterplots with best-fit line AFTER transformation ---
    # Transformation: log1p -> RobustScaler -> clip at ±7

    scaler = RobustScaler()  # median / IQR-based scaling [web:59][web:62]

    for col in scatter_x_cols:
        # Take raw values
        x_raw = sales_pdf[col].to_numpy()
        y_raw = sales_pdf["quantity"].to_numpy()

        # log1p transform (handle non-negative assumption yourself upsteam if needed) [web:64]
        x_log = np.log1p(x_raw)
        y_log = np.log1p(y_raw)

        # Robust scaling (fit per pair so each plot is self-contained)
        xy_log = np.column_stack([x_log, y_log])
        xy_scaled = scaler.fit_transform(xy_log)
        x_scaled = xy_scaled[:, 0]
        y_scaled = xy_scaled[:, 1]

        # Clip at ±7
        x_clipped = np.clip(x_scaled, -7, 7)
        y_clipped = np.clip(y_scaled, -7, 7)

        # Build a temporary DataFrame for plotting
        tmp_df = pd.DataFrame({
            f"{col}_trans": x_clipped,
            "quantity_trans": y_clipped,
        })

        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=tmp_df,
            x=f"{col}_trans",
            y="quantity_trans",
            scatter_kws={"alpha": 0.4, "s": 20},
            line_kws={"color": "red", "linewidth": 2},
            ci=None,
        )
        plt.title(f"Transformed quantity vs {col} (log1p + RobustScaler, clipped ±7)")
        plt.xlabel(f"{col} (transformed)")
        plt.ylabel("quantity (transformed)")
        plt.tight_layout()
        plt.show()
