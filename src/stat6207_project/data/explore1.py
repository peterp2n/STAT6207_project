from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


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

    sns.set_theme(style="whitegrid")

    # --- PART 1: Categorical Plots ---
    for col in ["format", "channel", "q_num"]:
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
            hue=col
        )
        plt.title(f"Total Quantity Sold by {col.capitalize()} for Target Books")
        plt.xlabel(col.capitalize())
        plt.ylabel("Total Quantity Sold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Define the columns of interest
    continuous_cols = [
        "q_since_first",
        "avg_discount_rate",
        "print_length",
        "item_weight",
        "length",
        "width",
        "height",
        "rating",
        "price",
        "quantity"
    ]

    # Dictionary to collect the final transformed data for the heatmap
    transformed_data_collection = {}

    # --- PART 2: Distribution Analysis (Boxplots) ---
    # We run this FIRST to calculate and collect the transformed values
    for col in continuous_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
        axes = axes.flatten()

        # 1. Extract original data
        original_data = sales_features[col].to_pandas()

        # 2. Calculations
        # A. Log1p
        log_data = np.log1p(original_data)

        # B. Clipped (Mean +/- 3 STD of LOG data)
        l_mean = log_data.mean()
        l_std = log_data.std()
        lower_bound = l_mean - 3 * l_std
        upper_bound = l_mean + 3 * l_std
        clipped_data = np.clip(log_data, lower_bound, upper_bound)

        # C. Standardized Clipped ((Clipped - Mean) / Std)
        c_mean = clipped_data.mean()
        c_std = clipped_data.std()
        if c_std != 0:
            std_clipped_data = (clipped_data - c_mean) / c_std
        else:
            std_clipped_data = clipped_data - c_mean

        # --- SAVE DATA FOR HEATMAP ---
        # Store the "Plot 4" data (Standardized Clipped Log1p)
        transformed_data_collection[col] = std_clipped_data

        # --- PLOTTING ---
        # Plot 1: Original (Skyblue)
        sns.boxplot(y=original_data, ax=axes[0], color="skyblue")
        axes[0].set_title(f"1. Original: {col}")
        axes[0].set_ylabel(col)

        # Plot 2: Log1p (Orange)
        sns.boxplot(y=log_data, ax=axes[1], color="orange")
        axes[1].set_title("2. Log1p Transformed")
        axes[1].set_ylabel("Log Value")

        # Plot 3: Clipped (Green)
        sns.boxplot(y=clipped_data, ax=axes[2], color="green")
        axes[2].set_title("3. Clipped (±3 std of Log)")
        axes[2].set_ylabel("Clipped Log Value")

        # Plot 4: Std Clipped (Purple)
        sns.boxplot(y=std_clipped_data, ax=axes[3], color="purple")
        axes[3].set_title("4. Standardized Clipped")
        axes[3].set_ylabel("Z-Score")

        fig.suptitle(f"Distribution Analysis: {col}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # --- PART 3: Correlation Heatmap (Moved After) ---
    # Create a DataFrame from the collected TRANSFORMED data
    transformed_df = pd.DataFrame(transformed_data_collection)

    # Calculate correlation matrix on the transformed data
    # We use dropna() to mimic the previous 'drop_nulls()' behavior for cleaner correlation
    corr_matrix = transformed_df.dropna().corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Matrix of Transformed Features\n(Standardized + Clipped + Log1p)")
    plt.tight_layout()
    plt.show()

