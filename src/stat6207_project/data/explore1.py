from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":


    data_folder = Path("data")

     # Load the sales features data using Polars with schema overrides
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

    # Ensure your styling preferences are set (optional)
    sns.set_theme(style="whitegrid")

    for col in ["format", "channel", "q_num"]:
        # 1. Aggregate in Polars (Fastest)
        # 2. Convert to Pandas for Seaborn compatibility
        format_counts = (
            sales_features.group_by(col)
            .agg(pl.col("quantity").sum().alias("total_quantity"))
            .to_pandas()
        )

        plt.figure(figsize=(8, 6))

        # Plot using Seaborn
        sns.barplot(
            data=format_counts,
            x=col,
            y="total_quantity",
            hue=col
        )

        # Dynamic title based on the column name
        plt.title(f"Total Quantity Sold by {col.capitalize()} for Target Books")
        plt.xlabel(col.capitalize())
        plt.ylabel("Total Quantity Sold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # 1. Define the columns of interest
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

    # 2. Compute correlation in Polars
    # We drop_nulls() to ensure data integrity for the correlation calculation
    corr_matrix = sales_features.select(continuous_cols).drop_nulls().corr()

    # 3. Convert to Pandas for plotting
    # Polars doesn't have an index, so we must manually set it for the heatmap labels
    corr_pdf = corr_matrix.to_pandas()
    corr_pdf.index = continuous_cols  # Set y-axis labels to match x-axis columns

    # 4. Create the Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_pdf,
        annot=True,  # Show correlation values
        fmt=".2f",  # Round to 2 decimal places
        cmap="coolwarm",  # Diverging colormap (Red=High, Blue=Low)
        center=0,  # Center the colormap at 0
        square=True,  # Force square cells
        linewidths=0.5,  # Add grid lines between cells
        cbar_kws={"shrink": 0.8}  # Shrink the color bar slightly
    )

    plt.title("Correlation Matrix of Sales Features")
    plt.tight_layout()
    plt.show()

    # For each column in continuous_cols, plot a boxplot showing outliers
    # Now plot a boxplot of np.log1p transformed values, side by side with the original
    # Iterate through each continuous column
    # --- DISTRIBUTION ANALYSIS ---
    # For each column in continuous_cols, plot a 2x2 grid of boxplots
    for col in continuous_cols:
        # Create a 2x2 grid of subplots
        # sharey=False is crucial because the scales (raw vs log vs z-score) are very different
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

        # Flatten axes array for easier indexing (0, 1, 2, 3)
        axes = axes.flatten()

        # 1. Extract original data
        original_data = sales_features[col].to_pandas()

        # 2. Calculations for transformations
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

        # --- PLOTTING ---

        # Plot 1: Original (Skyblue) - Top Left
        sns.boxplot(y=original_data, ax=axes[0], color="skyblue")
        axes[0].set_title(f"1. Original: {col}")
        axes[0].set_ylabel(col)

        # Plot 2: Log1p (Orange) - Top Right
        sns.boxplot(y=log_data, ax=axes[1], color="orange")
        axes[1].set_title("2. Log1p Transformed")
        axes[1].set_ylabel("Log Value")

        # Plot 3: Clipped (Green) - Bottom Left
        sns.boxplot(y=clipped_data, ax=axes[2], color="green")
        axes[2].set_title("3. Clipped (±3 std of Log)")
        axes[2].set_ylabel("Clipped Log Value")

        # Plot 4: Std Clipped (Purple) - Bottom Right
        sns.boxplot(y=std_clipped_data, ax=axes[3], color="purple")
        axes[3].set_title("4. Standardized Clipped")
        axes[3].set_ylabel("Z-Score")

        # Layout adjustments
        fig.suptitle(f"Distribution Analysis: {col}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
        plt.show()

