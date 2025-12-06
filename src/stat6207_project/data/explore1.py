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
    for col in continuous_cols:
        plt.figure(figsize=(14, 6))

        # 1. Extract original data
        original_data = sales_features[col].to_pandas()

        # 2. Calculate stats for clipping (Mean +/- 3 STD)
        mean_val = original_data.mean()
        std_val = original_data.std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val

        # 3. Create transformed series
        log_data = np.log1p(original_data)
        clipped_data = np.clip(original_data, lower_bound, upper_bound)

        # 4. Create Standardized version of the CLIPPED data
        # Formula: (x - mean) / std
        clipped_mean = clipped_data.mean()
        clipped_std = clipped_data.std()
        # Handle division by zero if std is 0 (constant column)
        if clipped_std != 0:
            standardized_clipped_data = (clipped_data - clipped_mean) / clipped_std
        else:
            standardized_clipped_data = clipped_data - clipped_mean

        # 5. Combine into DataFrame
        plot_data = pd.DataFrame({
            'Original': original_data,
            'Log1p': log_data,
            'Clipped (±3 std)': clipped_data,
            'Std Clipped': standardized_clipped_data
        })

        # 6. Plot with 4 distinct colors
        sns.boxplot(
            data=plot_data,
            palette=["skyblue", "orange", "green", "purple"]
        )

        plt.title(f"Boxplot of {col}: Comparison of Transformations")
        plt.ylabel("Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
