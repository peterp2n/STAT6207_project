from pathlib import Path

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
    cols = [
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
    corr_matrix = sales_features.select(cols).drop_nulls().corr()

    # 3. Convert to Pandas for plotting
    # Polars doesn't have an index, so we must manually set it for the heatmap labels
    corr_pdf = corr_matrix.to_pandas()
    corr_pdf.index = cols  # Set y-axis labels to match x-axis columns

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

    # for col in ["format", "channel", "q_num"]:
    #     # Plot a bar chart for me quantity groupby format
    #     plt.figure(figsize=(8, 6))
    #     format_counts = sales_features.group_by(col).agg(pl.col("quantity").sum().alias("total_quantity"))
    #     plt.bar(format_counts[col], format_counts["total_quantity"], color='skyblue')
    #     plt.title("Total Quantity Sold by Format for Target Books")
    #     plt.xlabel(col)
    #     plt.ylabel("Total Quantity Sold")
    #     plt.xticks(rotation=45)
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.tight_layout()
    #     plt.show()