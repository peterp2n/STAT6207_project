from src.stat6207_project.data.preprocess_util_04 import preprocess_data, to_tensors
from pathlib import Path
import polars as pl
import torch

# -------------------------- Device Setup --------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")
if device.type == "mps":
    print("MPS (Apple Silicon GPU) is active!\n")

if __name__ == "__main__":
    data_folder = Path("data")
    df = pl.read_csv(data_folder / "merged3.csv", schema_overrides={"isbn": pl.Utf8})

    sales_data = (
        pl.read_csv(
            "expand_quarterly_sales_retail.csv",
            schema_overrides={
                "barcode2": pl.Utf8,
                "Quarter_num": pl.Utf8,
            }
        )
        .rename({"barcode2": "isbn"})
    )

    unwanted_cols = ["isbn", "title", "author", "Next_Q1", "Next_Q2", "Next_Q3", "Next_Q4",
                     "Next_Q2_log1p", "Next_Q3_log1p", "Next_Q4_log1p", "Current_quarter",
                     "First_day", "Avg_discount"]
    X_train, X_test, book_scaler, sales_scaler = preprocess_data(books_df=df, sales_df=sales_data)
    X_train_tensor, y_train_tensor, _ = to_tensors(df=X_train, target_col="Next_Q1_log1p", drop_cols=unwanted_cols)


    pass