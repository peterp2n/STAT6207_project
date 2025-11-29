import polars as pl
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # -------------------------- Device Setup --------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print("MPS (Apple Silicon GPU) is active!\n")

    # -------------------------- Load & Split Data --------------------------
    data_folder = Path("data")
    scrape = pl.read_csv(data_folder / "merged3.csv", schema_overrides={"isbn": pl.Utf8})
    feat = scrape.drop(["isbn", "title", "publication_date"])
    feat_dummy = feat.to_dummies(columns=["publisher", "book_format", "reading_age"], drop_first=True)
    train_df, test_df = train_test_split(feat_dummy, test_size=0.2, random_state=42, shuffle=True)



    pass