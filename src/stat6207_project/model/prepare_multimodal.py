# prepare_multimodal_polars.py
import polars as pl
import numpy as np
from pathlib import Path

# --------------------------------------------------------------
data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)

print("Loading data from ./data/ using Polars...\n")

# Load all files
text_emb = pl.read_csv(data_folder / "text_embeddings.csv", dtypes={"isbn": pl.Utf8})
img_emb  = pl.read_csv(data_folder / "images_embeddings.csv", dtypes={"isbn": pl.Utf8})
X_train  = pl.read_csv(data_folder / "X_train_with_isbn.csv", dtypes={"isbn": pl.Utf8})
X_test   = pl.read_csv(data_folder / "X_test_with_isbn.csv",  dtypes={"isbn": pl.Utf8})
y_train  = pl.read_csv(data_folder / "y_train_with_isbn.csv", dtypes={"isbn": pl.Utf8})
y_test   = pl.read_csv(data_folder / "y_test_with_isbn.csv",  dtypes={"isbn": pl.Utf8})

print(f"Rows before inner join:")
print(f"   X_train : {len(X_train):,}")
print(f"   X_test  : {len(X_test):,}")
print(f"   Text emb ISBNs : {text_emb['isbn'].n_unique()}")
print(f"   Img  emb ISBNs : {img_emb['isbn'].n_unique()}")

# INNER JOIN → keep only rows that have text + image embeddings
train_merged = X_train.join(text_emb, on="isbn", how="inner").join(img_emb, on="isbn", how="inner")
test_merged  = X_test.join(text_emb,  on="isbn", how="inner").join(img_emb,  on="isbn", how="inner")

print(f"\nAfter clean inner join:")
print(f"   Train rows : {len(train_merged):,}  ({len(train_merged)/len(X_train):.3%})")
print(f"   Test  rows : {len(test_merged):,}   ({len(test_merged)/len(X_test):.3%})")

# --------------------------------------------------------------
# CRITICAL FIX: align y exactly with the (now possibly reordered) feature tables
# --------------------------------------------------------------
# 1. Get the ISBN order from the merged feature tables
train_isbn_order = train_merged["isbn"]
test_isbn_order  = test_merged["isbn"]

# 2. Re-order y to match that exact order
y_train_aligned = (
    y_train
    .filter(pl.col("isbn").is_in(train_isbn_order))
    .join(pl.DataFrame({"isbn": train_isbn_order}), on="isbn", how="inner")
)

y_test_aligned = (
    y_test
    .filter(pl.col("isbn").is_in(test_isbn_order))
    .join(pl.DataFrame({"isbn": test_isbn_order}), on="isbn", how="inner")
)

# Now the ISBN columns are guaranteed to be identical in order and content
assert train_merged["isbn"].equals(y_train_aligned["isbn"]), "Train ISBNs misaligned!"
assert test_merged["isbn"].equals(y_test_aligned["isbn"]),   "Test ISBNs misaligned!"

print("ISBN alignment verified – perfect match!\n")

# --------------------------------------------------------------
# Convert to numpy (float32) for PyTorch
# --------------------------------------------------------------
X_train_np = train_merged.drop("isbn").to_numpy().astype(np.float32)
X_test_np  = test_merged.drop("isbn").to_numpy().astype(np.float32)

y_train_np = y_train_aligned["Next_Q1_log1p"].to_numpy().reshape(-1, 1).astype(np.float32)
y_test_np  = y_test_aligned["Next_Q1_log1p"].to_numpy().reshape(-1, 1).astype(np.float32)

# # --------------------------------------------------------------
# # Save
# # --------------------------------------------------------------
# np.save(data_folder / "X_train_multimodal.npy", X_train_np)
# np.save(data_folder / "X_test_multimodal.npy",  X_test_np)
# np.save(data_folder / "y_train.npy", y_train_np)
# np.save(data_folder / "y_test.npy",  y_test_np)
#
# print("Multimodal dataset ready!")
# print(f"   Features dimension : {X_train_np.shape[1]}")
# print(f"   Train samples      : {X_train_np.shape[0]:,}")
# print(f"   Test samples       : {X_test_np.shape[0]:,}")
# print(f"\nFiles saved to → {data_folder.resolve()}/")

pass