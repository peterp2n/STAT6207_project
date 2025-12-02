import polars as pl
from pathlib import Path

data_folder = Path("data")

# Load everything
df_text  = (
    pl.read_csv(data_folder / "text_embeddings.csv", schema_overrides={"isbn": pl.Utf8})
)
df_img   = (
    pl.read_csv(data_folder / "images_embeddings.csv", schema_overrides={"isbn": pl.Utf8})
)

train = pl.read_csv(data_folder / "train_features_target_only.csv", schema_overrides={"isbn": pl.Utf8})
test = pl.read_csv(data_folder / "test_features_target_only.csv", schema_overrides={"isbn": pl.Utf8})

# Images are already unique → just verify
assert df_text["isbn"].is_duplicated().sum() == 0, "text_embeddings has duplicates!"
assert df_img["isbn"].is_duplicated().sum() == 0, "images_embeddings has duplicates!"

# =============================================================================
# Now safe left joins → order & row count 100% preserved
# =============================================================================
train_enriched = (
    train
    .join(df_text, on="isbn", how="left")
    .join(df_img,  on="isbn", how="left")
)

test_enriched = (
    test
    .join(df_text, on="isbn", how="left")
    .join(df_img,  on="isbn", how="left")
)

train_enriched.write_csv(data_folder / "train_enriched.csv", include_bom=True)
test_enriched.write_csv(data_folder / "test_enriched.csv", include_bom=True)

print(f"Success! X_train_enriched: {train_enriched.shape}")
print(f"Success! X_test_enriched:  {test_enriched.shape}")