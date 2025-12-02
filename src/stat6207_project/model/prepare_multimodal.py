import polars as pl
from pathlib import Path

data_folder = Path("data")

# Load everything
df_text  = pl.read_csv(data_folder / "text_embeddings.csv", schema_overrides={"isbn": pl.Utf8})
df_img   = pl.read_csv(data_folder / "images_embeddings.csv", schema_overrides={"isbn": pl.Utf8})
df_train = pl.read_csv(data_folder / "X_train_with_isbn.csv", schema_overrides={"isbn": pl.Utf8})
df_test  = pl.read_csv(data_folder / "X_test_with_isbn.csv", schema_overrides={"isbn": pl.Utf8})

# =============================================================================
# CRITICAL FIX: Deduplicate text embeddings (mean pooling is safe & common)
# =============================================================================
print(f"Before deduplication - text_embeddings: {len(df_text)} rows, {df_text['isbn'].n_unique()} unique ISBNs")

df_text = (
    df_text
    .group_by("isbn")
    .agg([
        # Mean over duplicate embeddings → robust and preserves semantics
        pl.col("^dim_.*$").mean()
    ])
    .sort("isbn")  # optional: makes debugging easier
)

print(f"After deduplication  - text_embeddings: {len(df_text)} rows (all unique)")

# Images are already unique → just verify
assert df_img["isbn"].is_duplicated().sum() == 0, "images_embeddings has duplicates!"

# =============================================================================
# Now safe left joins → order & row count 100% preserved
# =============================================================================
X_train_enriched = (
    df_train
    .join(df_text, on="isbn", how="left")
    .join(df_img,  on="isbn", how="left", suffix="_img")
)

X_test_enriched = (
    df_test
    .join(df_text, on="isbn", how="left")
    .join(df_img,  on="isbn", how="left", suffix="_img")
)

# =============================================================================
# These asserts will now PASS and are meaningful
# =============================================================================
assert len(X_train_enriched) == len(df_train)
assert len(X_test_enriched)  == len(df_test)
assert X_train_enriched["isbn"].to_list() == df_train["isbn"].to_list()
assert X_test_enriched["isbn"].to_list()  == df_test["isbn"].to_list()

X_train_enriched.write_csv(data_folder / "X_train_enriched.csv", include_bom=True)
X_test_enriched.write_csv(data_folder / "X_test_enriched.csv", include_bom=True)

print(f"Success! X_train_enriched: {X_train_enriched.shape}")
print(f"Success! X_test_enriched:  {X_test_enriched.shape}")