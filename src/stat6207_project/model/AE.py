import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
import torch

# Enable Polars output for Sklearn transformers
set_config(transform_output="polars")


def decode_pl(df, enc, cols):
    """Decodes imputed ordinal integers back to original category strings."""
    vals = df.select(cols).to_numpy().round()
    decoded = enc.inverse_transform(vals)
    return df.with_columns([pl.Series(c, decoded[:, i]) for i, c in enumerate(cols)])


if __name__ == "__main__":
    # -------------------------- Device Setup --------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print("MPS (Apple Silicon GPU) is active!\n")

    # -------------------------- Load & Split Data --------------------------
    data_folder = Path("data")
    df = pl.read_csv(data_folder / "merged3.csv", schema_overrides={"isbn": pl.Utf8})
    df_feat = df.drop(["title", "publication_date"])

    # --- Split (Polars Native) ---
    train_full, test_full = train_test_split(df_feat, test_size=0.2, random_state=42, shuffle=True)

    # Separate ISBN to prevent leakage/modification
    train_isbn, test_isbn = train_full.select("isbn"), test_full.select("isbn")
    X_train, X_test = train_full.drop("isbn"), test_full.drop("isbn")

    # --- Pipeline Setup ---
    cat_cols = ["publisher", "book_format", "reading_age"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Pipeline: Encode strings -> Impute missing values
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', Pipeline([
                ('encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan
                ))
            ]), cat_cols),
            ('num', 'passthrough', num_cols)
        ], verbose_feature_names_out=False)),
        ('imputer', IterativeImputer(max_iter=10, random_state=0, initial_strategy='most_frequent'))
    ])

    # --- Execute Imputation ---
    print("Running Imputation...")
    X_train_imp = pipeline.fit_transform(X_train)
    X_test_imp = pipeline.transform(X_test)

    # --- Decode & Reassemble ---
    # Retrieve encoder to map numbers back to strings
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']

    # Decode categories and re-attach ISBNs
    train_ready = train_isbn.hstack(decode_pl(X_train_imp, encoder, cat_cols))
    test_ready = test_isbn.hstack(decode_pl(X_test_imp, encoder, cat_cols))

    print(f"Final Train Shape: {train_ready.shape}")
    print(f"Sample: {train_ready.head(1)}")

    pass