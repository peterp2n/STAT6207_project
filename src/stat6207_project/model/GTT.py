from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# ==================== PATH & LOAD DATA ====================
data_folder = Path("data")
df = pd.read_csv(data_folder / "target_series_new_with_features2.csv")

target_col = 'quantity'

# Drop year_quarter completely – we only use q_since_first as time index
df = df.drop(columns=['year_quarter'], errors='ignore')

# Sort by isbn and q_since_first (critical!)
df = df.sort_values(['isbn', 'q_since_first']).reset_index(drop=True)

# Label-encode categorical columns (format, channel, series) - NOT isbn
# This is done globally as categories are shared across all data
cat_cols = ['format', 'channel', 'series']
label_maps = {}
for col in cat_cols:
    df[col], uniques = pd.factorize(df[col])
    label_maps[col] = uniques
    print(f"{col}: {len(uniques)} unique values")

# Factorize isbn for indexing purposes (but not as a categorical feature)
df['isbn'], isbn_uniques = pd.factorize(df['isbn'])
label_maps['isbn'] = isbn_uniques

# Numeric columns for imputation (NO PRICE - it's noise)
numeric_cols = ['print_length', 'item_weight', 'length', 'width', 'height', 'rating']

# Create a time series split based on 'q_since_first' using 3 splits
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

print("\n" + "=" * 60)
print("TIME SERIES SPLIT SETUP")
print("=" * 60)
print(f"Using {n_splits}-fold Time Series Cross-Validation")
print(f"Total rows in dataset: {len(df)}")

# Loop through each split
all_fold_data = []

for fold_idx, (train_val_idx, test_idx) in enumerate(tscv.split(df), 1):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx}/{n_splits}")
    print(f"{'=' * 60}")

    # Split train_val into train and validation (80/20 split)
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Further split train_val into train and validation
    val_size = int(len(train_val_df) * 0.2)
    train_df = train_val_df.iloc[:-val_size].copy()
    val_df = train_val_df.iloc[-val_size:].copy()

    print(f"Train set: {len(train_df)} rows ({len(train_df.isbn.unique())} unique ISBNs)")
    print(f"Validation set: {len(val_df)} rows ({len(val_df.isbn.unique())} unique ISBNs)")
    print(f"Test set: {len(test_df)} rows ({len(test_df.isbn.unique())} unique ISBNs)")

    # Fill missing static book features (per ISBN median) for each set separately
    for dataset_name, dataset in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for col in numeric_cols:
            if col in dataset.columns:
                dataset[col] = dataset.groupby('isbn')[col].transform(
                    lambda x: x.fillna(x.median() if x.notna().any() else 0)
                )

    # Store the split data
    all_fold_data.append({
        'fold': fold_idx,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    })

print(f"\n{'=' * 60}")
print(f"All {n_splits} folds prepared successfully!")
print(f"{'=' * 60}")
