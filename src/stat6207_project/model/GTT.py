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

# Numeric columns for imputation
numeric_cols = ['print_length', 'item_weight', 'length', 'width', 'height', 'rating', 'price']