"""
Data Loading & Feature Engineering Pipeline
===========================================
Handles:
1. Fast Parquet ingestion and series identification.
2. 24-Hour hourly stock state unpacking and historical feature extraction.
3. Cyclical temporal feature engineering (Day-of-Week sin/cos).
4. Rolling stockout and sales momentum features.
5. Entity encoding (Product ID, City ID).
6. Chronological train/validation/holdout test window generation.
7. PyTorch Dataset and DataLoader construction.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

HIST_STOCK_COLS = [f"stock_h{h:02d}" for h in range(24)]
TARGET_COLUMNS = [f"target_stock_h{h:02d}" for h in range(24)]

CONTINUOUS_FEATURES = [
    "sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag",
    "activity_flag", "precpt", "avg_temperature", "avg_humidity",
    "hours_sale_sum", "hours_stock_status_sum", "dow_sin", "dow_cos",
    "stockout_rolling_3", "sales_momentum"
]


def to_float_list(value, length=24):
    """Converts array, list, or scalar into a fixed-length float32 list."""
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    values = [0.0 if pd.isna(item) else float(item) for item in values[:length]]
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return values


class FullScaleDataset(Dataset):
    """PyTorch Dataset returning historical lookback windows, prior day stock, entity metadata, and targets."""
    def __init__(self, x, prev_stock, cat_meta, y):
        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        self.prev = torch.tensor(np.array(prev_stock), dtype=torch.float32)
        self.cat = torch.tensor(np.array(cat_meta), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.prev[i], self.cat[i], self.y[i]


def get_dataloaders(
    data_path="data/top15_train.parquet",
    seq_len=14,
    batch_size=256,
    val_days=29,
    test_days=14,
    verbose=True
):
    """
    Loads parquet data, processes features, extracts sliding windows,
    and returns DataLoaders for train, val, and test splits.
    """
    t0 = time.time()
    if verbose:
        print(f"Loading data from {data_path}...")
    
    df = pd.read_parquet(data_path)
    df["series_id"] = (
        df["city_id"].astype(str) + "_"
        + df["store_id"].astype(str) + "_"
        + df["product_id"].astype(str)
    )
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt", "city_id", "store_id", "product_id"])
    df = df.sort_values(["series_id", "dt"]).reset_index(drop=True)

    # 1. Expand 24-hour stock status matrix
    stock_matrix = np.array(df["hours_stock_status"].map(to_float_list).tolist(), dtype=np.float32)
    for h in range(24):
        df[HIST_STOCK_COLS[h]] = stock_matrix[:, h]

    df["hours_stock_status_sum"] = stock_matrix.sum(axis=1)
    df["hours_sale_sum"] = df["hours_sale"].map(lambda v: float(np.sum(to_float_list(v))))

    # 2. Cyclical temporal features
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 3. Numeric fills and dynamic features
    for col in ["sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag",
                "activity_flag", "precpt", "avg_temperature", "avg_humidity",
                "hours_sale_sum", "hours_stock_status_sum"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["stockout_rolling_3"] = df.groupby("series_id")["stock_hour6_22_cnt"].transform(
        lambda s: (s > 0).rolling(3, min_periods=1).sum()
    )
    df["sales_momentum"] = df.groupby("series_id")["sale_amount"].diff().fillna(0.0)

    # 4. Target definition (next day's 24h stock status)
    for src, tgt in zip(HIST_STOCK_COLS, TARGET_COLUMNS):
        df[tgt] = df.groupby("series_id")[src].shift(-1)

    clean_df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)

    # 5. Entity Mappings
    unique_prods = sorted(clean_df["product_id"].unique())
    unique_cities = sorted(clean_df["city_id"].unique())
    num_products = len(unique_prods)
    num_cities = len(unique_cities)

    prod2idx = {pid: i for i, pid in enumerate(unique_prods)}
    city2idx = {cid: i for i, cid in enumerate(unique_cities)}
    clean_df["pid_idx"] = clean_df["product_id"].map(prod2idx)
    clean_df["cid_idx"] = clean_df["city_id"].map(city2idx)

    # 6. Chronological split cutoffs
    dates_sorted = sorted(clean_df["dt"].unique())
    val_cutoff_start = dates_sorted[-val_days]
    test_cutoff_start = dates_sorted[-test_days]

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    train_mask = clean_df["dt"] < val_cutoff_start
    scaler.fit(clean_df.loc[train_mask, CONTINUOUS_FEATURES])
    clean_df[CONTINUOUS_FEATURES] = scaler.transform(clean_df[CONTINUOUS_FEATURES])

    # 7. Extract sliding windows
    train_x, train_prev, train_cat, train_y = [], [], [], []
    val_x,   val_prev,   val_cat,   val_y   = [], [], [], []
    test_x,  test_prev,  test_cat,  test_y  = [], [], [], []

    for _, group in clean_df.groupby("series_id", sort=False):
        group = group.sort_values("dt").reset_index(drop=True)
        feats = group[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)
        prev_stocks = group[HIST_STOCK_COLS].to_numpy(dtype=np.float32)
        tgts = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        p_idx = int(group["pid_idx"].iloc[0])
        c_idx = int(group["cid_idx"].iloc[0])
        dates = group["dt"].to_numpy()

        L = len(group)
        if L <= seq_len:
            continue

        for end in range(seq_len, L):
            target_date = pd.Timestamp(dates[end])
            x_win = feats[end - seq_len : end]
            prev_stock_day = prev_stocks[end - 1]
            y_tgt = tgts[end - 1]
            cat_meta = [p_idx, c_idx]

            if target_date < val_cutoff_start:
                train_x.append(x_win)
                train_prev.append(prev_stock_day)
                train_cat.append(cat_meta)
                train_y.append(y_tgt)
            elif target_date < test_cutoff_start:
                val_x.append(x_win)
                val_prev.append(prev_stock_day)
                val_cat.append(cat_meta)
                val_y.append(y_tgt)
            else:
                test_x.append(x_win)
                test_prev.append(prev_stock_day)
                test_cat.append(cat_meta)
                test_y.append(y_tgt)

    if verbose:
        print(f"Windows extracted in {time.time()-t0:.1f}s: {len(train_x):,} train, {len(val_x):,} val, {len(test_x):,} test")

    train_loader = DataLoader(FullScaleDataset(train_x, train_prev, train_cat, train_y), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(FullScaleDataset(val_x,   val_prev,   val_cat,   val_y),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(FullScaleDataset(test_x,  test_prev,  test_cat,  test_y),  batch_size=batch_size, shuffle=False)

    metadata = {
        "num_products": num_products,
        "num_cities": num_cities,
        "num_features": len(CONTINUOUS_FEATURES),
        "scaler": scaler,
        "train_size": len(train_x),
        "val_size": len(val_x),
        "test_size": len(test_x),
    }

    return train_loader, val_loader, test_loader, metadata
