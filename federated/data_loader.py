"""
Data loading and preprocessing for the CYBRIA IoT FL dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .client import FederatedClient
from .exceptions import DataLoadingError, DataValidationError

# Immutable tuple of required columns for the label
REQUIRED_COLS = ("Label",)  # adjust if your Kaggle CSV uses a different label name


def load_cybria_base(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the base CYBRIA CSV and perform minimal validation.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise DataLoadingError(f"CYBRIA CSV not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise DataLoadingError(f"Failed to read CSV: {csv_path}") from exc

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # Simple cleaning: drop rows with missing labels
    df = df.dropna(subset=["Label"])

    return df


def select_feature_columns(df: pd.DataFrame, max_features: int = 20) -> List[str]:
    """
    Select a subset of numeric feature columns from the dataset.
    """
    numeric_cols = [
        col
        for col, dtype in df.dtypes.items()
        if ("float" in str(dtype) or "int" in str(dtype)) and col != "Label"
    ]
    feature_cols = numeric_cols[:max_features]
    return feature_cols


def split_into_clients(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "Label",
    n_clients: int = 3,
) -> List[FederatedClient]:
    """
    Split the full dataset into n_clients random shards, one per FederatedClient.
    """
    # Shuffle to randomize distribution across clients
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shards = np.array_split(df_shuffled, n_clients)

    clients: List[FederatedClient] = []
    for idx, shard in enumerate(shards):
        client_id = f"client_{idx + 1}"
        client = FederatedClient(client_id, shard, feature_cols, label_col)
        clients.append(client)

    # Use enumerate to log what we created
    for idx, client in enumerate(clients):
        print(f"[DataLoader] Client {idx}: {client}")

    return clients


def make_centralized_train_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "Label",
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a centralized train/test split for baseline comparison.
    """
    X = df[feature_cols].values.astype(float)
    y = df[label_col].values.astype(int)

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    csv_demo = root / "data" / "cybria.csv"
    if csv_demo.exists():
        df_demo = load_cybria_base(csv_demo)
        print(f"[DataLoader] Demo: loaded {len(df_demo)} rows from {csv_demo}")
    else:
        print("[DataLoader] Demo: cybria.csv not found (demo only).")

