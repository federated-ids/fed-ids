"""
Utility functions for the federated IDS project.
"""

from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def batch_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield mini-batches from (X, y).

    This is a generator function used to simulate streaming or
    mini-batch training on each client.
    """
    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]

def exploratory_data_analysis(df):
    '''
        Executes EDA on a df
        - DF shape and info
        - Attack count and distribution
        - Feature distrubtion and correlation heatmap
    '''
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe(include="all"))


    missing = df.isnull().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        plt.figure(figsize=(10,4))
        missing.plot(kind="bar")
        plt.title("Missing Value Ratio per Column")
        plt.ylabel("Fraction Missing")
        plt.tight_layout()
        plt.show()


    # The column that contains attack types (including 'normal' or other categories)
    attack_col = "attack"  # replace if your column name is slightly different
    # Filter only rows where this column is NOT 'normal' (or you can just keep all attacks)
    attacks = df[df[attack_col].str.lower() != "normal"]
    # Count occurrences of each attack type
    counts = attacks[attack_col].value_counts()
    

    # ---- Visualization ----

    # Bar chart
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="salmon")
    plt.title("Attack Occurrences by Type")
    plt.xlabel("Attack Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Pie chart
    plt.figure(figsize=(8, 8))
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title("Attack Distribution")
    plt.ylabel("")
    plt.show()
    
    #Feature Distribution
    num_cols = df.select_dtypes(include="number").columns[:5]

    df[num_cols].hist(figsize=(12,6), bins=30)
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.show()

    #Correlation Heatmap
    corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(8,6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.tight_layout()
    plt.show()
  