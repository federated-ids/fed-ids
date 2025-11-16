"""
Utility functions for the federated IDS project.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


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

