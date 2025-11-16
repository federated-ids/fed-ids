"""
Client module: defines the FederatedClient class using logistic regression.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .utils import batch_generator


class FederatedClient:
    """
    Represents one federated learning client with local IoT traffic data.
    """

    def __init__(self, client_id: str, data: pd.DataFrame, feature_cols: List[str], label_col: str) -> None:
        self.client_id = client_id
        # Convert feature columns to numeric, handling errors
        X_df = data[feature_cols].apply(pd.to_numeric, errors='coerce')
        # Replace NaN and inf values with 0
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.X = X_df.values.astype(float)
        self.y = data[label_col].values.astype(int)
        self.model = LogisticRegression(
            max_iter=200,
            solver="lbfgs",
            n_jobs=-1,
        )

    def train_local_model(self, batch_size: int = 2048) -> None:
        """
        Fit local logistic regression using mini-batches.
        """
        X_batches = []
        y_batches = []
        for X_batch, y_batch in batch_generator(self.X, self.y, batch_size=batch_size):
            X_batches.append(X_batch)
            y_batches.append(y_batch)

        X_train = np.vstack(X_batches)
        y_train = np.hstack(y_batches)

        self.model.fit(X_train, y_train)

    def get_model_params(self) -> "ModelParams":
        """
        Extract model parameters as a ModelParams object.
        """
        from .server import ModelParams

        if not hasattr(self.model, "coef_"):
            raise ValueError(f"Client {self.client_id}: model not fitted yet.")

        coef = self.model.coef_.copy()
        intercept = self.model.intercept_.copy()

        return ModelParams(coef=coef, intercept=intercept)

    def set_model_params(self, params: "ModelParams") -> None:
        """
        Set this client's model parameters from a ModelParams object.
        """
        self.model.coef_ = params.coef.copy()
        self.model.intercept_ = params.intercept.copy()
        self.model.n_features_in_ = self.X.shape[1]
        self.model.classes_ = np.array([0, 1], dtype=int)

    def evaluate(self) -> float:
        """
        Evaluate accuracy on this client's local data.
        """
        y_pred = self.model.predict(self.X)
        return accuracy_score(self.y, y_pred)

    def __str__(self) -> str:
        return f"FederatedClient(id={self.client_id}, n_samples={len(self.y)})"

