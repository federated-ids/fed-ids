"""
Server module: defines ModelParams and FederatedServer classes.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .client import FederatedClient


class ModelParams:
    """
    Container for logistic regression parameters.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.coef = coef.astype(float)
        self.intercept = intercept.astype(float)

    def __add__(self, other: "ModelParams") -> "ModelParams":
        """
        Elementwise addition of two models.
        """
        return ModelParams(
            coef=self.coef + other.coef,
            intercept=self.intercept + other.intercept,
        )

    def __truediv__(self, scalar: float) -> "ModelParams":
        """
        Divide both coef and intercept by a scalar.
        """
        return ModelParams(
            coef=self.coef / scalar,
            intercept=self.intercept / scalar,
        )

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (coef, intercept) as an immutable tuple.
        """
        return (self.coef, self.intercept)

    def __str__(self) -> str:
        mean_weight = float(np.mean(np.abs(self.coef)))
        return f"ModelParams(|w|_mean={mean_weight:.4f})"


class FederatedServer:
    """
    Coordinates federated training over multiple clients.
    """

    def __init__(self, clients: List[FederatedClient]) -> None:
        self.clients = clients
        self.global_params: ModelParams | None = None
        self.round_accuracies: list[list[float]] = []

    def aggregate(self) -> None:
        """
        Federated averaging of client model parameters.
        """
        params_list = [c.get_model_params() for c in self.clients]
        total = params_list[0]
        for params in params_list[1:]:
            total = total + params

        self.global_params = total / len(params_list)

    def aggregate_median(self) -> None:
        """
        Federated Median - coordinate-wise median for robustness against outliers.
        More robust to Byzantine failures than FedAvg.
        """
        params_list = [c.get_model_params() for c in self.clients]
        
        # Stack all coefficients and intercepts
        coef_stack = np.stack([p.coef for p in params_list], axis=0)
        intercept_stack = np.stack([p.intercept for p in params_list], axis=0)
        
        # Compute coordinate-wise median
        median_coef = np.median(coef_stack, axis=0)
        median_intercept = np.median(intercept_stack, axis=0)
        
        from .server import ModelParams
        self.global_params = ModelParams(coef=median_coef, intercept=median_intercept)


    def aggregate_trimmed_mean(self, trim_ratio: float = 0.1) -> None:
        """
        Federated Trimmed Mean - removes extreme values before averaging.
        Robust to outliers while maintaining averaging properties.
        
        Args:
            trim_ratio: Fraction of extreme values to remove from each side (0 to 0.5)
        """
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        
        params_list = [c.get_model_params() for c in self.clients]
        n_clients = len(params_list)
        n_trim = int(n_clients * trim_ratio)
        
        if n_clients - 2 * n_trim < 1:
            warnings.warn("Too few clients after trimming, using all clients")
            n_trim = 0
        
        # Stack parameters
        coef_stack = np.stack([p.coef for p in params_list], axis=0)
        intercept_stack = np.stack([p.intercept for p in params_list], axis=0)
        
        # Sort along client axis and trim
        coef_sorted = np.sort(coef_stack, axis=0)
        intercept_sorted = np.sort(intercept_stack, axis=0)
        
        if n_trim > 0:
            coef_trimmed = coef_sorted[n_trim:-n_trim]
            intercept_trimmed = intercept_sorted[n_trim:-n_trim]
        else:
            coef_trimmed = coef_sorted
            intercept_trimmed = intercept_sorted
        
        # Mean of trimmed values
        trimmed_coef = np.mean(coef_trimmed, axis=0)
        trimmed_intercept = np.mean(intercept_trimmed, axis=0)
        
        from .server import ModelParams
        self.global_params = ModelParams(coef=trimmed_coef, intercept=trimmed_intercept)


    def aggregate_weighted(self, client_weights: list[float]) -> None:
        """
        Weighted Federated Averaging - weight clients by data size or importance.
        
        Args:
            client_weights: Weight for each client (e.g., number of samples)
        """
        if len(client_weights) != len(self.clients):
            raise ValueError("Number of weights must match number of clients")
        
        params_list = [c.get_model_params() for c in self.clients]
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Weighted sum
        weighted_coef = sum(w * p.coef for w, p in zip(normalized_weights, params_list))
        weighted_intercept = sum(w * p.intercept for w, p in zip(normalized_weights, params_list))
        
        from .server import ModelParams
        self.global_params = ModelParams(coef=weighted_coef, intercept=weighted_intercept)
    
    def broadcast(self) -> None:
        """
        Broadcast the global model to all clients.
        """
        if self.global_params is None:
            raise ValueError("Global model is not set yet.")

        for client in self.clients:
            client.set_model_params(self.global_params)

    def run_training(self, num_rounds: int = 3, aggregation_fn: str = 'aggregate') -> None:
        """
        Federated training loop with configurable aggregation.
        
        Args:
            num_rounds: Number of training rounds
            aggregation_fn: Name of aggregation method to use. Options:
                        - 'aggregate' (default FedAvg)
                        - 'aggregate_median'
                        - 'aggregate_trimmed_mean'
                        - 'aggregate_weighted'
        
        Examples:
            server.run_training(num_rounds=5)  # FedAvg
            server.run_training(num_rounds=5, aggregation_fn='aggregate_median')
            server.run_training(num_rounds=5, aggregation_fn='aggregate_weighted')
        """
        # Get the aggregation method by name
        agg_fn = getattr(self, aggregation_fn)
        
        current_round = 0
        while current_round < num_rounds:
            for client in self.clients:
                client.train_local_model()

            agg_fn()
            self.broadcast()

            accuracies = [client.evaluate() for client in self.clients]
            self.round_accuracies.append(accuracies)

            avg_acc = sum(accuracies) / len(accuracies)
            if self.global_params is not None:
                print(
                    f"[Server] Round {current_round + 1}: "
                    f"{self.global_params}, avg_acc={avg_acc:.3f}"
                )

            current_round += 1


if __name__ == "__main__":
    print("This module is intended to be used via main.ipynb, not run directly.")

