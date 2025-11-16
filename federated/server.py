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

    def broadcast(self) -> None:
        """
        Broadcast the global model to all clients.
        """
        if self.global_params is None:
            raise ValueError("Global model is not set yet.")

        for client in self.clients:
            client.set_model_params(self.global_params)

    def run_training(self, num_rounds: int = 3) -> None:
        """
        Simple federated training loop using while + if.
        """
        current_round = 0
        while current_round < num_rounds:
            for client in self.clients:
                client.train_local_model()

            self.aggregate()
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

