# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import torch
import tqdm
from sklearn import base


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))


def sigmoid_derivative(z: torch.Tensor) -> torch.Tensor:
    exp = torch.exp(z)
    return exp / (1 + exp) ** 2


class Rumelhart1986PerceptronClassifier(
    base.BaseEstimator, base.ClassifierMixin
):
    def __init__(
        self,
        n_hidden: T.Tuple[int] = (10, 5),
        eps: float = 0.1,
        alpha: float = 0.0,
        random_state: int = 42,
        verbose: bool = False,
        epochs: int = 10,
    ) -> None:
        self.n_hidden = n_hidden
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose
        self.epochs = epochs
        self.alpha = alpha

    def _handle_Xy(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> T.Tuple[torch.tensor, T.Optional[torch.tensor]]:
        _X = torch.from_numpy(X).double()  # (N_samples, N_features)
        if y is None:
            return _X, None
        _y = torch.from_numpy(y).double()  # (N_samples,)
        return _X, _y

    @torch.no_grad()
    def _forward(self, X: torch.tensor, training: bool = False) -> torch.tensor:
        a = torch.concat(
            (X, torch.ones((X.shape[0], 1), dtype=torch.double)), dim=1
        )

        if training:
            self.telemetry_["z"] = []
            self.telemetry_["a"] = [a]

        for w in self.weights_:
            z = a @ w
            a = sigmoid(z)

            if training:
                self.telemetry_["z"].append(z)
                self.telemetry_["a"].append(a)

        return a

    @torch.no_grad()
    def _backward(self, y_pred: torch.tensor, y: torch.tensor) -> None:
        # MSE loss derivative
        delta = y_pred - y.view((-1, 1))

        # iterating through the layers in reverse order
        # from the output layer to the input layer
        ix_layers = list(range(len(self.weights_)))[::-1]
        gradients = [None for _ in ix_layers]
        if self.alpha > 0:
            old_gradients = (
                self.telemetry_["gradients"]
                if "gradients" in self.telemetry_
                else None
            )

        for i in ix_layers:
            z = self.telemetry_["z"][i]
            a = self.telemetry_["a"][i]

            delta = delta * sigmoid_derivative(z)

            # part to be added to the weight for layer i
            dw = a.T @ delta
            gradients[i] = dw

            w = self.weights_[i]
            delta = delta @ w.T

        for i in ix_layers:
            dw = self.eps * gradients[i]

            if self.alpha > 0 and old_gradients is not None:
                dw += self.alpha * old_gradients[i]

            self.weights_[i] -= dw

        if self.alpha > 0:
            self.telemetry_["gradients"] = gradients

    @torch.no_grad()
    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "Rumelhart1986PerceptronClassifier":
        torch.manual_seed(self.random_state)

        self.units_ = (
            [X.shape[1] + 1] + [h + 1 for h in self.n_hidden] + [1]
        )  # +1 because of bias
        self.weights_ = [
            torch.randn(self.units_[i], self.units_[i + 1], dtype=torch.double)
            for i in range(len(self.units_) - 1)
        ]
        self.telemetry_ = dict()
        self.errors_ = torch.zeros(
            self.epochs, dtype=torch.float32
        )  # (N_epochs,)
        X, y = self._handle_Xy(X, y)

        for epoch in tqdm.tqdm(range(self.epochs), disable=not self.verbose):
            y_pred = self._forward(X, training=True)
            self._backward(y_pred, y)
            self.errors_[epoch] = torch.mean((y_pred - y) ** 2)

        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X, _ = self._handle_Xy(X)
        y_pred = self._forward(X, training=False)
        return y_pred.detach().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.predict_proba(X)
        return y_pred > 0.5
