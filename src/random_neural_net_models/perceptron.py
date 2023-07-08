# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn import base

import random_neural_net_models.utils as utils

logger = utils.get_logger("perceptron", level="DEBUG")


class PerceptronClassifier(base.BaseEstimator, base.ClassifierMixin):
    """OG neural net

    Implementation follows: https://en.wikipedia.org/wiki/Perceptron
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _handle_Xy(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> T.Tuple[torch.tensor, T.Optional[torch.tensor]]:
        _X = torch.from_numpy(X).double()  # (N_samples, N_features)
        if y is None:
            return _X, None
        _y = torch.from_numpy(y).double()  # (N_samples,)
        return _X, _y

    @torch.no_grad()
    def _fit_two_class(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)

        self.weights_ = torch.randn(
            X.shape[1], dtype=torch.double
        )  # (N_features,)
        self.bias_ = torch.zeros(1, dtype=torch.double)  # (1,)

        _X, _y = self._handle_Xy(X, y)
        self.errors_ = torch.zeros(
            self.epochs, dtype=torch.float32
        )  # (N_epochs,)

        for epoch in tqdm.tqdm(range(self.epochs), disable=not self.verbose):
            y_hat = _X @ self.weights_ + self.bias_

            y_hat_discrete = torch.where(
                y_hat > 0, 1.0, 0.0
            )  # heaviside step function

            dy = _y - y_hat_discrete
            dw = self.learning_rate * _X.T @ dy
            y_error = torch.sum(dy)
            db = self.learning_rate * y_error

            self.weights_ += dw
            self.bias_ += db

            self.errors_[epoch] = y_error.abs()

    @torch.no_grad()
    def _fit_multi_class(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)

        # different to two-class perceptron, we need to have a weight vector for each class
        self.weights_ = torch.randn(
            (X.shape[1], self.n_classes_), dtype=torch.double
        )  # (N_features,N_classes)
        self.bias_ = torch.zeros(1, dtype=torch.double)  # (1,)

        _X, _y = self._handle_Xy(X, y)
        _y_int = _y.clone().long()
        # we convert the integer labels to one-hot vectors to be able to apply the same math as for the two-class perceptron
        y_onehot = F.one_hot(_y_int, num_classes=self.n_classes_)
        self.errors_ = torch.zeros(
            self.epochs, dtype=torch.float32
        )  # (N_epochs,)

        for epoch in tqdm.tqdm(range(self.epochs), disable=not self.verbose):
            y_hat = _X @ self.weights_ + self.bias_

            # different to two-class perceptron, we select the class with the highest score
            # directly, without using the heaviside step function
            y_hat_discrete = torch.argmax(y_hat, dim=1)

            # now onehot encoding the discrete predictions
            y_hat_onehot = F.one_hot(
                y_hat_discrete, num_classes=self.n_classes_
            )

            dy = y_onehot - y_hat_onehot

            dw = self.learning_rate * _X.T @ dy.double()

            y_error = torch.sum(
                dy.abs()
            )  # TODO: why no convergence if .abs is missing here?
            db = self.learning_rate * y_error

            self.weights_ += dw
            self.bias_ += db

            self.errors_[epoch] = y_error

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronClassifier":
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.is_multi_class_ = self.n_classes_ > 2

        if self.is_multi_class_:
            logger.debug(
                f"More than two classes detected ({self.classes_}), treating as multi-class problem."
            )
            self._fit_multi_class(X, y)
        else:
            logger.debug(
                f"Two classes detected ({self.classes_}), treating as two-class problem."
            )
            self._fit_two_class(X, y)

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        _X, _ = self._handle_Xy(X, y=None)
        y_hat = _X @ self.weights_ + self.bias_

        if self.is_multi_class_:
            y_pred = torch.argmax(y_hat, dim=1).detach().numpy()
            return y_pred
        else:
            y_pred = torch.where(y_hat > 0, 1.0, 0.0).detach().numpy()
            return y_pred
