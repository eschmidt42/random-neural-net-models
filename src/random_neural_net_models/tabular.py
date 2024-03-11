# -*- coding: utf-8 -*-
import torch.nn as nn
import typing as T
import torch
import random_neural_net_models.data as rnnm_data
import random_neural_net_models.utils as utils

logger = utils.get_logger("tabular.py")


class Layer(nn.Module):
    def __init__(
        self, n_in: int, n_out: int, use_batch_norm: bool, use_activation: bool
    ) -> None:
        super().__init__()

        if use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features=n_in)
        else:
            self.bn = nn.Identity()

        self.lin = nn.Linear(n_in, n_out)

        if use_activation:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self.net = nn.Sequential(self.bn, self.lin, self.act)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class TabularModel(nn.Module):
    def __init__(
        self, n_hidden: T.List[int], use_batch_norm: bool
    ) -> None:  # ,task:Task
        super().__init__()

        layers = []
        for i, (n_in, n_out) in enumerate(zip(n_hidden[:-1], n_hidden[1:])):
            is_not_last = i <= len(n_hidden) - 3

            layers.append(
                Layer(
                    n_in=n_in,
                    n_out=n_out,
                    use_batch_norm=use_batch_norm,
                    use_activation=is_not_last,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, input: rnnm_data.XyBlock) -> torch.Tensor:
        return self.net(input.x)


class TabularModelClassification(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: T.List[int],
        n_classes: int,
        use_batch_norm: bool,
    ) -> None:  # ,task:Task
        super().__init__()

        n_acts = [n_features] + n_hidden + [n_classes]
        self.net = TabularModel(n_hidden=n_acts, use_batch_norm=use_batch_norm)

    def forward(self, input: rnnm_data.XyBlock) -> torch.Tensor:
        return self.net(input)


class StandardNormalScaler(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean


class TabularModelRegression(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: T.List[int],
        mean: float,
        std: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()

        n_acts = (
            [n_features] + n_hidden + [1]
        )  # hard-coded that only one target is predicted
        self.net = TabularModel(n_hidden=n_acts, use_batch_norm=use_batch_norm)
        self.scaler = StandardNormalScaler(mean=mean, std=std)

    def forward(self, input: rnnm_data.XyBlock) -> torch.Tensor:
        y = self.net(input)
        y = self.scaler(y)
        return y
