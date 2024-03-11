# -*- coding: utf-8 -*-
import torch.nn as nn
import typing as T
import torch
import random_neural_net_models.data as rnnm_data
import random_neural_net_models.utils as utils
from enum import Enum
import numpy as np

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


class BiasSources(Enum):
    zero = "zero"
    normal = "normal"


class ImputeMissingness(nn.Module):
    def __init__(self, n_features: int, bias_source: BiasSources):
        super().__init__()

        match bias_source:
            case BiasSources.zero:
                bias = torch.zeros((1, n_features), dtype=torch.float)
            case BiasSources.normal:
                bias = torch.rand((1, n_features), dtype=torch.float)
            case _:
                raise NotImplementedError(
                    f"{bias_source=} is not implemented in BiasSources, knows members: {BiasSources._member_names_()}"
                )

        self.bias = nn.Parameter(bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        is_finite = torch.isfinite(X)
        is_finite = is_finite  # .to(X.device)
        is_infinite = torch.logical_not(is_finite)
        bias = self.bias.expand(X.shape[0], -1)  # .to(X.device)
        X_imputed = X.masked_fill(is_infinite, 0)
        X_imputed += bias.masked_fill(is_finite, 0)

        is_infinite = is_infinite.float()
        X_out = torch.hstack((X_imputed, is_infinite))
        return X_out


class TabularModel(nn.Module):
    def __init__(
        self,
        n_hidden: T.List[int],
        use_batch_norm: bool,
        do_impute: bool = False,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        layers = []
        if do_impute:
            layers.append(
                ImputeMissingness(
                    n_features=n_hidden[0], bias_source=impute_bias_source
                )
            )
            n_hidden[0] = (
                n_hidden[0] * 2
            )  # because ImputeMissingness horizontally stacks boolean missingness flags

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
        do_impute: bool = False,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        n_acts = [n_features] + n_hidden + [n_classes]
        self.net = TabularModel(
            n_hidden=n_acts,
            use_batch_norm=use_batch_norm,
            do_impute=do_impute,
            impute_bias_source=impute_bias_source,
        )

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
        do_impute: bool = False,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        n_acts = (
            [n_features] + n_hidden + [1]
        )  # hard-coded that only one target is predicted
        self.net = TabularModel(
            n_hidden=n_acts,
            use_batch_norm=use_batch_norm,
            do_impute=do_impute,
            impute_bias_source=impute_bias_source,
        )
        self.scaler = StandardNormalScaler(mean=mean, std=std)

    def forward(self, input: rnnm_data.XyBlock) -> torch.Tensor:
        y = self.net(input)
        y = self.scaler(y)
        return y


def make_missing(
    X: np.ndarray, p_missing: float = 0.1
) -> T.Tuple[np.ndarray, np.ndarray]:
    mask = np.random.choice(
        [False, True], size=X.shape, p=[1 - p_missing, p_missing]
    )
    X_miss = X.copy()
    X_miss[mask] = float("inf")
    return X_miss, mask
