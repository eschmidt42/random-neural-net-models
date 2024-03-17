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
        is_finite = is_finite
        is_infinite = torch.logical_not(is_finite)
        bias = self.bias.expand(X.shape[0], -1)
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
        n_categories_per_column: T.List[int] = None,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        n_acts = [n_features] + n_hidden + [n_classes]
        if n_categories_per_column is None:
            self.net = TabularModel(
                n_hidden=n_acts,
                use_batch_norm=use_batch_norm,
                do_impute=do_impute,
                impute_bias_source=impute_bias_source,
            )
        else:
            self.net = TabularModelNumericalAndCategorical(
                n_hidden=n_acts,
                n_categories_per_column=n_categories_per_column,
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
        n_categories_per_column: T.List[int] = None,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        n_acts = (
            [n_features] + n_hidden + [1]
        )  # hard-coded that only one target is predicted

        if n_categories_per_column is None:
            self.net = TabularModel(
                n_hidden=n_acts,
                use_batch_norm=use_batch_norm,
                do_impute=do_impute,
                impute_bias_source=impute_bias_source,
            )
        else:
            self.net = TabularModelNumericalAndCategorical(
                n_hidden=n_acts,
                n_categories_per_column=n_categories_per_column,
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


def make_missing_categorical(
    X: np.ndarray, p_missing: float = 0.1, value: int = -1
) -> T.Tuple[np.ndarray, np.ndarray]:
    mask = np.random.choice(
        [False, True], size=X.shape, p=[1 - p_missing, p_missing]
    )
    X_miss = X.copy()
    X_miss[mask] = value
    return X_miss, mask


def calc_categorical_feature_embedding_dimension(n_cat: int) -> int:
    # https://github.com/fastai/fastai/blob/1fec8a2380d6de28d081435e88683a440c47a2f1/fastai/tabular/model.py#L16C12-L16C46
    return min(600, round(1.6 * n_cat**0.56))


class TabularModelNumericalAndCategorical(nn.Module):
    def __init__(
        self,
        n_hidden: T.List[int],
        n_categories_per_column: T.List[int],
        use_batch_norm: bool,
        do_impute: bool = False,
        impute_bias_source: BiasSources = BiasSources.zero,
    ) -> None:
        super().__init__()

        n_num_in = n_hidden[0] - len(n_categories_per_column)
        self.do_impute = do_impute

        if do_impute:
            self.impute = ImputeMissingness(
                n_features=n_num_in, bias_source=impute_bias_source
            )
            n_num_in *= 2  # because ImputeMissingness horizontally stacks boolean missingness flags

        self.categoric_column_ids = list(range(len(n_categories_per_column)))

        ids_to_keep = [
            i for i, v in enumerate(n_categories_per_column) if v > 1
        ]
        if len(ids_to_keep) < len(n_categories_per_column):
            ids_to_drop = [
                v
                for i, v in enumerate(self.categoric_column_ids)
                if i not in ids_to_keep
            ]
            n_cats_per_col_to_drop = [
                v
                for i, v in enumerate(n_categories_per_column)
                if i not in ids_to_keep
            ]
            msg = f"found that ({len(ids_to_drop)}) provided categorical columns ({ids_to_drop}) had an ordinality of < 2 ({n_cats_per_col_to_drop}). dropping them."
            logger.warning(msg)

        self.categoric_column_ids = [
            self.categoric_column_ids[i] for i in ids_to_keep
        ]
        self.n_categories_per_column = [
            n_categories_per_column[i] + 1 for i in ids_to_keep
        ]  # +1 for the fallback category, e.g. missing value

        self.emb_dims = [
            calc_categorical_feature_embedding_dimension(n_categories)
            for n_categories in self.n_categories_per_column
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(n_categories, emb_dim)
                for n_categories, emb_dim in zip(
                    self.n_categories_per_column, self.emb_dims
                )
            ]
        )

        n_emb_in = sum(self.emb_dims)

        n_hidden[0] = n_num_in + n_emb_in

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

    def forward(self, input: rnnm_data.XyBlock_numcat) -> torch.Tensor:
        x_num = input.x_numerical
        if self.do_impute:
            x_num = self.impute(x_num)
        x_cat = input.x_categorical
        x_emb = torch.cat(
            [
                embedding(x_cat[:, i])
                for i, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        x = torch.cat([x_num, x_emb], dim=1)
        return self.net(x)
