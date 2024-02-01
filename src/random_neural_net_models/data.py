# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import TensorDict, tensorclass
from torch.utils.data import DataLoader, Dataset

# ============================================
# numpy tabular dataset
# ============================================


@tensorclass
class XyDataTrain:
    x: torch.Tensor
    y: torch.Tensor


class TabularNumpyDatasetTrain(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.n = len(X)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same length, got {X.shape[0]} and {y.shape[0]}"
            )
        if y is not None and y.ndim > 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[[idx], :]).float()
        y = torch.tensor([self.y[idx]])
        y = rearrange(y, "n -> n 1")

        return x, y


def tabular_numpy_collate_train(
    input: T.Tuple[torch.Tensor, torch.Tensor]
) -> XyDataTrain:
    x = torch.concat([v[0] for v in input]).float()
    y = torch.concat([v[1] for v in input]).float()
    return XyDataTrain(x=x, y=y, batch_size=[x.shape[0]])


@tensorclass
class XyDataEval:
    x: torch.Tensor


class TabularNumpyDatasetEval(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(self.X[[idx], :]).float()

        return x


def tabular_numpy_collate_eval(input: T.Tuple[torch.Tensor]) -> XyDataEval:
    x = torch.concat(input).float()
    return XyDataEval(x=x, batch_size=[x.shape[0]])


# ============================================
# mnist image dataset
# ============================================


@tensorclass
class MNISTDataTrain:
    image: torch.Tensor
    label: torch.Tensor


class MNISTDatasetTrain(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        edge: int = 28,
        f: float = 255.0,
        num_classes: int = 10,
    ):
        self.X = X
        self.y = y
        self.n = len(X)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same length, got {X.shape[0]} and {y.shape[0]}"
            )
        if y is not None and y.ndim > 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}")
        self.edge = edge
        self.f = f
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(
            self.X.iloc[idx].values / self.f
        ).float()  # normalizing
        img = rearrange(img, "(h w) -> 1 h w", h=self.edge, w=self.edge)

        label = torch.tensor([int(self.y.iloc[idx])])

        label = F.one_hot(label, num_classes=self.num_classes)
        label[label == 0] = -1  # True = 1, False = -1

        return img, label


def mnist_collate_train(
    input: T.Tuple[torch.Tensor, torch.Tensor]
) -> MNISTDataTrain:
    images = torch.concat([v[0] for v in input]).float()
    labels = torch.concat([v[1] for v in input]).float()
    return MNISTDataTrain(
        image=images, label=labels, batch_size=[images.shape[0]]
    )
