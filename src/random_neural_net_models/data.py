# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import tensorclass
from torch.utils.data import Dataset
from functools import partial

# ============================================
# numpy tabular dataset
# ============================================


class NumpyTrainingDataset(Dataset):
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


class NumpyNumCatTrainingDataset(Dataset):
    # negative values in X_categorical indicate missingness, because X_categorical is expected to be of type int

    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray
    n: int
    cat_maps: T.Dict[int, T.Dict[int, int]]
    inv_cat_maps: T.Dict[int, T.Dict[int, int]]
    cat_fallbacks: T.Dict[int, int]

    def __init__(self, X_numerical: np.ndarray, X_categorical, y: np.ndarray):
        self.X_num = X_numerical
        self.X_cat = X_categorical
        self.y = y
        self.n = len(X_numerical)

        if X_numerical.shape[0] != X_categorical.shape[0]:
            raise ValueError(
                f"X_numerical and X_categorical must have same length, got {X_numerical.shape[0]} and {X_categorical.shape[0]}"
            )
        if X_numerical.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same length, got {X_numerical.shape[0]} and {y.shape[0]}"
            )
        if y is not None and y.ndim > 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}")

        self._create_cat_id_maps()

    def _create_cat_id_maps(self):
        self.cat_maps = {}
        self.cat_fallbacks = {}
        self.inv_cat_maps = {}

        for col_i, col_vals in enumerate(self.X_cat.T):
            unique_categories = set(v for v in col_vals if v >= 0)
            _map = {val: _id for _id, val in enumerate(unique_categories)}
            self.cat_maps[col_i] = _map
            _inv_map = {_id: val for _id, val in _map.items()}
            self.cat_fallbacks[col_i] = len(_map)
            self.inv_cat_maps[col_i] = _inv_map

    def __len__(self):
        return self.n

    def _map_cats_to_ids(self, x_cat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    self.cat_maps[i].get(v, self.cat_fallbacks[i])
                    for i, v in enumerate(x_cat)
                ]
            ],
            dtype=np.integer,
        )

    def __getitem__(
        self, idx: int
    ) -> T.Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        x_num = torch.from_numpy(self.X_num[[idx], :]).float()
        x_cat = torch.from_numpy(
            self._map_cats_to_ids(self.X_cat[idx, :])
        ).long()
        y = torch.tensor([self.y[idx]])
        y = rearrange(y, "n -> n 1")

        return x_num, x_cat, y


class NumpyNumCatTrainingDatasetXOnly(Dataset):
    # negative values in X_categorical indicate missingness, because X_categorical is expected to be of type int

    X_num: np.ndarray
    X_cat: np.ndarray
    n: int
    cat_maps: T.Dict[int, T.Dict[int, int]]
    inv_cat_maps: T.Dict[int, T.Dict[int, int]]
    cat_fallbacks: T.Dict[int, int]

    def __init__(self, X_numerical: np.ndarray, X_categorical):
        self.X_num = X_numerical
        self.X_cat = X_categorical
        self.n = len(X_numerical)

        if X_numerical.shape[0] != X_categorical.shape[0]:
            raise ValueError(
                f"X_numerical and X_categorical must have same length, got {X_numerical.shape[0]} and {X_categorical.shape[0]}"
            )

        self._create_cat_id_maps()

    def _create_cat_id_maps(self):
        self.cat_maps = {}
        self.cat_fallbacks = {}
        self.inv_cat_maps = {}

        for col_i, col_vals in enumerate(self.X_cat.T):
            unique_categories = set(v for v in col_vals if v >= 0)
            _map = {val: _id for _id, val in enumerate(unique_categories)}
            self.cat_maps[col_i] = _map
            _inv_map = {_id: val for _id, val in _map.items()}
            self.cat_fallbacks[col_i] = len(_map)
            self.inv_cat_maps[col_i] = _inv_map

    def __len__(self):
        return self.n

    def _map_cats_to_ids(self, x_cat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    self.cat_maps[i].get(v, self.cat_fallbacks[i])
                    for i, v in enumerate(x_cat)
                ]
            ],
            dtype=np.integer,
        )

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.LongTensor]:
        x_num = torch.from_numpy(self.X_num[[idx], :]).float()
        x_cat = torch.from_numpy(
            self._map_cats_to_ids(self.X_cat[idx, :])
        ).long()

        return x_num, x_cat


def calc_n_categories_per_column(X_cat: np.ndarray) -> T.List[int]:
    # X_cat is expected to be an array of integers.
    # hence only values >= 0 count as a category and negative values indicate missingness
    unique_cats = [np.unique(X_cat[:, i]) for i in range(X_cat.shape[1])]
    unique_cats = [[v for v in col_vals if v >= 0] for col_vals in unique_cats]
    return [len(col_vals) for col_vals in unique_cats]


def get_index_ranges_from_n_cats_per_col(
    n_categories_per_column: T.Iterable[int],
) -> T.List[T.Tuple[int]]:
    if len(n_categories_per_column) == 0:
        msg = f"{n_categories_per_column=} must have at least one element"
        raise ValueError(msg)
    if any(v < 1 for v in n_categories_per_column):
        msg = (
            f"{n_categories_per_column=} must have only elements greater than 0"
        )
        raise ValueError(msg)

    index_ranges = []
    offset = 0
    for n_cats in n_categories_per_column:
        _range = tuple(range(offset, offset + n_cats))
        index_ranges.append(_range)
        offset += n_cats
    return index_ranges


@tensorclass
class XyBlock:
    x: torch.Tensor
    y: torch.Tensor


@tensorclass
class XyBlock_numcat:  # for separate numerical and categorical data in x
    x_numerical: torch.Tensor
    x_categorical: torch.LongTensor
    y: torch.Tensor


def collate_numpy_dataset_to_xyblock_template(
    input: T.Tuple[torch.Tensor, torch.Tensor],
    make_y_float: bool,
) -> XyBlock:
    x = torch.concat([v[0] for v in input]).float()
    y = torch.concat([v[1] for v in input])
    if make_y_float:
        y = y.float()
    return XyBlock(x=x, y=y, batch_size=[x.shape[0]])


collate_numpy_dataset_to_xyblock = partial(
    collate_numpy_dataset_to_xyblock_template, make_y_float=True
)
collate_numpy_dataset_to_xyblock_keep_orig_y = partial(
    collate_numpy_dataset_to_xyblock_template, make_y_float=False
)


def collate_numpy_numcat_dataset_to_xyblock_template(
    input: T.Tuple[torch.Tensor, torch.LongTensor, torch.Tensor],
    make_y_float: bool,
) -> XyBlock:
    x_num = torch.concat([v[0] for v in input]).float()
    x_cat = torch.concat([v[1] for v in input]).long()
    y = torch.concat([v[2] for v in input])
    if make_y_float:
        y = y.float()
    return XyBlock_numcat(
        x_numerical=x_num, x_categorical=x_cat, y=y, batch_size=[x_num.shape[0]]
    )


collate_numpy_numcat_dataset_to_xyblock = partial(
    collate_numpy_numcat_dataset_to_xyblock_template, make_y_float=True
)
collate_numpy_numcat_dataset_to_xyblock_keep_orig_y = partial(
    collate_numpy_numcat_dataset_to_xyblock_template, make_y_float=False
)


class NumpyInferenceDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(self.X[[idx], :]).float()

        return x


@tensorclass
class XBlock:
    x: torch.Tensor


@tensorclass
class XBlock_numcat:  # for separate numerical and categorical data in x
    x_numerical: torch.Tensor
    x_categorical: torch.LongTensor


def collate_numpy_dataset_to_xblock(
    input: torch.Tensor,
) -> XBlock:
    x = torch.stack([v[0] for v in input]).float()

    return XBlock(x=x, batch_size=[x.shape[0]])


def collate_numpy_numcat_dataset_to_xblock(
    input: T.Tuple[torch.Tensor, torch.LongTensor],
) -> XBlock_numcat:
    x_num = torch.concat([v[0] for v in input]).float()
    x_cat = torch.concat([v[1] for v in input]).long()
    return XBlock_numcat(
        x_numerical=x_num, x_categorical=x_cat, batch_size=[x_num.shape[0]]
    )


# ============================================
# mnist image dataset
# ============================================


class MNISTDatasetWithLabels(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        edge: int = 28,
        f: float = 255.0,
        num_classes: int = 10,
        one_hot: bool = True,
        transform: nn.Module = None,
        add_channel: bool = True,
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
        self.one_hot = one_hot
        self.transform = transform
        self.add_channel = add_channel

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(
            self.X.iloc[idx].values / self.f
        ).float()  # normalizing
        if self.add_channel:
            img = rearrange(img, "(h w) -> 1 h w", h=self.edge, w=self.edge)
        else:
            img = rearrange(img, "(h w) -> h w", h=self.edge, w=self.edge)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor([int(self.y.iloc[idx])])

        if self.one_hot:
            label = F.one_hot(label, num_classes=self.num_classes)
            label[label == 0] = -1  # True = 1, False = -1
            label = label.float()
        else:
            label = label.type(torch.int64)

        return img, label


@tensorclass
class MNISTBlockWithLabels:
    image: torch.Tensor
    label: torch.Tensor


def collate_mnist_dataset_to_block_with_labels(
    input: T.List[T.Tuple[torch.Tensor, torch.Tensor]],
) -> MNISTBlockWithLabels:
    images = torch.concat([v[0] for v in input])
    labels = torch.concat([v[1] for v in input])
    return MNISTBlockWithLabels(
        image=images, label=labels, batch_size=[images.shape[0]]
    )


class MNISTDatasetWithNoise(Dataset):
    # to generate images from noise
    def __init__(
        self,
        images: torch.Tensor,
        noises: torch.Tensor,
        add_channel: bool = True,
    ):
        self.images = images
        self.noises = noises
        self.n = len(images)
        if images.shape[0] != noises.shape[0]:
            raise ValueError(
                f"images and noises must have same length, got {images.shape[0]} and {noises.shape[0]}"
            )
        self.add_channel = add_channel

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        noise = self.noises[idx]

        if self.add_channel:
            img = rearrange(img, "h w -> 1 h w")

        return img, torch.tensor([noise])


@tensorclass
class MNISTBlockWithNoise:
    noisy_image: torch.Tensor
    noise_level: torch.Tensor


def collate_mnist_dataset_to_block_with_noise(
    input: T.List[T.Tuple[torch.Tensor, torch.Tensor]],
) -> MNISTBlockWithNoise:
    images = torch.concat([v[0] for v in input])  # .float()
    noise_levels = torch.concat([v[1] for v in input])  # .float()

    return MNISTBlockWithNoise(
        noisy_image=images,
        noise_level=noise_levels,
        batch_size=[images.shape[0]],
    )
