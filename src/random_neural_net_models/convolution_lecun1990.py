# -*- coding: utf-8 -*-
import typing as T

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset

import random_neural_net_models.utils as utils

logger = utils.get_logger("convolution_lecun1990.py")


class DigitsDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, edge: int = 28):
        self.X = X
        self.y = y
        self.edge = edge

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, int]:
        img = (
            torch.from_numpy(self.X.iloc[idx].values / 255.0)  # normalizing
            .reshape(self.edge, self.edge)
            .double()
        )
        label = int(self.y.iloc[idx])
        return (img, label)


def calc_conv_output_dim(input_dim, kernel_size, padding, stride):
    return int((input_dim - kernel_size + 2 * padding) / stride + 1)


def densify_y(y: torch.Tensor) -> torch.Tensor:
    new_y = F.one_hot(y, num_classes=10)
    new_y[new_y == 0] = -1
    return new_y.double()


class Tanh(nn.Module):
    def __init__(self, A: float = 1.716, S: float = 2 / 3):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.register_buffer("S", torch.tensor(S))

    def forward(self, x: torch.Tensor):
        return self.A * torch.tanh(self.S * x)


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        edge: int,
        n_in_channels: int = 1,
        n_out_channels: int = 1,
        kernel_width: int = 5,
        kernel_height: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        lecun_init: bool = True,
    ):
        super().__init__()

        self.register_buffer("edge", torch.tensor(edge))
        self.register_buffer("n_in_channels", torch.tensor(n_in_channels))
        self.register_buffer("n_out_channels", torch.tensor(n_out_channels))
        self.register_buffer("kernel_width", torch.tensor(kernel_width))
        self.register_buffer("kernel_height", torch.tensor(kernel_height))
        self.register_buffer("stride", torch.tensor(stride))
        self.register_buffer("padding", torch.tensor(padding))
        self.register_buffer("dilation", torch.tensor(dilation))

        self.weight = nn.Parameter(
            torch.empty(
                n_in_channels * kernel_width * kernel_height,
                n_out_channels,
                dtype=torch.double,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(1, n_out_channels, 1, 1, dtype=torch.double)
        )

        # self.bias = rearrange(self.bias, "out_channels -> 1 out_channels 1 1")

        if lecun_init:
            s = 2.4 / (n_in_channels * kernel_width * kernel_height)
            self.weight.data.uniform_(-s, s)
            self.bias.data.uniform_(-s, s)

        else:
            self.weight.data.normal_(0, 1.0)
            self.bias.data.normal_(0, 1.0)

        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_height, kernel_width),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        out_h = out_w = calc_conv_output_dim(
            edge, kernel_width, padding, stride
        )
        self.fold = torch.nn.Fold(
            output_size=(out_h, out_w),
            kernel_size=(1, 1),
            dilation=dilation,
            padding=0,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # inspired by: https://discuss.pytorch.org/t/make-custom-conv2d-layer-efficient-wrt-speed-and-memory/70175/2

        # (N,C,in_h,in_w) -> (N, C*kh*kw, num_patches)
        # N = batch_size, C = in_channels, kh = kernel_height, kw = kernel_width
        input_unfolded = self.unfold(input)

        input_unfolded = rearrange(
            input_unfolded, "N r num_patches -> N num_patches r"
        )

        output_unfolded = input_unfolded @ self.weight
        output_unfolded = rearrange(
            output_unfolded,
            "N num_patches out_channels -> N out_channels num_patches",
        )

        output = self.fold(output_unfolded)  # (N, out_channels, out_h, out_w)
        if self.bias is not None:
            output += self.bias

        return output


class Model(nn.Module):
    # based on LeCun et al. 1990, _Handwritten Digit Recognition: Applications of Neural Net Chips and Automatic Learning_, Neurocomputing, https://link.springer.com/chapter/10.1007/978-3-642-76153-9_35
    # inspired by https://einops.rocks/pytorch-examples.html
    def __init__(
        self,
        edge: int = 28,
        n_classes: int = 10,
        lecun_init: bool = True,
        lecun_act: bool = True,
        A: float = 1.716,
        S: float = 2 / 3,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            edge=edge,
            n_in_channels=1,
            n_out_channels=12,
            kernel_width=5,
            kernel_height=5,
            stride=2,
            padding=2,
            lecun_init=lecun_init,
        )
        edge = edge // 2  # effect of stride

        self.conv2 = Conv2d(
            edge=edge,
            n_in_channels=12,
            n_out_channels=12,
            kernel_width=5,
            kernel_height=5,
            stride=2,
            padding=2,
            lecun_init=lecun_init,
        )
        edge = edge // 2  # effect of stride
        self.lin1 = nn.Linear(edge * edge * 12, 30)
        self.lin2 = nn.Linear(30, n_classes)

        if lecun_init:
            s = 2.4 / self.lin1.weight.shape[0]
            self.lin1.weight.data.uniform_(-s, s)

            s = 2.4 / self.lin2.weight.shape[0]
            self.lin2.weight.data.uniform_(-s, s)

        if lecun_act:
            self.act_conv1 = Tanh(A, S)
            self.act_conv2 = Tanh(A, S)
            self.act_lin1 = Tanh(A, S)
            self.act_lin2 = Tanh(A, S)
        else:
            self.act_conv1 = nn.Tanh()
            self.act_conv2 = nn.Tanh()
            self.act_lin1 = nn.Tanh()
            self.act_lin2 = nn.Tanh()

        self.net = nn.Sequential(
            Rearrange("b h w -> b 1 h w"),
            self.conv1,
            self.act_conv1,
            self.conv2,
            self.act_conv2,
            Rearrange("b c h w -> b (c h w)"),
            self.lin1,
            self.act_lin1,
            self.lin2,
            self.act_lin2,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
