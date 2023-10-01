# -*- coding: utf-8 -*-
# based on https://github.com/fastai/course22p2/blob/master/nbs/13_resnet.ipynb
import typing as T

import torch
import torch.nn as nn
import torchinfo
from einops.layers.torch import Rearrange

import random_neural_net_models.utils as utils

logger = utils.get_logger("resnet.py")


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            ni, nf, kernel_size=ks, stride=1, padding=ks // 2
        )
        self.bn1 = nn.BatchNorm2d(nf)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            nf, nf, kernel_size=ks, stride=stride, padding=ks // 2
        )
        self.bn2 = nn.BatchNorm2d(nf)

        self.convs = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act1,
            self.conv2,
            self.bn2,
        )

        self.idconv = (
            nn.Identity()
            if ni == nf
            else nn.Conv2d(ni, nf, kernel_size=1, stride=1)
        )
        self.pool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        )
        self.idconvs = nn.Sequential(self.idconv, self.pool)
        self.act_out = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x_conv = self.convs(x)
        x_id = self.idconvs(x)
        return self.act_out(x_conv + x_id)


class ResNet(nn.Module):
    def __init__(
        self,
        nfs: T.Tuple[int] = (8, 16, 32, 64, 128, 256),
        ks: int = 3,
        n_classes: int = 10,
    ):
        super().__init__()

        layers = [ResBlock(1, nfs[0], stride=1, ks=ks)]
        for ni, nf in zip(nfs[:-1], nfs[1:]):
            layers.append(ResBlock(ni, nf, stride=2, ks=ks))

        self.resblocks = nn.Sequential(*layers)
        resblock_info = torchinfo.summary(self.resblocks, (1, 1, 28, 28))

        _, c, h, w = resblock_info.summary_list[0].output_size
        n_lin_in = c * h * w
        self.lin = nn.Linear(n_lin_in, n_classes)

        self.bn1 = nn.BatchNorm1d(n_classes)
        self.add_dim = Rearrange("b h w -> b 1 h w")
        self.flatten = Rearrange("b c h w -> b (c h w)")

        self.net = nn.Sequential(
            self.add_dim, self.resblocks, self.flatten, self.lin, self.bn1
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
