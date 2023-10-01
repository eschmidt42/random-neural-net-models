# -*- coding: utf-8 -*-
# based on https://github.com/fastai/course22p2/blob/master/nbs/26_diffusion_unet.ipynb
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from einops.layers.torch import Rearrange

import random_neural_net_models.utils as utils

logger = utils.get_logger("resnet.py")


class ResBlock(nn.Module):
    def __init__(self, ni: int, nf: int, stride: int = 1, ks: int = 3):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(ni)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            ni, nf, kernel_size=ks, stride=1, padding=ks // 2
        )

        self.bn2 = nn.BatchNorm2d(nf)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            nf, nf, kernel_size=ks, stride=stride, padding=ks // 2
        )

        self.convs = nn.Sequential(
            self.bn1,
            self.act1,
            self.conv1,
            self.bn2,
            self.act2,
            self.conv2,
        )

        self.idconvs = (
            nn.Identity()
            if ni == nf
            else nn.Conv2d(ni, nf, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.convs(x)
        x_id = self.idconvs(x)
        return x_conv + x_id


class UpBlock(nn.Module):
    def __init__(
        self,
        ni: int,
        prev_nf: int,
        nf: int,
        add_up: bool = True,
        num_layers: int = 2,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            _ni = (prev_nf if i == 0 else nf) + (
                ni if (i == num_layers - 1) else nf
            )
            resnets.append(ResBlock(_ni, nf))
        self.resnets = nn.ModuleList(resnets)
        if add_up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            )
        else:
            self.up = nn.Identity()

    def forward(
        self, x: torch.Tensor, ups: T.List[torch.Tensor]
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x_down = ups.pop()
            # gluing the upsampling and the copied tensor together
            x = resnet(torch.cat([x, x_down], dim=1))
        return self.up(x)


class SaveModule:
    def forward(self, x, *args, **kwargs):
        self.saved = super().forward(x, *args, **kwargs)
        return self.saved


class SavedResBlock(SaveModule, ResBlock):
    pass


class SavedConv(SaveModule, nn.Conv2d):
    pass


def down_block(ni: int, nf: int, add_down: bool = True, num_layers: int = 1):
    res = nn.Sequential()
    for i in range(num_layers):
        _ni = ni if i == 0 else nf
        res.append(SavedResBlock(ni=_ni, nf=nf))

    if add_down:
        res.append(SavedConv(nf, nf, 3, stride=2, padding=1))
    return res


class UNet2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nfs=(224, 448, 672, 896),
        num_layers: int = 1,
    ):
        super().__init__()
        if in_channels != out_channels:
            logger.warning(
                f"in_channels ({in_channels}) != out_channels ({out_channels})"
            )

        # input
        if in_channels == 1:
            self.add_dim = Rearrange("b h w -> b 1 h w")
        else:
            self.add_dim = nn.Identity()
        self.add_padding = nn.ZeroPad2d(2)

        self.conv_in = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)

        # down block
        nf = nfs[0]
        self.downs = nn.Sequential()
        for i in range(len(nfs)):
            ni = nf
            nf = nfs[i]
            add_down = i != len(nfs) - 1
            self.downs.append(
                down_block(ni, nf, add_down=add_down, num_layers=num_layers)
            )

        # mid block
        self.mid_block = ResBlock(nfs[-1], nfs[-1])

        # up block
        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(len(nfs)):
            prev_nf = nf
            nf = rev_nfs[i]
            _ni = rev_nfs[min(i + 1, len(nfs) - 1)]
            add_up = i != len(nfs) - 1
            upblock = UpBlock(
                _ni, prev_nf, nf, add_up=add_up, num_layers=num_layers + 1
            )
            self.ups.append(upblock)

        # output
        self.bn_out = nn.BatchNorm2d(nfs[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(nfs[0], out_channels, kernel_size=1)
        self.out = nn.Sequential(self.bn_out, self.act_out, self.conv_out)

        if out_channels == 1:
            self.rm_dim = Rearrange("b 1 h w -> b h w")
        else:
            self.rm_dim = nn.Identity()

        self.rm_padding = nn.ZeroPad2d(-2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.add_dim(inp)

        x = self.add_padding(x)
        x = self.conv_in(x)
        saved = [x]

        # down projections
        x = self.downs(x)
        saved.extend([layer.saved for block in self.downs for layer in block])

        x = self.mid_block(x)

        # up projections
        for upblock in self.ups:
            x = upblock(x, saved)

        x = self.out(x)

        x = self.rm_dim(x)
        x = self.rm_padding(x)
        return x
