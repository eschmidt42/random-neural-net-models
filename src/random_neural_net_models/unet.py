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


def get_conv_pieces(
    num_features_in: int, num_features_out: int, kernel_size: int, stride: int
) -> T.Tuple[nn.Module, nn.Module, nn.Module]:
    """Batch norm, SiLU activation and conv2d layer for the unet's resnet blocks."""
    bn = nn.BatchNorm2d(num_features=num_features_in)
    act = nn.SiLU()
    padding = kernel_size // 2
    conv = nn.Conv2d(
        num_features_in,
        num_features_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return bn, act, conv


class ResBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        stride: int = 1,
        ks: int = 3,
    ):
        super().__init__()

        self.bn1, self.act1, self.conv1 = get_conv_pieces(
            num_features_in, num_features_out, ks, stride=1
        )
        self.bn2, self.act2, self.conv2 = get_conv_pieces(
            num_features_out, num_features_out, ks, stride=stride
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
            if num_features_in == num_features_out
            else nn.Conv2d(
                num_features_in, num_features_out, kernel_size=1, stride=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.convs(x)
        x_id = self.idconvs(x)
        return x_conv + x_id


class SaveModule:
    def forward(self, x, *args, **kwargs):
        self.saved_output = super().forward(x, *args, **kwargs)
        return self.saved_output


class SavedResBlock(SaveModule, ResBlock):
    pass


class SavedConv(SaveModule, nn.Conv2d):
    pass


def down_block(
    num_features_in: int,
    num_features_out: int,
    add_down: bool = True,
    num_layers: int = 1,
):
    """Sequence of resnet blocks with a downsample at the end, see stride."""
    res = nn.Sequential()
    for i in range(num_layers):
        ni = num_features_in if i == 0 else num_features_out
        res.append(
            SavedResBlock(
                num_features_in=ni, num_features_out=num_features_out, stride=1
            )
        )

    if add_down:
        res.append(
            SavedConv(
                num_features_out, num_features_out, 3, stride=2, padding=1
            )
        )
    return res


class UNetDown(nn.Module):
    def __init__(self, num_features: T.Tuple[int], num_layers: int) -> None:
        super().__init__()

        n_ins = [num_features[0]] + list(num_features[:-1])
        n_outs = [num_features[0]] + list(num_features[1:])
        add_downs = [True] * (len(num_features) - 1) + [False]

        self.downs = nn.Sequential(
            *[
                down_block(
                    n_in, n_out, add_down=add_down, num_layers=num_layers
                )
                for n_in, n_out, add_down in zip(n_ins, n_outs, add_downs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downs(x)

    def __iter__(self) -> torch.Tensor:
        for layer in self.downs:
            for sub_layer in layer:
                yield sub_layer.saved_output


class UpBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        prev_num_features_out: int,
        num_features_out: int,
        add_up: bool = True,
        num_layers: int = 2,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            ni = prev_num_features_out if i == 0 else num_features_out
            ni += num_features_out if (i < num_layers - 1) else num_features_in
            resnets.append(ResBlock(ni, num_features_out))

        self.resnets = nn.ModuleList(resnets)
        if add_up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    num_features_out, num_features_out, kernel_size=3, padding=1
                ),
            )
        else:
            self.up = nn.Identity()

    def forward(
        self, x_up: torch.Tensor, xs_down: T.List[torch.Tensor]
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x_down = xs_down.pop()
            x_glue = torch.cat([x_up, x_down], dim=1)
            x_up = resnet(x_glue)
        return self.up(x_up)


class UNetUp(nn.Module):
    def __init__(self, num_features: T.Tuple[int], num_layers: int) -> None:
        super().__init__()
        rev_nfs = list(reversed(num_features))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(len(num_features)):
            prev_nf = nf
            nf = rev_nfs[i]
            _ni = rev_nfs[min(i + 1, len(num_features) - 1)]
            add_up = i != len(num_features) - 1
            upblock = UpBlock(
                _ni, prev_nf, nf, add_up=add_up, num_layers=num_layers + 1
            )
            self.ups.append(upblock)

    def forward(
        self, x: torch.Tensor, saved: T.List[torch.Tensor]
    ) -> torch.Tensor:
        for upblock in self.ups:
            x = upblock(x, saved)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: T.Tuple[int] = (224, 448, 672, 896),
        num_layers: int = 1,
    ):
        super().__init__()

        self.setup_io(in_channels, out_channels)

        self.conv_in = nn.Conv2d(
            in_channels, num_features[0], kernel_size=3, padding=1
        )

        self.downs = UNetDown(num_features, num_layers=num_layers)

        self.mid_block = ResBlock(num_features[-1], num_features[-1])

        self.ups = UNetUp(num_features, num_layers=num_layers)

        # output
        self.bn_out, self.act_out, self.conv_out = get_conv_pieces(
            num_features[0], out_channels, kernel_size=1, stride=1
        )
        self.out = nn.Sequential(self.bn_out, self.act_out, self.conv_out)

    def setup_io(self, in_channels: int, out_channels: int):
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

        if out_channels == 1:
            self.rm_dim = Rearrange("b 1 h w -> b h w")
        else:
            self.rm_dim = nn.Identity()

        self.rm_padding = nn.ZeroPad2d(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add dimension and padding
        x = self.add_dim(x)
        x = self.add_padding(x)

        # input
        x = self.conv_in(x)
        saved = [x]

        # down projections
        x = self.downs(x)

        saved.extend([output for output in self.downs])

        x = self.mid_block(x)

        # up projections
        x = self.ups(x, saved)

        # output
        x = self.out(x)

        # remove padding and dimension
        x = self.rm_dim(x)
        x = self.rm_padding(x)

        return x
