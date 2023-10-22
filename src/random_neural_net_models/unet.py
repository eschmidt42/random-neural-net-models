# -*- coding: utf-8 -*-
# based on https://github.com/fastai/course22p2/blob/master/nbs/26_diffusion_unet.ipynb
import typing as T

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

import random_neural_net_models.utils as utils

logger = utils.get_logger("unet.py")


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

        self.use_identity = num_features_in == num_features_out
        self.idconv = (
            nn.Identity()
            if self.use_identity
            else nn.Conv2d(
                num_features_in, num_features_out, kernel_size=1, stride=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.convs(x)
        x_id = self.idconv(x)
        return x_conv + x_id


class SaveModule:
    def forward(self, x, *args, **kwargs):
        self.saved_output = super().forward(x, *args, **kwargs)
        return self.saved_output


class SavedResBlock(SaveModule, ResBlock):
    pass


class DownBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        add_down: bool = True,
        num_resnet_layers: int = 1,
    ):
        """Sequence of resnet blocks with a downsample at the end, see stride."""
        super().__init__()

        self.add_down = add_down
        self.setup_res_blocks(
            num_features_in,
            num_features_out,
            num_resnet_layers=num_resnet_layers,
        )

        self.setup_downscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        num_features_out: int,
        num_resnet_layers: int = 2,
    ):
        self.res_blocks = nn.ModuleList()
        for i in range(num_resnet_layers - 1):
            n_in = num_features_in if i == 0 else num_features_out
            self.res_blocks.append(ResBlock(n_in, num_features_out))

        self.res_blocks.append(
            SavedResBlock(
                num_features_in=num_features_out,
                num_features_out=num_features_out,
            )
        )

    def setup_downscaling(self, num_features_out: int):
        if self.add_down:
            self.down = nn.Conv2d(
                num_features_out, num_features_out, 3, stride=2, padding=1
            )
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.down(x)

    @property
    def saved_output(self):
        return self.res_blocks[-1].saved_output


class UNetDown(nn.Module):
    def __init__(self, num_features: T.Tuple[int], num_layers: int) -> None:
        super().__init__()

        n_ins = [num_features[0]] + list(num_features[:-1])
        n_outs = [num_features[0]] + list(num_features[1:])
        add_downs = [True] * (len(num_features) - 1) + [False]

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    n_in, n_out, add_down=add_down, num_resnet_layers=num_layers
                )
                for n_in, n_out, add_down in zip(n_ins, n_outs, add_downs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down_block in self.down_blocks:
            x = down_block(x)
        return x

    def __iter__(self) -> torch.Tensor:
        for down_block in self.down_blocks:
            yield down_block.saved_output


class UpBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        prev_num_features_out: int,
        num_features_out: int,
        add_up: bool = True,
        num_resnet_layers: int = 2,
    ):
        super().__init__()
        self.add_up = add_up
        self.setup_res_blocks(
            num_features_in,
            prev_num_features_out,
            num_features_out,
            num_resnet_layers=num_resnet_layers,
        )

        self.setup_upscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        prev_num_features_out: int,
        num_output_features: int,
        num_resnet_layers: int = 2,
    ):
        self.res_blocks = nn.ModuleList()
        n_out = num_output_features

        for i in range(num_resnet_layers):
            if i == 0:
                n_in = prev_num_features_out
            else:
                n_in = n_out

            # handling unet copy
            if i == 0:
                n_in += num_features_in

            self.res_blocks.append(ResBlock(n_in, n_out))

    def setup_upscaling(self, num_features_out: int):
        if self.add_up:
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
        x_glue = torch.cat([x_up, xs_down.pop()], dim=1)
        x = self.res_blocks[0](x_glue)

        for res_block in self.res_blocks[1:]:
            x = res_block(x)

        if self.add_up:
            return self.up(x)
        else:
            return x


class UNetUp(nn.Module):
    def __init__(
        self,
        downs: UNetDown,
    ) -> None:
        super().__init__()

        self.ups = nn.ModuleList()
        n = len(downs.down_blocks)
        up_block = None

        for i, down_block in enumerate(reversed(downs.down_blocks)):
            is_final_layer = i == n - 1

            # 3 infos we need:
            # n_in_down: input features from parallel down block
            # n_in_prev_up: input features from previous up block
            # n_out_up: output features of current up block

            # n_in_down
            if not is_final_layer:  # res block
                down_out_conv = down_block.res_blocks[-1].convs[2]
            else:  # down conv
                down_out_conv = down_block.down

            n_in_down = down_out_conv.out_channels

            # n_in_prev_up
            if up_block is None:
                n_in_prev_up = n_in_down
            elif up_block.add_up:  # up conv
                n_in_prev_up = up_block.up[1].out_channels
            else:
                raise ValueError(f"unexpected case for {up_block=}")

            # n_out_up
            down_input_conv = down_block.res_blocks[0].convs[
                2
            ]  # (bn, act, conv)
            n_out_up = down_input_conv.in_channels

            add_up = not is_final_layer

            num_resnet_layers = len(down_block.res_blocks)

            up_block = UpBlock(
                n_in_down,
                n_in_prev_up,
                n_out_up,
                add_up=add_up,
                num_resnet_layers=num_resnet_layers,
            )
            self.ups.append(up_block)

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
        list_num_features: T.Tuple[int] = (8, 16),
        num_layers: int = 2,
    ):
        super().__init__()
        if in_channels != out_channels:
            logger.warning(
                f"in_channels ({in_channels}) != out_channels ({out_channels})"
            )

        self.setup_input(in_channels, list_num_features)

        self.downs = UNetDown(list_num_features, num_layers=num_layers)

        self.mid_block = ResBlock(list_num_features[-1], list_num_features[-1])

        self.ups = UNetUp(self.downs)

        self.setup_output(list_num_features, out_channels)

    def setup_input(self, in_channels: int, list_num_features: T.Tuple[int]):
        if in_channels == 1:
            self.add_dim = Rearrange("b h w -> b 1 h w")
        else:
            self.add_dim = nn.Identity()

        self.add_padding = nn.ZeroPad2d(2)
        self.conv_in = nn.Conv2d(
            in_channels, list_num_features[0], kernel_size=3, padding=1
        )
        self.wrangle_input = nn.Sequential(
            self.add_dim, self.add_padding, self.conv_in
        )

    def setup_output(self, list_num_features: T.Tuple[int], out_channels: int):
        self.bn_out, self.act_out, self.conv_out = get_conv_pieces(
            list_num_features[0], out_channels, kernel_size=1, stride=1
        )

        if out_channels == 1:
            self.rm_dim = Rearrange("b 1 h w -> b h w")
        else:
            self.rm_dim = nn.Identity()

        self.rm_padding = nn.ZeroPad2d(-2)

        self.wrangle_output = nn.Sequential(
            self.bn_out,
            self.act_out,
            self.conv_out,
            self.rm_dim,
            self.rm_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input
        x = self.wrangle_input(x)
        saved = [x]

        # down projections
        x = self.downs(x)

        # copy from down projections for up projections
        saved.extend([output for output in self.downs])

        x = self.mid_block(x)

        # up projections
        x = self.ups(x, saved)

        # output
        x = self.wrangle_output(x)

        return x
