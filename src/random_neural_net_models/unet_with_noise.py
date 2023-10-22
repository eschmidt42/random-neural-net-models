# -*- coding: utf-8 -*-
# based on https://github.com/fastai/course22p2/blob/master/nbs/26_diffusion_unet.ipynb
import math
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

import random_neural_net_models.unet as unet
import random_neural_net_models.utils as utils

logger = utils.get_logger("unet_with_noise.py")


def get_noise_level_embedding(
    noise_levels: torch.Tensor, emb_dim: int, max_period: int = 10_000
) -> torch.Tensor:
    x = torch.linspace(0, 1, emb_dim // 2, device=noise_levels.device)
    exponent = -math.log(max_period) * x

    noise_levels_ = rearrange(noise_levels, "b -> b 1")
    exponent = rearrange(exponent, "d -> 1 d")

    emb = noise_levels_.double() * exponent.exp()  # (batch_size, emb_dim//2)

    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (batch_size, emb_dim)

    if emb_dim % 2 == 1:
        return F.pad(emb, (0, 1, 0, 0))
    else:
        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        num_emb: int,
        stride: int = 1,
        ks: int = 3,
    ):
        super().__init__()

        self.emb_act = nn.SiLU()
        self.emb_dense = nn.Linear(num_emb, num_features_out * 2)
        self.emb_proj = nn.Sequential(
            self.emb_act,
            self.emb_dense,
            Rearrange("b n -> b n 1 1"),
        )

        self.bn1, self.act1, self.conv1 = unet.get_conv_pieces(
            num_features_in, num_features_out, ks, stride=1
        )
        self.bn2, self.act2, self.conv2 = unet.get_conv_pieces(
            num_features_out, num_features_out, ks, stride=stride
        )

        self.conv1 = nn.Sequential(
            self.bn1,
            self.act1,
            self.conv1,
        )
        self.conv2 = nn.Sequential(
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.emb_proj(t)

        scale, shift = emb.chunk(2, dim=1)

        x_conv = self.conv1(x)
        x_conv = x_conv * (1 + scale) + shift
        x_conv = self.conv2(x_conv)

        x_id = self.idconv(x)
        return x_conv + x_id


class SavedResBlock(unet.SaveModule, ResBlock):
    pass


class DownBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        num_emb: int,
        add_down: bool = True,
        num_resnet_layers: int = 1,
    ):
        """Sequence of resnet blocks with a downsample at the end, see stride."""
        super().__init__()

        self.add_down = add_down
        self.setup_res_blocks(
            num_features_in,
            num_features_out,
            num_emb,
            num_resnet_layers=num_resnet_layers,
        )

        self.setup_downscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        num_features_out: int,
        num_emb: int,
        num_resnet_layers: int = 2,
    ):
        self.res_blocks = nn.ModuleList()
        for i in range(num_resnet_layers - 1):
            n_in = num_features_in if i == 0 else num_features_out
            self.res_blocks.append(ResBlock(n_in, num_features_out, num_emb))

        self.res_blocks.append(
            SavedResBlock(
                num_features_in=num_features_out,
                num_features_out=num_features_out,
                num_emb=num_emb,
            )
        )

    def setup_downscaling(self, num_features_out: int):
        if self.add_down:
            self.down = nn.Conv2d(
                num_features_out, num_features_out, 3, stride=2, padding=1
            )
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, t)
        return self.down(x)

    @property
    def saved_output(self):
        return self.res_blocks[-1].saved_output


class UNetDown(nn.Module):
    def __init__(
        self, num_features: T.Tuple[int], num_layers: int, num_emb: int
    ) -> None:
        super().__init__()

        n_ins = [num_features[0]] + list(num_features[:-1])
        n_outs = [num_features[0]] + list(num_features[1:])
        add_downs = [True] * (len(num_features) - 1) + [False]

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    n_in,
                    n_out,
                    num_emb,
                    add_down=add_down,
                    num_resnet_layers=num_layers,
                )
                for n_in, n_out, add_down in zip(n_ins, n_outs, add_downs)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for down_block in self.down_blocks:
            x = down_block(x, t)
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
        num_emb: int,
        add_up: bool = True,
        num_resnet_layers: int = 2,
    ):
        super().__init__()
        self.add_up = add_up
        self.setup_res_blocks(
            num_features_in,
            prev_num_features_out,
            num_features_out,
            num_emb,
            num_resnet_layers=num_resnet_layers,
        )

        self.setup_upscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        prev_num_features_out: int,
        num_output_features: int,
        num_emb: int,
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

            self.res_blocks.append(ResBlock(n_in, n_out, num_emb))

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
        self, x_up: torch.Tensor, xs_down: T.List[torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        x_glue = torch.cat([x_up, xs_down.pop()], dim=1)
        x = self.res_blocks[0](x_glue, t)

        for res_block in self.res_blocks[1:]:
            x = res_block(x, t)

        if self.add_up:
            return self.up(x)
        else:
            return x


class UNetUp(nn.Module):
    def __init__(
        self,
        downs: UNetDown,
        num_emb: int,
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
                down_out_conv = down_block.res_blocks[-1].conv1[2]
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
            down_input_conv = down_block.res_blocks[0].conv1[
                2
            ]  # (bn, act, conv)
            n_out_up = down_input_conv.in_channels

            add_up = not is_final_layer

            num_resnet_layers = len(down_block.res_blocks)

            up_block = UpBlock(
                n_in_down,
                n_in_prev_up,
                n_out_up,
                num_emb,
                add_up=add_up,
                num_resnet_layers=num_resnet_layers,
            )
            self.ups.append(up_block)

    def forward(
        self, x: torch.Tensor, saved: T.List[torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        for upblock in self.ups:
            x = upblock(x, saved, t)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        list_num_features: T.Tuple[int] = (8, 16),
        num_layers: int = 2,
        max_emb_period: int = 10000,
    ):
        super().__init__()
        if in_channels != out_channels:
            logger.warning(
                f"in_channels ({in_channels}) != out_channels ({out_channels})"
            )

        self.max_emb_period = max_emb_period
        self.n_noise_level_input = list_num_features[0]
        self.n_noise_level_emb = self.n_noise_level_input * 4
        self.setup_embedding_projection()

        self.setup_input(in_channels, list_num_features)

        self.downs = UNetDown(
            list_num_features, num_layers, self.n_noise_level_emb
        )

        self.mid_block = ResBlock(
            list_num_features[-1], list_num_features[-1], self.n_noise_level_emb
        )

        self.ups = UNetUp(self.downs, self.n_noise_level_emb)

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

    def setup_embedding_projection(self):
        self.emb_bn = nn.BatchNorm1d(self.n_noise_level_input)
        self.emb_dense1 = nn.Linear(
            self.n_noise_level_input, self.n_noise_level_emb
        )
        self.emb_dense2 = nn.Linear(
            self.n_noise_level_emb, self.n_noise_level_emb
        )

        self.emb_mlp = nn.Sequential(
            self.emb_bn,
            nn.SiLU(),
            self.emb_dense1,
            nn.SiLU(),
            self.emb_dense2,
        )

    def setup_output(self, list_num_features: T.Tuple[int], out_channels: int):
        self.bn_out, self.act_out, self.conv_out = unet.get_conv_pieces(
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

    def forward(
        self, imgs: torch.Tensor, noise_levels: torch.Tensor
    ) -> torch.Tensor:
        # input image
        x = self.wrangle_input(imgs)
        saved = [x]

        # input noise level
        noise_emb = get_noise_level_embedding(
            noise_levels,
            self.n_noise_level_input,
            max_period=self.max_emb_period,
        )
        noise_emb = self.emb_mlp(noise_emb)

        # down projections
        x = self.downs(x, noise_emb)

        # copy from down projections for up projections
        saved.extend([output for output in self.downs])

        x = self.mid_block(x, noise_emb)

        # up projections
        x = self.ups(x, saved, noise_emb)

        # output
        x = self.wrangle_output(x)

        return x
