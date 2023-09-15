# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class DeConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.upscale = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.net = nn.Sequential(self.upscale, self.conv)

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    # based on https://avandekleut.github.io/vae/
    def __init__(self, ks: int = 3):
        super(CNNEncoder, self).__init__()

        stride = 2
        padding = ks // 2
        self.add_dim = Rearrange("b h w -> b 1 h w")
        self.add_padding = nn.ZeroPad2d(2)
        self.enc_conv1 = nn.Conv2d(
            1, 2, kernel_size=ks, stride=stride, padding=padding
        )
        self.enc_act1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(
            2, 4, kernel_size=ks, stride=stride, padding=padding
        )
        self.enc_act2 = nn.ReLU()

        self.encoder = nn.Sequential(
            self.add_dim,  # 28x28 -> 1x28x28
            self.add_padding,  # 1x28x28 -> 1x32x32
            self.enc_conv1,  # 1x32x32 -> 1x16x16
            self.enc_act1,
            self.enc_conv2,  # 1x16x16 -> 1x8x8
            self.enc_act2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z


class CNNDecoder(nn.Module):
    def __init__(self, ks: int = 3):
        super(CNNDecoder, self).__init__()
        self.dec_deconv1 = DeConv2d(4, 2, kernel_size=ks, stride=1)
        self.dec_act1 = nn.ReLU()
        self.dec_deconv2 = DeConv2d(2, 1, kernel_size=ks, stride=1)
        self.dec_act2 = nn.Sigmoid()
        self.rm_padding = nn.ZeroPad2d(-2)
        self.rm_dim = Rearrange("b 1 h w -> b h w")

        self.decoder = nn.Sequential(
            self.dec_deconv1,  # 1x8x8 -> 1x16x16
            self.dec_act1,
            self.dec_deconv2,  # 1x16x16 -> 1x32x32
            self.rm_padding,  # 1x32x32 -> 1x28x28
            self.dec_act2,
            self.rm_dim,  # 1x28x28 -> 28x28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        return x_hat


class CNNAutoEncoder(nn.Module):
    def __init__(self, ks: int = 3) -> None:
        super(CNNAutoEncoder, self).__init__()
        self.encoder = CNNEncoder(ks=ks)
        self.decoder = CNNDecoder(ks=ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# TODO: remove once refactoring completed
class Model(nn.Module):
    # https://github.com/fastai/course22p2/blob/master/nbs/08_autoencoder.ipynb
    def __init__(self):
        super(Model, self).__init__()
        ks = 3
        stride = 2
        padding = ks // 2
        self.add_dim = Rearrange("b h w -> b 1 h w")
        self.add_padding = nn.ZeroPad2d(2)
        self.enc_conv1 = nn.Conv2d(
            1, 2, kernel_size=ks, stride=stride, padding=padding
        )
        self.enc_act1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(
            2, 4, kernel_size=ks, stride=stride, padding=padding
        )
        self.enc_act2 = nn.ReLU()

        self.encoder = nn.Sequential(
            self.add_dim,  # 28x28 -> 1x28x28
            self.add_padding,  # 1x28x28 -> 1x32x32
            self.enc_conv1,  # 1x32x32 -> 1x16x16
            self.enc_act1,
            self.enc_conv2,  # 1x16x16 -> 1x8x8
            self.enc_act2,
        )

        self.dec_deconv1 = DeConv2d(4, 2, kernel_size=ks, stride=1)
        self.dec_act1 = nn.ReLU()
        self.dec_deconv2 = DeConv2d(2, 1, kernel_size=ks, stride=1)
        self.dec_act2 = nn.Sigmoid()
        self.rm_padding = nn.ZeroPad2d(-2)
        self.rm_dim = Rearrange("b 1 h w -> b h w")

        self.decoder = nn.Sequential(
            self.dec_deconv1,  # 1x8x8 -> 1x16x16
            self.dec_act1,
            self.dec_deconv2,  # 1x16x16 -> 1x32x32
            self.rm_padding,  # 1x32x32 -> 1x28x28
            self.dec_act2,
            self.rm_dim,  # 1x28x28 -> 28x28
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat
