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
    def __init__(self, ks: int = 3, post_bn: bool = False):
        super(CNNEncoder, self).__init__()

        stride = 2
        padding = ks // 2
        self.add_dim = Rearrange("b h w -> b 1 h w")
        self.add_padding = nn.ZeroPad2d(2)
        self.enc_conv1 = nn.Conv2d(
            1, 2, kernel_size=ks, stride=stride, padding=padding
        )
        nn.init.kaiming_normal_(self.enc_conv1.weight, nonlinearity="relu")
        self.enc_act1 = nn.ReLU()

        self.enc_conv2 = nn.Conv2d(
            2, 4, kernel_size=ks, stride=stride, padding=padding
        )
        nn.init.kaiming_normal_(self.enc_conv2.weight, nonlinearity="relu")
        self.enc_act2 = nn.ReLU()

        if post_bn:
            self.enc_bn1 = nn.BatchNorm2d(num_features=2)
            self.enc_bn2 = nn.BatchNorm2d(num_features=4)
        else:
            self.enc_bn1 = nn.Identity()
            self.enc_bn2 = nn.Identity()

        self.encoder = nn.Sequential(
            self.add_dim,  # 28x28 -> 1x28x28
            self.add_padding,  # 1x28x28 -> 1x32x32
            self.enc_conv1,  # 1x32x32 -> 1x16x16
            self.enc_act1,
            self.enc_bn1,
            self.enc_conv2,  # 1x16x16 -> 1x8x8
            self.enc_act2,
            self.enc_bn2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z


class CNNDecoder(nn.Module):
    def __init__(self, ks: int = 3, post_bn: bool = False):
        super(CNNDecoder, self).__init__()

        self.dec_deconv1 = DeConv2d(4, 2, kernel_size=ks, stride=1)
        nn.init.kaiming_normal_(self.dec_deconv1.weight, nonlinearity="relu")
        self.dec_act1 = nn.ReLU()

        self.dec_deconv2 = DeConv2d(2, 1, kernel_size=ks, stride=1)
        nn.init.kaiming_normal_(self.dec_deconv2.weight, nonlinearity="relu")
        # self.dec_act2 = nn.Sigmoid()
        self.rm_padding = nn.ZeroPad2d(-2)
        self.rm_dim = Rearrange("b 1 h w -> b h w")

        if post_bn:
            self.dec_bn1 = nn.BatchNorm2d(2)
            # self.dec_bn2 = nn.BatchNorm2d(1)
        else:
            self.dec_bn1 = nn.Identity()
            # self.dec_bn2 = nn.Identity()

        self.decoder = nn.Sequential(
            self.dec_deconv1,  # 1x8x8 -> 1x16x16
            self.dec_act1,
            self.dec_bn1,
            self.dec_deconv2,  # 1x16x16 -> 1x32x32
            self.rm_padding,  # 1x32x32 -> 1x28x28
            # self.dec_act2,
            # self.dec_bn2,
            self.rm_dim,  # 1x28x28 -> 28x28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        return x_hat


class DenseEncoder(nn.Module):
    # based on https://avandekleut.github.io/vae/
    def __init__(
        self,
        n_in: int,
        n_hidden1: int,
        n_hidden2: int,
        post_bn: bool = False,
        flatten_input: bool = False,
    ):
        super(DenseEncoder, self).__init__()

        self.enc_dense1 = nn.Linear(n_in, n_hidden1)
        nn.init.kaiming_normal_(self.enc_dense1.weight, nonlinearity="relu")
        self.enc_act1 = nn.ReLU()

        self.enc_dense2 = nn.Linear(n_hidden1, n_hidden2)
        nn.init.kaiming_normal_(self.enc_dense2.weight, nonlinearity="relu")
        self.enc_act2 = nn.ReLU()

        if post_bn:
            self.enc_bn1 = nn.BatchNorm1d(num_features=n_hidden1)
            self.enc_bn2 = nn.BatchNorm1d(num_features=n_hidden2)
        else:
            self.enc_bn1 = nn.Identity()
            self.enc_bn2 = nn.Identity()

        if flatten_input:
            self.rectangle2flat = Rearrange("b h w -> b (h w)")
        else:
            self.rectangle2flat = nn.Identity()

        self.encoder = nn.Sequential(
            self.rectangle2flat,
            self.enc_dense1,
            self.enc_act1,
            self.enc_bn1,
            self.enc_dense2,
            self.enc_act2,
            self.enc_bn2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z


class DenseDecoder(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_hidden1: int,
        n_hidden2: int,
        n_out: int,
        post_bn: bool = False,
        unflatten_output: bool = False,
        w: int = None,
        h: int = None,
    ):
        super(DenseDecoder, self).__init__()
        self.dec_dense1 = nn.Linear(n_latent, n_hidden1)
        nn.init.kaiming_normal_(self.dec_dense1.weight, nonlinearity="relu")
        self.dec_act1 = nn.ReLU()

        self.dec_dense2 = nn.Linear(n_hidden1, n_hidden2)
        nn.init.kaiming_normal_(self.dec_dense2.weight, nonlinearity="relu")
        self.dec_act2 = nn.ReLU()

        self.dec_dense3 = nn.Linear(n_hidden2, n_out)
        # self.dec_act3 = nn.Sigmoid()

        if post_bn:
            self.dec_bn1 = nn.BatchNorm1d(n_hidden1)
            self.dec_bn2 = nn.BatchNorm1d(n_hidden2)
        else:
            self.dec_bn1 = nn.Identity()
            self.dec_bn2 = nn.Identity()

        if unflatten_output:
            if w is None or h is None:
                raise ValueError(
                    f"If unflatten_output is True, w and h must be specified, got {w=}, {h=}."
                )
            if w * h != n_out:
                raise ValueError(
                    f"If unflatten_output is True, w * h must equal n_out, got {w=}, {h=}, {n_out=}."
                )
            self.flat2rectangle = Rearrange("b (h w) -> b h w", h=h, w=w)
        else:
            self.flat2rectangle = nn.Identity()

        self.decoder = nn.Sequential(
            self.dec_dense1,
            self.dec_act1,
            self.dec_bn1,
            self.dec_dense2,
            self.dec_act2,
            self.dec_bn2,
            self.dec_dense3,
            # self.dec_act3,
            self.flat2rectangle,
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
