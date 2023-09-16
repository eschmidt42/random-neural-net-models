# -*- coding: utf-8 -*-
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import random_neural_net_models.autoencoder_fastai2022 as ae


class Variational(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_latent: int,
        flatten_input: bool = True,
        unflatten_output: bool = True,
        c: int = None,
        w: int = None,
        h: int = None,
        post_bn: bool = True,
    ) -> None:
        super(Variational, self).__init__()

        self.mu = nn.Linear(n_in, n_latent)
        self.logvar = nn.Linear(n_in, n_latent)

        if post_bn:
            self.mu_bn_post = nn.BatchNorm1d(n_latent)
            self.logvar_bn_post = nn.BatchNorm1d(n_latent)
        else:
            self.mu_bn_post = nn.Identity()
            self.logvar_bn_post = nn.Identity()

        if flatten_input:
            self.rectangle2flat = Rearrange("b c h w -> b (c h w)")
        else:
            self.rectangle2flat = nn.Identity()

        if unflatten_output:
            if c is None or w is None or h is None:
                raise ValueError(
                    f"If unflatten_output is True, c, w, h must be specified, got {c=}, {w=}, {h=}."
                )
            self.flat2rectangle = Rearrange(
                "b (c h w) -> b c h w", c=c, h=h, w=w
            )
        else:
            self.flat2rectangle = nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.rectangle2flat(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        mu = self.mu_bn_post(mu)
        logvar = self.logvar_bn_post(logvar)

        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)

        z = mu + eps * std

        z = self.flat2rectangle(z)

        return z, mu, logvar


class CNNVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        ks: int = 3,
        n_latent: int = 4 * 8 * 8,
        post_bn_encoder: bool = True,
        post_bn_variational: bool = True,
        post_bn_decoder: bool = True,
    ) -> None:
        super(CNNVariationalAutoEncoder, self).__init__()
        self.encoder = ae.CNNEncoder(ks=ks, post_bn=post_bn_encoder)
        n_enc = 4 * 8 * 8
        self.variational = Variational(
            n_in=n_enc,
            n_latent=n_latent,
            flatten_input=True,
            unflatten_output=True,
            c=4,
            w=8,
            h=8,
            post_bn=post_bn_variational,
        )
        self.decoder = ae.CNNDecoder(ks=ks, post_bn=post_bn_decoder)

    def forward(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        z, mu, logvar = self.variational(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class DenseVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        w: int,
        h: int,
        n_latent: int = 200,
        post_bn_encoder: bool = True,
        post_bn_variational: bool = True,
        post_bn_decoder: bool = True,
    ) -> None:
        super(DenseVariationalAutoEncoder, self).__init__()
        n_in = w * h
        n_enc_hidden1 = 400
        n_enc_hidden2 = 200
        self.encoder = ae.DenseEncoder(
            n_in=n_in,
            n_hidden1=n_enc_hidden1,
            n_hidden2=n_enc_hidden2,
            post_bn=post_bn_encoder,
            flatten_input=True,
        )

        self.variational = Variational(
            n_in=n_enc_hidden2,
            n_latent=n_latent,
            flatten_input=False,
            unflatten_output=False,
            post_bn=post_bn_variational,
        )

        n_dec_hidden1 = 200
        n_dec_hidden2 = 400

        self.decoder = ae.DenseDecoder(
            n_latent=n_latent,
            n_hidden1=n_dec_hidden1,
            n_hidden2=n_dec_hidden2,
            n_out=n_in,
            post_bn=post_bn_decoder,
            unflatten_output=True,
            w=w,
            h=h,
        )

    def forward(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        z, mu, logvar = self.variational(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class CNNDenseVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        w: int,
        h: int,
        ks: int = 3,
        n_latent: int = 200,
        post_bn_encoder: bool = True,
        post_bn_variational: bool = True,
        post_bn_decoder: bool = True,
    ) -> None:
        super(CNNDenseVariationalAutoEncoder, self).__init__()
        n_in = w * h

        self.encoder = ae.CNNEncoder(ks=ks, post_bn=post_bn_encoder)

        n_enc = 4 * 8 * 8
        self.variational = Variational(
            n_in=n_enc,
            n_latent=n_latent,
            flatten_input=True,
            unflatten_output=False,
            c=4,
            w=8,
            h=8,
            post_bn=post_bn_variational,
        )

        n_dec_hidden1 = 200
        n_dec_hidden2 = 400

        self.decoder = ae.DenseDecoder(
            n_latent=n_latent,
            n_hidden1=n_dec_hidden1,
            n_hidden2=n_dec_hidden2,
            n_out=n_in,
            post_bn=post_bn_decoder,
            unflatten_output=True,
            w=w,
            h=h,
        )

    def forward(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        z, mu, logvar = self.variational(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def calc_distribution_divergence_loss(
    input: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    _, mu, logvar = input
    s = 1 + logvar - mu.pow(2) - logvar.exp()
    return -0.5 * s.mean()


def calc_reconstruction_loss(
    input: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    is_sigmoid: bool = False,
) -> torch.Tensor:
    x_hat, _, _ = input
    if is_sigmoid:
        return F.mse_loss(x, x_hat)
    else:
        return F.binary_cross_entropy_with_logits(x_hat, x)


def calc_vae_loss(
    input: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    is_sigmoid: bool = False,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = calc_reconstruction_loss(
        input, x, is_sigmoid=is_sigmoid
    )
    divergence_loss = calc_distribution_divergence_loss(input, x)
    total_loss = reconstruction_loss + divergence_loss
    return total_loss, reconstruction_loss, divergence_loss


def calc_vae_test_loss(
    model_output: T.List[T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    is_sigmoid: bool = False,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_hat = torch.cat([_x[0] for _x in model_output], dim=0)
    mu = torch.cat([_x[1] for _x in model_output], dim=0)
    logvar = torch.cat([_x[2] for _x in model_output], dim=0)
    _model_output = (x_hat, mu, logvar)
    reconstruction_loss = calc_reconstruction_loss(
        _model_output, x, is_sigmoid=is_sigmoid
    )
    divergence_loss = calc_distribution_divergence_loss(_model_output, x)
    total_loss = reconstruction_loss + divergence_loss
    return total_loss, reconstruction_loss, divergence_loss


import re


def check_module_name_is_activation(name: str) -> bool:
    return re.match(r".*act\d$", name) is not None


def check_module_name_grad_relevant(name: str) -> bool:
    is_enc_relevant = re.match(r"^enc_(bn|conv|dense)", name) is not None
    is_var_relevant = re.match(r"^mu|logvar|bn_post", name) is not None
    is_dec_relevant = re.match(r"^dec_(bn|deconv|dense)", name) is not None
    is_not_act = re.match(r".*act\d$", name) is None
    return (
        is_enc_relevant or is_var_relevant or is_dec_relevant
    ) and is_not_act
