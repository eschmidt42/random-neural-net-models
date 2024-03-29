# -*- coding: utf-8 -*-
import random_neural_net_models.tabular as rnnm_tab
import random_neural_net_models.data as rnnm_data
import torch.nn as nn
import torch
import typing as T
from tensordict import tensorclass
import torch.nn.functional as F
import copy


@tensorclass
class LatentOutput:
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


class Latent(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_batch_norm: bool) -> None:
        super().__init__()

        self.mu = nn.Linear(n_in, n_out)
        self.log_var = nn.Linear(n_in, n_out)

        if use_batch_norm:
            self.mu_bn_post = nn.BatchNorm1d(n_out)
            self.logvar_bn_post = nn.BatchNorm1d(n_out)
        else:
            self.mu_bn_post = nn.Identity()
            self.logvar_bn_post = nn.Identity()

    def forward(self, x: torch.Tensor) -> LatentOutput:
        mu = self.mu(x)
        log_var = self.log_var(x)
        mu = self.mu_bn_post(mu)
        log_var = self.logvar_bn_post(log_var)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return LatentOutput(z, mu, log_var, batch_size=[z.shape[0]])


class StandardNormalScalers(nn.Module):
    def __init__(self, means: T.Tuple[float], stds: T.Tuple[float]):
        super().__init__()

        self.register_buffer("means", torch.tensor(means).float())
        self.register_buffer("stds", torch.tensor(stds).float())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.stds + self.means


VAEOutput = T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TabularVariationalAutoEncoderNumerical(nn.Module):
    def __init__(
        self,
        n_hidden: T.Tuple[int],
        n_latent: int,
        means: T.Tuple[float],
        stds: T.Tuple[float],
        use_batch_norm: bool,
        do_impute: bool,
        impute_bias_source: rnnm_tab.BiasSources = rnnm_tab.BiasSources.zero,
        cols_with_missing: T.Tuple[int] = None,
    ):
        super().__init__()
        self.encoder = rnnm_tab.TabularModel(
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            do_impute=do_impute,
            impute_bias_source=impute_bias_source,
            cols_with_missing=cols_with_missing,
        )
        self.n_latent = n_latent
        self.latent = Latent(
            n_in=n_hidden[-1], n_out=n_latent, use_batch_norm=use_batch_norm
        )

        n_hidden_decoder = [n_latent] + list(reversed(n_hidden))

        self.decoder = rnnm_tab.TabularModel(
            n_hidden=n_hidden_decoder,
            use_batch_norm=use_batch_norm,
            do_impute=False,
            impute_bias_source=impute_bias_source,
        )
        self.scalers = StandardNormalScalers(means=means, stds=stds)

    def forward(self, input: rnnm_data.XBlock) -> VAEOutput:
        x_enc = self.encoder(input)

        latent_stuff: LatentOutput = self.latent(x_enc)

        z = rnnm_data.XBlock(
            x=latent_stuff.z, batch_size=[latent_stuff.z.shape[0]]
        )

        x_recon = self.decoder(z)
        x_recon = self.scalers(x_recon)
        return x_recon, latent_stuff.mu, latent_stuff.log_var

    def generate(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.n_latent)
        z = rnnm_data.XBlock(x=z, batch_size=[z.shape[0]])
        x = self.decoder(z)
        x = self.scalers(x)
        return x


class KullbackLeiblerNumericalOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, inference: VAEOutput, input: rnnm_data.XBlock
    ) -> torch.Tensor:
        x_recon, mu, log_var = inference
        recon_loss = nn.functional.mse_loss(x_recon, input.x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_div

        return loss


class SoftmaxForCategoricalColumns(nn.Module):
    def __init__(self, n_categories_per_column: T.Iterable[int]):
        super().__init__()

        self.index_ranges = rnnm_data.get_index_ranges_from_n_cats_per_col(
            n_categories_per_column
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.index_ranges:
            x[:, r] = F.softmax(x[:, r], dim=1)
        return x


VAEOutput_num_cat = T.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class TabularVariationalAutoEncoderNumericalAndCategorical(nn.Module):
    def __init__(
        self,
        n_hidden: T.Tuple[int],
        n_categories_per_column: T.Tuple[int],
        n_latent: int,
        means: T.Tuple[float],
        stds: T.Tuple[float],
        use_batch_norm: bool,
        do_impute: bool,
        impute_bias_source: rnnm_tab.BiasSources = rnnm_tab.BiasSources.zero,
        cols_with_missing: T.Tuple[int] = None,
    ):
        super().__init__()
        self.n_categories_per_column = n_categories_per_column
        self.encoder = rnnm_tab.TabularModelNumericalAndCategorical(
            n_hidden=n_hidden,
            n_categories_per_column=n_categories_per_column,
            use_batch_norm=use_batch_norm,
            do_impute=do_impute,
            impute_bias_source=impute_bias_source,
            cols_with_missing_num=cols_with_missing,
        )
        self.n_latent = n_latent
        self.latent = Latent(
            n_in=self.encoder.n_hidden[-1],
            n_out=n_latent,
            use_batch_norm=use_batch_norm,
        )

        n_hidden_decoder = [n_latent] + list(reversed(self.encoder.n_hidden))
        self.n_num_out = self.encoder.n_num_in
        self.n_cat_softmax_out = sum(n_categories_per_column)
        n_hidden_decoder[-1] = self.n_num_out + self.n_cat_softmax_out

        self.decoder = rnnm_tab.TabularModel(
            n_hidden=n_hidden_decoder,
            use_batch_norm=use_batch_norm,
            do_impute=False,
            impute_bias_source=impute_bias_source,
        )
        self.scalers = StandardNormalScalers(means=means, stds=stds)
        self.cats_softmax = SoftmaxForCategoricalColumns(
            n_categories_per_column
        )

    def forward(self, input: rnnm_data.XBlock_numcat) -> VAEOutput:
        x_enc = self.encoder(input)

        latent_stuff: LatentOutput = self.latent(x_enc)

        z = rnnm_data.XBlock(
            x=latent_stuff.z, batch_size=[latent_stuff.z.shape[0]]
        )

        x_recon = self.decoder(z)

        x_recon_num = x_recon[:, : self.n_num_out]
        x_recon_cat = x_recon[:, self.n_num_out :]

        x_recon_num = self.scalers(x_recon_num)
        x_recon_cat = self.cats_softmax(x_recon_cat)

        return x_recon_num, x_recon_cat, latent_stuff.mu, latent_stuff.log_var

    def generate(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.n_latent)
        z = rnnm_data.XBlock(x=z, batch_size=[z.shape[0]])
        x = self.decoder(z)
        x = self.scalers(x)
        return x


class KullbackLeiblerNumericalAndCategoricalLoss(nn.Module):
    def __init__(self, n_categories_per_column: T.Iterable[int]):
        super().__init__()
        self.index_ranges = rnnm_data.get_index_ranges_from_n_cats_per_col(
            n_categories_per_column
        )

    def forward(
        self, inference: VAEOutput_num_cat, input: rnnm_data.XBlock_numcat
    ) -> torch.Tensor:
        x_recon_num, x_recon_cat, mu, log_var = inference
        recon_loss_num = F.mse_loss(
            x_recon_num, input.x_numerical, reduction="sum"
        )

        n_cols = input.x_categorical.shape[1]
        if n_cols != len(self.index_ranges):
            msg = f"Number of columns in input ({n_cols}) does not match number of index ranges ({len(self.index_ranges)})"
            raise ValueError(msg)

        recon_loss_cat = torch.tensor(
            [
                F.cross_entropy(
                    x_recon_cat[:, self.index_ranges[i]],
                    input.x_categorical[:, i],
                )
                for i in range(n_cols)
            ]
        ).mean()

        recon_loss = recon_loss_num + recon_loss_cat

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + kl_div

        return loss


def transform_X_cat_probs_to_classes(
    X_cat_probs: torch.Tensor, n_categories_per_column: T.Iterable[int]
) -> torch.Tensor:
    index_ranges = rnnm_data.get_index_ranges_from_n_cats_per_col(
        n_categories_per_column
    )
    X_cat = torch.stack(
        [X_cat_probs[:, r].argmax(dim=1) for r in index_ranges], dim=1
    )
    return X_cat


def freeze_weights(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


class TabularModelReusingTrainedEncoder(nn.Module):
    def __init__(
        self,
        pretrained_encoder: rnnm_tab.TabularModel,
        n_out: int,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()

        self.pretrained_encoder = copy.deepcopy(pretrained_encoder)
        freeze_weights(self.pretrained_encoder)

        self.output_layer = rnnm_tab.Layer(
            n_in=pretrained_encoder.n_hidden[-1],
            n_out=n_out,
            use_batch_norm=use_batch_norm,
            use_activation=False,
        )

        self.net = nn.Sequential(self.pretrained_encoder, self.output_layer)

    def forward(self, input: rnnm_data.XyBlock) -> torch.Tensor:
        return self.net(input)
