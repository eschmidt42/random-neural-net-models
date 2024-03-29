# -*- coding: utf-8 -*-
# based on https://github.com/fastai/course22p2/blob/master/nbs/26_diffusion_unet.ipynb
import math
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as torch_loss
import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import tensorclass
from torch.utils.data import DataLoader

import random_neural_net_models.data as rnnm_data
import random_neural_net_models.learner as rnnm_learner
import random_neural_net_models.telemetry as telemetry
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

    emb = noise_levels_ * exponent.exp()  # (batch_size, emb_dim//2)

    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (batch_size, emb_dim)

    if emb_dim % 2 == 1:
        return F.pad(emb, (0, 1, 0, 0)).float()
    else:
        return emb.float()


class Attention2D(nn.Module):
    # based on https://github.com/fastai/course22p2/blob/master/nbs/28_diffusion-attn-cond.ipynb
    def __init__(self, n_cnn_channels: int, n_channels_per_head: int):
        super().__init__()
        assert (
            n_channels_per_head % n_channels_per_head == 0
        ), f"{n_channels_per_head=} must be a multiple of {n_cnn_channels=}"
        self.n_heads = n_cnn_channels // n_channels_per_head
        logger.info(
            f"Attention for {n_cnn_channels=} with {n_channels_per_head=} -> {self.n_heads=}"
        )
        self.scale = math.sqrt(n_cnn_channels / self.n_heads)
        self.norm = nn.LayerNorm(n_cnn_channels)
        self.qkv = nn.Linear(n_cnn_channels, n_cnn_channels * 3)
        self.proj = nn.Linear(n_cnn_channels, n_cnn_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO: replace @ with einops.einsum
        _, _, h, w = X.shape

        # (h d) = latent dim * 3
        X = rearrange(X, "batch channels h w -> batch channels (h w)")
        # X = rearrange(X, 'batch channels (heads d) -> (batch heads) channels d', h=self.nheads)

        X = rearrange(X, "batch channels hw -> batch hw channels")
        X = self.norm(X)
        X = self.qkv(X)
        X = rearrange(
            X,
            "batch hw (heads latent) -> (batch heads) hw latent",
            heads=self.n_heads,
        )
        Q, K, V = torch.chunk(X, 3, dim=-1)  # each "(batch heads) hw latent"
        Ktrans = rearrange(
            K,
            "(batch heads) hw latent -> (batch heads) latent hw",
            heads=self.n_heads,
        )
        S = (Q @ Ktrans) / self.scale  # "(batch heads) hw hw"
        X = S.softmax(dim=-1) @ V  # "(batch heads) hw latent"
        X = rearrange(
            X,
            "(batch heads) hw latent -> batch hw (heads latent)",
            heads=self.n_heads,
        )
        X = self.proj(X)  # "batch hw (heads latent)"
        X = rearrange(X, "batch (h w) channels -> batch channels h w", h=h, w=w)

        return X


# TODO - CONTINUE HERE: add Attention2D to ResBlock
class ResBlock(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_features_out: int,
        num_emb: int,
        stride: int = 1,
        ks: int = 3,
        n_channels_per_head: int = 0,
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
        self.attention = (
            nn.Identity()
            if n_channels_per_head == 0
            else Attention2D(num_features_out, n_channels_per_head)
        )

    # TODO: implement x + self.attention(x) at the end of the function
    # using transformer.py's AttentionBlock
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.emb_proj(t)

        scale, shift = emb.chunk(2, dim=1)

        x_conv = self.conv1(x)
        x_conv = x_conv * (1 + scale) + shift
        x_conv = self.conv2(x_conv)

        x_id = self.idconv(x)
        x_skip1 = x_conv + x_id
        x_att = self.attention(x_skip1)
        x_skip2 = x_skip1 + x_att
        return x_skip2


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
        n_channels_per_head: int = 0,
    ):
        """Sequence of resnet blocks with a downsample at the end, see stride."""
        super().__init__()

        self.add_down = add_down
        self.setup_res_blocks(
            num_features_in,
            num_features_out,
            num_emb,
            num_resnet_layers=num_resnet_layers,
            n_channels_per_head=n_channels_per_head,
        )

        self.setup_downscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        num_features_out: int,
        num_emb: int,
        num_resnet_layers: int = 2,
        n_channels_per_head: int = 0,
    ):
        self.res_blocks = nn.ModuleList()
        for i in range(num_resnet_layers - 1):
            n_in = num_features_in if i == 0 else num_features_out
            self.res_blocks.append(
                ResBlock(
                    n_in,
                    num_features_out,
                    num_emb,
                    n_channels_per_head=n_channels_per_head,
                )
            )

        self.res_blocks.append(
            SavedResBlock(
                num_features_in=num_features_out,
                num_features_out=num_features_out,
                num_emb=num_emb,
                n_channels_per_head=n_channels_per_head,
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
        self,
        num_features: T.Tuple[int],
        num_layers: int,
        num_emb: int,
        n_channels_per_head: int = 0,
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
                    n_channels_per_head=n_channels_per_head,
                )
                for n_in, n_out, add_down in zip(n_ins, n_outs, add_downs)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for down_block in self.down_blocks:
            x = down_block(x, t)
        return x

    def __iter__(self) -> T.Iterator[torch.Tensor]:
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
        n_channels_per_head: int = 0,
    ):
        super().__init__()
        self.add_up = add_up
        self.setup_res_blocks(
            num_features_in,
            prev_num_features_out,
            num_features_out,
            num_emb,
            num_resnet_layers=num_resnet_layers,
            n_channels_per_head=n_channels_per_head,
        )

        self.setup_upscaling(num_features_out)

    def setup_res_blocks(
        self,
        num_features_in: int,
        prev_num_features_out: int,
        num_output_features: int,
        num_emb: int,
        num_resnet_layers: int = 2,
        n_channels_per_head: int = 0,
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

            self.res_blocks.append(
                ResBlock(
                    n_in,
                    n_out,
                    num_emb,
                    n_channels_per_head=n_channels_per_head,
                )
            )

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
        n_channels_per_head: int = 0,
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
                n_channels_per_head=n_channels_per_head,
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
        n_channels_per_head: int = 0,  # 0 = no attention
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
            list_num_features,
            num_layers,
            self.n_noise_level_emb,
            n_channels_per_head=n_channels_per_head,
        )

        self.mid_block = ResBlock(
            list_num_features[-1],
            list_num_features[-1],
            self.n_noise_level_emb,
            n_channels_per_head=0,
        )

        self.ups = UNetUp(
            self.downs,
            self.n_noise_level_emb,
            n_channels_per_head=n_channels_per_head,
        )

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
        noise_emb = self.emb_mlp(noise_emb.float())

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


def list_of_tuples_to_tensors(
    batch: T.List[T.Tuple[torch.Tensor, int]],
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=int)
    return images, labels


SIG_DATA = 0.66


def get_cs(
    noise_level: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: wtf is happening here?
    totvar = noise_level**2 + SIG_DATA**2
    c_skip = SIG_DATA**2 / totvar
    c_out = noise_level * SIG_DATA / totvar.sqrt()
    c_in = 1 / totvar.sqrt()
    return c_skip, c_out, c_in


def draw_noise_level_from_noise_prior(n: int) -> torch.Tensor:
    "Draws noise level (prior) from a log normal distribution"
    noise_level = torch.randn(n)
    noise_level = 1.2 * noise_level - 1.2
    noise_level = noise_level.exp()
    return noise_level


def draw_img_noise_given_noise_level(
    sig: torch.Tensor,
    images: torch.Tensor = None,
    images_shape: T.Tuple[int, int, int] = None,
) -> torch.Tensor:
    "Draws noise from a normal distribution given the noise level (sig)"
    if images is not None:
        images_shape = images.shape

    img_noise = torch.randn(images_shape)
    img_noise = img_noise * sig
    return img_noise


def fudge_original_images(images: torch.Tensor) -> torch.Tensor:
    return images * 2 - 1


def apply_noise(
    batch: T.List[T.Tuple[torch.Tensor, int]],
) -> T.Tuple[T.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    "Applies noise to the input image and returns the noisy image, the noise level and the de-noised image"

    orig_images, _ = list_of_tuples_to_tensors(batch)

    orig_images = fudge_original_images(orig_images)

    # drawing noise level (prior) from a log normal distribution
    noise_level = draw_noise_level_from_noise_prior(orig_images.shape[0])
    noise_level = noise_level.reshape(-1, 1, 1)

    c_skip, c_out, c_in = get_cs(noise_level)

    # adding noise to the image
    noise = draw_img_noise_given_noise_level(noise_level, images=orig_images)
    noisy_images = orig_images + noise

    target_noise = (orig_images - c_skip * noisy_images) / c_out
    noisy_images = noisy_images * c_in

    noise_level = noise_level.squeeze()

    return (noisy_images, noise_level), target_noise


@tensorclass
class MNISTNoisyDataTrain:
    noisy_image: torch.Tensor
    noise_level: torch.Tensor
    target_noise: torch.Tensor


def mnist_noisy_collate_train(
    batch: T.List[T.Tuple[torch.Tensor, int]],
) -> MNISTNoisyDataTrain:
    (noisy_images, noise_level), target_noise = apply_noise(batch)

    return MNISTNoisyDataTrain(
        noisy_images, noise_level, target_noise, batch_size=[len(noisy_images)]
    )


class UNetModelTensordict(unet.UNetModel):
    def forward(self, input: MNISTNoisyDataTrain) -> torch.Tensor:
        return super().forward(input.noisy_image)


class NoisyUNetModelTensordict(UNetModel):
    def forward(self, input: MNISTNoisyDataTrain) -> torch.Tensor:
        return super().forward(input.noisy_image, input.noise_level)


def get_denoised_images(
    noisy_images: torch.Tensor, noises: torch.Tensor, noise_level: torch.Tensor
) -> torch.Tensor:
    "Returns the de-noised images given the noisy images, noise and the noise level (sig)"
    c_skip, c_out, c_in = get_cs(noise_level)
    denoised_images = noises * c_out + (noisy_images / c_in) * c_skip
    return denoised_images


def compare_input_noise_and_denoised_image(
    noisy_image: torch.Tensor,
    noise: torch.Tensor,
    denoised_image: torch.Tensor,
    bin_bounds: T.Tuple[int, int] = (-3, 3),
    figsize: T.Tuple[int, int] = (10, 10),
    title: str = None,
):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    # images
    ax = axs[0, 0]
    ax.imshow(noisy_image, cmap="gray")
    ax.set_title("Noisy input image")
    ax.axis("off")
    ax = axs[1, 0]
    ax.imshow(noise, cmap="gray")
    ax.set_title("Noise")
    ax.axis("off")
    ax = axs[2, 0]
    ax.imshow(denoised_image, cmap="gray")
    ax.set_title("Denoised image")
    ax.axis("off")

    # histograms
    lb, ub = bin_bounds
    bins = np.linspace(lb, ub, 100)

    ax = axs[0, 1]
    ax.hist(noisy_image.flatten(), bins=bins)
    ax.set_title("Noisy input image")
    ax = axs[1, 1]
    ax.hist(noise.flatten(), bins=bins)
    ax.set_title("Noise")
    ax = axs[2, 1]
    ax.hist(denoised_image.flatten(), bins=bins)
    ax.set_title("Denoised image")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    plt.show()


def denoise_with_model(
    model: T.Union[telemetry.ModelTelemetry, rnnm_learner.Learner],
    images: torch.Tensor,
    noise_levels: torch.Tensor,
) -> T.Tuple[T.List[torch.Tensor], T.List[torch.Tensor]]:
    "Denoises an image with the model for a range of noise levels"
    noise_preds = []
    denoised_preds = []
    for noise_level in tqdm.tqdm(
        noise_levels, total=len(noise_levels), desc="Sigmas"
    ):
        levels = noise_level.repeat(images.shape[0])

        c_skip, c_out, c_in = get_cs(levels.reshape(-1, 1, 1))
        images = images * c_in

        if isinstance(model, telemetry.ModelTelemetry):
            pred_noise = model(images, levels)
        elif isinstance(model, rnnm_learner.Learner):
            ds = rnnm_data.MNISTDatasetWithNoise(
                images,
                levels,
            )
            dl = DataLoader(
                ds,
                batch_size=len(images),
                shuffle=False,
                collate_fn=rnnm_data.collate_mnist_dataset_to_block_with_noise,
            )
            pred_noise = model.predict(dl)
        else:
            raise ValueError(f"Unexpected type for {model=}")

        images = get_denoised_images(
            images, pred_noise, levels.reshape(-1, 1, 1)
        )

        noise_preds.append(pred_noise.detach().cpu())
        denoised_preds.append(images.detach().cpu())
    return noise_preds, denoised_preds


class MSELossMNISTNoisy(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: MNISTNoisyDataTrain
    ) -> torch.Tensor:
        return super().forward(inference, input.target_noise)
