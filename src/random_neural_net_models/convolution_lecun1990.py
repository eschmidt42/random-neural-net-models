# -*- coding: utf-8 -*-
import typing as T
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset

import random_neural_net_models.utils as utils

logger = utils.get_logger("convolution_lecun1990.py")


class DigitsDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, edge: int = 28):
        self.X = X
        self.y = y
        self.edge = edge

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, int]:
        img = (
            torch.from_numpy(self.X.iloc[idx].values / 255.0)  # normalizing
            .reshape(self.edge, self.edge)
            .double()
        )
        label = int(self.y.iloc[idx])
        return (img, label)


def calc_conv_output_dim(input_dim, kernel_size, padding, stride):
    return int((input_dim - kernel_size + 2 * padding) / stride + 1)


def densify_y(y: torch.Tensor) -> torch.Tensor:
    new_y = F.one_hot(y, num_classes=10)
    new_y[new_y == 0] = -1
    return new_y.double()


class Tanh(nn.Module):
    def __init__(self, A: float = 1.716, S: float = 2 / 3):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.register_buffer("S", torch.tensor(S))

    def forward(self, x: torch.Tensor):
        return self.A * torch.tanh(self.S * x)


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        edge: int,
        n_in_channels: int = 1,
        n_out_channels: int = 1,
        kernel_width: int = 5,
        kernel_height: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        lecun_init: bool = True,
    ):
        super().__init__()

        self.register_buffer("edge", torch.tensor(edge))
        self.register_buffer("n_in_channels", torch.tensor(n_in_channels))
        self.register_buffer("n_out_channels", torch.tensor(n_out_channels))
        self.register_buffer("kernel_width", torch.tensor(kernel_width))
        self.register_buffer("kernel_height", torch.tensor(kernel_height))
        self.register_buffer("stride", torch.tensor(stride))
        self.register_buffer("padding", torch.tensor(padding))
        self.register_buffer("dilation", torch.tensor(dilation))

        self.weight = nn.Parameter(
            torch.empty(
                n_in_channels * kernel_width * kernel_height,
                n_out_channels,
                dtype=torch.double,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(1, n_out_channels, 1, 1, dtype=torch.double)
        )

        # self.bias = rearrange(self.bias, "out_channels -> 1 out_channels 1 1")

        if lecun_init:
            s = 2.4 / (n_in_channels * kernel_width * kernel_height)
            self.weight.data.uniform_(-s, s)
            self.bias.data.uniform_(-s, s)

        else:
            self.weight.data.normal_(0, 1.0)
            self.bias.data.normal_(0, 1.0)

        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_height, kernel_width),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        out_h = out_w = calc_conv_output_dim(
            edge, kernel_width, padding, stride
        )
        self.fold = torch.nn.Fold(
            output_size=(out_h, out_w),
            kernel_size=(1, 1),
            dilation=dilation,
            padding=0,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # inspired by: https://discuss.pytorch.org/t/make-custom-conv2d-layer-efficient-wrt-speed-and-memory/70175/2

        # (N,C,in_h,in_w) -> (N, C*kh*kw, num_patches)
        # N = batch_size, C = in_channels, kh = kernel_height, kw = kernel_width
        input_unfolded = self.unfold(input)

        input_unfolded = rearrange(
            input_unfolded, "N r num_patches -> N num_patches r"
        )

        output_unfolded = input_unfolded @ self.weight
        output_unfolded = rearrange(
            output_unfolded,
            "N num_patches out_channels -> N out_channels num_patches",
        )

        output = self.fold(output_unfolded)  # (N, out_channels, out_h, out_w)
        if self.bias is not None:
            output += self.bias

        return output


class Model(nn.Module):
    # based on LeCun et al. 1990, _Handwritten Digit Recognition: Applications of Neural Net Chips and Automatic Learning_, Neurocomputing, https://link.springer.com/chapter/10.1007/978-3-642-76153-9_35
    # inspired by https://einops.rocks/pytorch-examples.html
    def __init__(
        self,
        edge: int = 28,
        n_classes: int = 10,
        lecun_init: bool = True,
        lecun_act: bool = True,
        A: float = 1.716,
        S: float = 2 / 3,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            edge=edge,
            n_in_channels=1,
            n_out_channels=12,
            kernel_width=5,
            kernel_height=5,
            stride=2,
            padding=2,
            lecun_init=lecun_init,
        )
        edge = edge // 2  # effect of stride

        self.conv2 = Conv2d(
            edge=edge,
            n_in_channels=12,
            n_out_channels=12,
            kernel_width=5,
            kernel_height=5,
            stride=2,
            padding=2,
            lecun_init=lecun_init,
        )
        edge = edge // 2  # effect of stride
        self.lin1 = nn.Linear(edge * edge * 12, 30)
        self.lin2 = nn.Linear(30, n_classes)

        if lecun_init:
            s = 2.4 / self.lin1.weight.shape[0]
            self.lin1.weight.data.uniform_(-s, s)

            s = 2.4 / self.lin2.weight.shape[0]
            self.lin2.weight.data.uniform_(-s, s)

        if lecun_act:
            self.act_conv1 = Tanh(A, S)
            self.act_conv2 = Tanh(A, S)
            self.act_lin1 = Tanh(A, S)
            self.act_lin2 = Tanh(A, S)
        else:
            self.act_conv1 = nn.Tanh()
            self.act_conv2 = nn.Tanh()
            self.act_lin1 = nn.Tanh()
            self.act_lin2 = nn.Tanh()

        self.net = nn.Sequential(
            Rearrange("b h w -> b 1 h w"),
            self.conv1,
            self.act_conv1,
            self.conv2,
            self.act_conv2,
            Rearrange("b c h w -> b (c h w)"),
            self.lin1,
            self.act_lin1,
            self.lin2,
            self.act_lin2,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ParameterHistory:
    def __init__(self, every_n: int = 1):
        self.history = defaultdict(list)
        self.every_n = every_n
        self.iter = []

    def __call__(self, model: nn.Module, _iter: int):
        if _iter % self.every_n != 0:
            return
        state_dict = model.state_dict()

        for name, tensor in state_dict.items():
            self.history[name].append(
                tensor.clone().detach().flatten().cpu().numpy()
            )

        self.iter.append(_iter)

    def get_df(self, name: str) -> pd.DataFrame:
        df = [
            pd.DataFrame({"value": w}).assign(iter=i)
            for i, w in zip(self.iter, self.history[name])
        ]
        return pd.concat(df, ignore_index=True)[["iter", "value"]]

    def get_rolling_mean_df(self, name: str, window: int = 10) -> pd.DataFrame:
        df = self.get_df(name)
        df_roll = df.rolling(window=window, on="iter", min_periods=1).mean()
        if "iter" not in df_roll.columns:
            df_roll["iter"] = range(len(df_roll))
        return df_roll


class LossHistory:
    def __init__(self, every_n: int = 1, names: T.Tuple[str] = ("loss",)):
        self.names = names
        self.history = []
        self.iter = []
        self.every_n = every_n

    def __call__(
        self, losses: T.Union[torch.Tensor, T.Tuple[torch.Tensor]], _iter: int
    ):
        if _iter % self.every_n != 0:
            return
        if isinstance(losses, torch.Tensor):
            self.history.append(losses.item())
        else:
            self.history.append([l.item() for l in losses])
        self.iter.append(_iter)

    def get_df(self) -> pd.DataFrame:
        df = pd.DataFrame({"iter": self.iter})
        if len(self.names) == 1:
            df[self.names[0]] = self.history
        else:
            for i, name in enumerate(self.names):
                df[name] = [l[i] for l in self.history]

        return df

    def get_rolling_mean_df(self, window: int = 10) -> pd.DataFrame:
        df = self.get_df()
        df_roll = df.rolling(window=window, on="iter", min_periods=1).mean()
        if "iter" not in df_roll.columns:
            df_roll["iter"] = range(len(df_roll))
        return df_roll

    def draw(
        self,
        label: str,
        window: int = 10,
        figsize: T.Tuple[int, int] = (12, 4),
        yscale: str = "linear",
    ):
        df = self.get_df()
        df_roll = self.get_rolling_mean_df(window=window)

        for name in self.names:
            fig, ax = plt.subplots(figsize=figsize)
            sns.lineplot(
                data=df, x="iter", y=name, ax=ax, label=label, alpha=0.5
            )
            sns.lineplot(
                data=df_roll,
                x="iter",
                y=name,
                ax=ax,
                label=f"{label} (rolling mean)",
                alpha=0.5,
            )
            ax.set(
                xlabel="Iter",
                ylabel="Loss",
                title=f"Loss History: {name}",
                yscale=yscale,
            )

            plt.tight_layout()

        return fig, ax


def draw_history(
    history: ParameterHistory,
    name: str,
    figsize: T.Tuple[int, int] = (12, 4),
    weight_bins: int = 20,
    bias_bins: int = 10,
) -> None:
    fig, axs = plt.subplots(figsize=figsize, nrows=2, sharex=True)

    ax = axs[0]
    _name = f"{name}.weight"
    df = history.get_df(_name)

    n_iter = df["iter"].nunique()
    bins = (n_iter, weight_bins)
    sns.histplot(
        data=df,
        x="iter",
        y="value",
        ax=ax,
        thresh=None,
        cmap="plasma",
        bins=bins,
    )
    ax.set_ylabel("weight")
    ax.set_title(name)

    ax = axs[1]
    _name = f"{name}.bias"
    df = history.get_df(_name)

    bins = (n_iter, bias_bins)
    sns.histplot(
        data=df,
        x="iter",
        y="value",
        ax=ax,
        thresh=None,
        cmap="plasma",
        bins=bins,
    )
    ax.set_xlabel("iter")
    ax.set_ylabel("bias")

    plt.tight_layout()
    plt.show()


@dataclass
class ActivationStats:
    mean: float
    std: float
    frac_dead: int


@dataclass
class ParameterStats:
    iter: int
    mean: float
    std: float
    abs_perc90: float


@dataclass
class GradientStats:
    mean: float
    std: float
    abs_perc90: float
    max: float
    frac_dead: float


class ParameterHistory2:
    def __init__(
        self, model: nn.Module, every_n: int = 1, sub_modules: T.Tuple[str] = ()
    ):
        self.history: T.Dict[str, T.List[ParameterStats]] = defaultdict(list)
        self.every_n = every_n
        self.model = model
        self.sub_modules = sub_modules

    def __call__(self, _iter: int):
        if _iter % self.every_n != 0:
            return

        state_dict = self.model.state_dict()

        for name, parameter_values in state_dict.items():
            parameter_values = parameter_values.detach().flatten().float()
            mean = parameter_values.mean().cpu().item()
            std = parameter_values.std().cpu().item()
            abs_perc90 = parameter_values.abs().quantile(0.9).cpu().item()
            self.history[name].append(
                ParameterStats(_iter, mean, std, abs_perc90)
            )

        for sub_module in self.sub_modules:
            if hasattr(self.model, sub_module):
                state_dict = getattr(self.model, sub_module).state_dict()

                for name, parameter_values in state_dict.items():
                    parameter_values = (
                        parameter_values.detach().flatten().float()
                    )
                    mean = parameter_values.mean().cpu().item()
                    std = parameter_values.std().cpu().item()
                    abs_perc90 = (
                        parameter_values.abs().quantile(0.9).cpu().item()
                    )
                    self.history[name].append(
                        ParameterStats(_iter, mean, std, abs_perc90)
                    )

    def draw(
        self,
        name: str,
        figsize: T.Tuple[int, int] = (12, 7),
    ) -> None:
        fig, axs = plt.subplots(figsize=figsize, nrows=2, sharex=True)

        ax = axs[0]
        _name = f"{name}.weight"
        df = self.get_df(_name)

        ax.fill_between(
            df["iter"],
            df["mean"] - df["std"],
            df["mean"] + df["std"],
            alpha=0.5,
            label="mean+-std",
        )
        sns.lineplot(
            data=df,
            x="iter",
            y="mean",
            ax=ax,
            label="mean",
        )
        sns.lineplot(
            data=df,
            x="iter",
            y="abs_perc90",
            ax=ax,
            label="90%(abs(param))",
        )
        ax.legend()
        ax.set_ylabel("weight")
        ax.set_title(f"{_name}")

        ax = axs[1]
        _name = f"{name}.bias"
        df = self.get_df(_name)

        is_useful_std = (~(df["std"].isin([np.inf, -np.inf]).any())) & df[
            "std"
        ].notna().all()
        if is_useful_std:
            ax.fill_between(
                df["iter"],
                df["mean"] - df["std"],
                df["mean"] + df["std"],
                alpha=0.5,
                label="mean+-std",
            )
        else:
            logger.error(
                f"{_name=} df['std'] has inf or nan values, skipping fillbetween plot."
            )

        sns.lineplot(
            data=df,
            x="iter",
            y="mean",
            ax=ax,
            label="mean",
        )
        sns.lineplot(
            data=df,
            x="iter",
            y="abs_perc90",
            ax=ax,
            label="90%(abs(param))",
        )
        ax.legend()
        ax.set_ylabel("bias")
        ax.set_title(f"{_name}")

        plt.tight_layout()
        plt.show()

    def get_df(self, name: str) -> pd.DataFrame:
        df = pd.DataFrame(self.history[name])
        # print error to log if any column has inf or nan values
        isna = df.isna()
        if df.isna().any().any():
            mean_na = (
                isna.mean().sort_values(ascending=False).rename("fraction")
            )
            mean_na.index.name = "column"
            logger.error(
                f"{name=} df has missing values: {mean_na.to_markdown()}"
            )
        isinf = df.isin([np.inf, -np.inf])
        if isinf.any().any():
            mean_inf = (
                isinf.mean().sort_values(ascending=False).rename("fraction")
            )
            mean_inf.index.name = "column"
            logger.error(f"{name=} df has inf values: {mean_inf.to_markdown()}")
        return df


def check_module_name_is_activation(name: str) -> bool:
    return name.startswith("act_")


def check_module_name_grad_relevant(name: str) -> bool:
    return name != "net"


class ModelTelemetry(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        func_is_act: T.Callable = check_module_name_is_activation,
        func_is_grad_relevant: T.Callable = check_module_name_grad_relevant,
        loss_train_every_n: int = 1,
        loss_test_every_n: int = 1,
        parameter_every_n: int = 1,
        activations_every_n: int = 1,
        gradients_every_n: int = 1,
        loss_names: T.Tuple[str] = ("loss",),
        sub_modules: T.Tuple[str] = (),
    ):
        super().__init__()
        self.model = model

        # activations bit
        self.hooks_activations = defaultdict(None)
        self.stats_activations: T.Dict[
            str, T.List[ActivationStats]
        ] = defaultdict(list)

        for name, child in self.model.named_children():
            if func_is_act(name):
                cas = CollectorActivationStats(
                    self, name, every_n=activations_every_n
                )
                self.hooks_activations[name] = child.register_forward_hook(cas)
        for sub_module in sub_modules:
            if hasattr(self.model, sub_module):
                for name, child in getattr(
                    self.model, sub_module
                ).named_children():
                    if func_is_act(name):
                        cas = CollectorActivationStats(
                            self, name, every_n=activations_every_n
                        )
                        self.hooks_activations[
                            name
                        ] = child.register_forward_hook(cas)

        # loss bit
        self.loss_history_train = LossHistory(
            loss_train_every_n, names=loss_names
        )
        self.loss_history_test = LossHistory(
            loss_test_every_n, names=loss_names
        )

        # parameter bit
        self.parameter_history = ParameterHistory2(
            self.model, every_n=parameter_every_n, sub_modules=sub_modules
        )

        # gradient bit
        self.hooks_gradients = defaultdict(None)
        self.stats_gradients: T.Dict[str, T.List[ParameterStats]] = defaultdict(
            list
        )

        for name, child in self.model.named_children():
            if func_is_grad_relevant(name):
                cgs = CollectorGradientStats(
                    self, name, every_n=gradients_every_n
                )
                self.hooks_gradients[name] = child.register_full_backward_hook(
                    cgs
                )
        for sub_module in sub_modules:
            if hasattr(self.model, sub_module):
                for name, child in getattr(
                    self.model, sub_module
                ).named_children():
                    if func_is_grad_relevant(name):
                        cgs = CollectorGradientStats(
                            self, name, every_n=gradients_every_n
                        )
                        self.hooks_gradients[
                            name
                        ] = child.register_full_backward_hook(cgs)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def clean_hooks(self):
        for hook in self.hooks_activations.values():
            hook.remove()
        self.hooks_activations.clear()

    def draw_activation_stats(
        self,
        figsize: T.Tuple[int, int] = (12, 8),
        yscale: str = "linear",
        leg_lw: float = 5.0,
    ):
        fig, axs = plt.subplots(figsize=figsize, nrows=3, sharex=True)
        plt.suptitle("Activation Stats")

        # activation mean
        ax = axs[1]
        for _name, _stats in self.stats_activations.items():
            ax.plot([s.mean for s in _stats], label=_name, alpha=0.5)
        ax.set(title="mean", yscale=yscale)

        # activation std
        ax = axs[2]
        for _name, _stats in self.stats_activations.items():
            ax.plot([s.std for s in _stats], label=_name, alpha=0.5)
        ax.set(title="standard deviation", yscale=yscale)

        # share of dead neurons
        ax = axs[0]
        for _name, _stats in self.stats_activations.items():
            ax.plot([s.frac_dead for s in _stats], label=_name, alpha=0.5)
        ax.set(title="fraction of dead neurons", xlabel="iter")

        axs[1].legend()
        for leg_obj in axs[1].legend().legendHandles:
            leg_obj.set_linewidth(leg_lw)

        plt.tight_layout()

    def draw_gradient_stats(
        self,
        figsize: T.Tuple[int, int] = (12, 15),
        yscale: str = "linear",
        leg_lw: float = 5.0,
    ):
        fig, axs = plt.subplots(figsize=figsize, nrows=5, sharex=True)
        plt.suptitle("Gradient Stats")

        # gradient mean
        ax = axs[4]
        for _name, _stats in self.stats_gradients.items():
            ax.plot([s.mean for s in _stats], label=_name, alpha=0.5)
        ax.set(title="mean", yscale=yscale)

        # gradient std
        ax = axs[3]
        for _name, _stats in self.stats_gradients.items():
            ax.plot([s.std for s in _stats], label=_name, alpha=0.5)
        ax.set(title="standard deviation", yscale=yscale)

        # abs_perc90
        ax = axs[2]
        for _name, _stats in self.stats_gradients.items():
            ax.plot([s.abs_perc90 for s in _stats], label=_name, alpha=0.5)
        ax.legend()
        ax.set(title="90%(abs)", yscale=yscale)

        # vanishing
        ax = axs[1]
        for _name, _stats in self.stats_gradients.items():
            ax.plot([s.frac_dead for s in _stats], label=_name, alpha=0.5)
        ax.set(title="frac(dead)")

        # exploding
        ax = axs[0]
        for _name, _stats in self.stats_gradients.items():
            ax.plot([s.max for s in _stats], label=_name, alpha=0.5)
        ax.set(title="max(abs)", xlabel="iter", yscale=yscale)

        for leg_obj in axs[2].legend().legendHandles:
            leg_obj.set_linewidth(leg_lw)

        plt.tight_layout()

    def draw_loss_history_train(self, **kwargs):
        self.loss_history_train.draw("train", **kwargs)

    def draw_loss_history_test(self, **kwargs):
        if len(self.loss_history_test.history) == 0:
            logger.warning("No test loss history available")
            return
        self.loss_history_test.draw("test", **kwargs)

    def draw_parameter_stats(self, *names, **kwargs):
        for name in names:
            self.parameter_history.draw(name, **kwargs)


class CollectorActivationStats:
    def __init__(
        self,
        hook: ModelTelemetry,
        name: str,
        every_n: int = 1,
        threshold_dead=1e-6,
    ):
        self.hook = hook
        self.name = name
        self.every_n = every_n
        self.iter = 0
        self.threshold_dead = threshold_dead

    def __call__(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        self.iter += 1
        if self.iter % self.every_n != 0:
            return

        acts = output.detach().flatten()
        mean = acts.mean().cpu().item()
        std = acts.std().cpu().item()
        frac_dead = (acts.abs() < self.threshold_dead).sum().cpu().item() / len(
            acts
        )

        self.hook.stats_activations[self.name].append(
            ActivationStats(mean, std, frac_dead)
        )


class CollectorGradientStats:
    def __init__(
        self,
        hook: ModelTelemetry,
        name: str,
        every_n: int = 1,
        threshold_dead: float = 1e-8,
    ):
        self.hook = hook
        self.name = name
        self.every_n = every_n
        self.iter = 0
        self.threshold_dead = threshold_dead

    def __call__(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        self.iter += 1
        if self.iter % self.every_n != 0:
            return
        vals = output[0].detach().flatten()
        mean = vals.mean().cpu().item()
        std = vals.std().cpu().item()
        abs_perc90 = vals.abs().quantile(0.9).cpu().item()
        _max = vals.abs().max().cpu().item()
        frac_dead = (vals.abs() < self.threshold_dead).sum().cpu().item() / len(
            vals
        )

        self.hook.stats_gradients[self.name].append(
            GradientStats(mean, std, abs_perc90, _max, frac_dead)
        )


class Hook:
    def __init__(self, module: nn.Module, func: T.Callable, name: str = None):
        self.hook = module.register_forward_hook(partial(func, self))
        self.name = name

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_stats(
    hook,
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
    hist_bins: int = 80,
    hist_range: T.Tuple[float, float] = (0.0, 10.0),
):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    acts = output.cpu().detach()
    mean, std = acts.mean().item(), acts.std().item()
    hist = acts.abs().histc(hist_bins, hist_range[0], hist_range[1])
    hook.stats[0].append(mean)
    hook.stats[1].append(std)
    hook.stats[2].append(hist)


def get_hooks(
    model: Model,
    hook_func: T.Callable = partial(append_stats, hist_range=(0, 2)),
) -> T.List[Hook]:
    # model_acts = [
    #     model.act_conv1,
    #     model.act_conv2,
    #     model.act_lin1,
    #     model.act_lin2,
    # ]
    # act_names = ["act_conv1", "act_conv2", "act_lin1", "act_lin2"]
    hooks = [
        Hook(layer, hook_func, name=name)
        for (name, layer) in model.named_children()
        if name.startswith("act_")
    ]
    return hooks


def draw_activation_stats(hooks: T.List[Hook], hist_aspect: float = 10.0):
    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, sharex=True)

    for h in hooks:
        axs[0].plot(h.stats[0], label=h.name, alpha=0.5)
        axs[1].plot(h.stats[1], label=h.name, alpha=0.5)

    axs[0].legend()
    axs[0].set(title="activation mean")
    axs[1].legend()
    axs[1].set(title="activation std")
    plt.tight_layout()

    for h in hooks:
        fig, ax = plt.subplots(figsize=(12, 4), nrows=1)
        hist = torch.stack(h.stats[2]).t().float().log1p().numpy()
        ax.imshow(hist, aspect=hist_aspect, origin="lower")
        ax.grid(False)
        ax.set_axis_off()

        ax.set_title(h.name, fontsize=16)
        plt.tight_layout()


def clear_hooks(hooks: T.List[Hook]):
    for h in hooks:
        h.remove()
    del hooks[:]
