# -*- coding: utf-8 -*-
import typing as T
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import random_neural_net_models.search as search
import random_neural_net_models.utils as utils

logger = utils.get_logger("telemetry.py")


class History:
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        self.not_initialized = name_patterns is None or len(name_patterns) == 0
        if self.not_initialized:
            logger.info("Not collecting history.")
            return

        self.child_search = search.ChildSearch(
            model, max_depth=max_depth_search
        )
        self.every_n = every_n
        self.modules = self.child_search(*name_patterns)

    @property
    def name_matches(self):
        if self.not_initialized:
            return []
        return [a.name for a in self.modules]

    def clean(self):
        if self.not_initialized:
            logger.info("No activation hooks to clean up.")
            return
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


class ActivationsHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)

        self.stats: T.Dict[str, T.List[ActivationStats]] = defaultdict(list)
        self.register_hooks(model, name_patterns, max_depth_search)

    def register_hooks(
        self,
        model: nn.Module,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        self.hooks = defaultdict(None)

        for named_module in self.modules:
            cas = CollectorActivationStats(
                self, named_module.name, every_n=self.every_n
            )
            self.hooks[
                named_module.name
            ] = named_module.module.register_forward_hook(cas)

        logger.info(
            f"Will collect activiation history every {self.every_n}th iteration for: {self.name_matches=}"
        )


class GradientsHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)
        self.stats: T.Dict[str, T.List[ParameterStats]] = defaultdict(list)
        self.register_hooks(model, name_patterns, max_depth_search)

    def register_hooks(
        self,
        model: nn.Module,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        self.child_search = search.ChildSearch(
            model, max_depth=max_depth_search
        )
        self.modules = self.child_search(*name_patterns)
        self.hooks = defaultdict(None)

        for named_module in self.modules:
            cgs = CollectorGradientStats(
                self, named_module.name, every_n=self.every_n
            )
            self.hooks[
                named_module.name
            ] = named_module.module.register_full_backward_hook(cgs)

        logger.info(
            f"Will collect gradient history every {self.every_n}th iteration for: {self.name_matches=}"
        )


class HistoryException(Exception):
    pass


class ParametersHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)
        self.stats: T.Dict[str, T.List[ParameterStats]] = defaultdict(list)
        self.every_n = every_n
        logger.info(
            f"Will collect parameter history every {self.every_n}th iteration for: {self.name_matches=}"
        )

    def __call__(self, _iter: int):
        if self.not_initialized or _iter % self.every_n != 0:
            return

        for named_module in self.modules:
            state_dict = named_module.module.state_dict()
            for name, parameter_values in state_dict.items():
                parameter_values = parameter_values.detach().flatten().float()
                mean = parameter_values.mean().cpu().item()
                std = parameter_values.std().cpu().item()
                abs_perc90 = parameter_values.abs().quantile(0.9).cpu().item()
                self.stats[f"{named_module.name}.{name}"].append(
                    ParameterStats(_iter, mean, std, abs_perc90)
                )

    def draw(
        self,
        name: str,
        figsize: T.Tuple[int, int] = (12, 7),
    ) -> None:
        if self.not_initialized:
            logger.info("Not drawing parameter history.")
            return
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
        if self.not_initialized:
            logger.info("Not getting parameter history.")
            return
        df = pd.DataFrame(self.stats[name])

        # print error to log if any column has inf or nan values
        isna = df.isna()
        if df.isna().any().any():
            mean_na = (
                isna.mean().sort_values(ascending=False).rename("fraction")
            )
            mean_na.index.name = "column"
            raise HistoryException(
                f"{name=} df has missing values: {mean_na.to_markdown()}"
            )

        isinf = df.isin([np.inf, -np.inf])
        if isinf.any().any():
            mean_inf = (
                isinf.mean().sort_values(ascending=False).rename("fraction")
            )
            mean_inf.index.name = "column"
            raise HistoryException(
                f"{name=} df has inf values: {mean_inf.to_markdown()}"
            )

        # print error if df is empty
        if len(df) == 0:
            raise HistoryException(f"{name=} df is empty")

        return df


class CollectorActivationStats:
    def __init__(
        self,
        hook: ActivationsHistory,
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

        self.hook.stats[self.name].append(ActivationStats(mean, std, frac_dead))


class CollectorGradientStats:
    def __init__(
        self,
        hook: GradientsHistory,
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

        self.hook.stats[self.name].append(
            GradientStats(mean, std, abs_perc90, _max, frac_dead)
        )


class ModelTelemetry(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_train_every_n: int = 1,
        loss_test_every_n: int = 1,
        parameters_every_n: int = 1,
        activations_every_n: int = 1,
        gradients_every_n: int = 1,
        loss_names: T.Tuple[str] = ("loss",),
        activations_name_patterns: T.Tuple[str] = None,
        gradients_name_patterns: T.Tuple[str] = None,
        parameters_name_patterns: T.Tuple[str] = None,
        max_depth_search: int = 3,
    ):
        super().__init__()
        self.model = model

        # activations bit
        self.activations_history = ActivationsHistory(
            self.model,
            every_n=activations_every_n,
            name_patterns=activations_name_patterns,
            max_depth_search=max_depth_search,
        )

        # loss bit
        self.loss_history_train = LossHistory(
            loss_train_every_n, names=loss_names
        )
        self.loss_history_test = LossHistory(
            loss_test_every_n, names=loss_names
        )

        # parameter bit
        self.parameter_history = ParametersHistory(
            self.model,
            every_n=parameters_every_n,
            name_patterns=parameters_name_patterns,
            max_depth_search=max_depth_search,
        )

        # gradient bit
        self.gradients_history = GradientsHistory(
            self.model,
            every_n=gradients_every_n,
            name_patterns=gradients_name_patterns,
            max_depth_search=max_depth_search,
        )

    @property
    def name_matches_activations(self):
        return list(self.activations_history.name_matches)

    @property
    def name_matches_gradients(self):
        return list(self.gradients_history.name_matches)

    @property
    def name_matches_parameters(self):
        return list(self.parameter_history.name_matches)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def clean_hooks(self):
        self.activations_history.clean()
        self.gradients_history.clean()

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
        for _name, _stats in self.activations_history.stats.items():
            ax.plot([s.mean for s in _stats], label=_name, alpha=0.5)
        ax.set(title="mean", yscale=yscale)

        # activation std
        ax = axs[2]
        for _name, _stats in self.activations_history.stats.items():
            ax.plot([s.std for s in _stats], label=_name, alpha=0.5)
        ax.set(title="standard deviation", yscale=yscale)

        # share of dead neurons
        ax = axs[0]
        for _name, _stats in self.activations_history.stats.items():
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
        for _name, _stats in self.gradients_history.stats.items():
            ax.plot([s.mean for s in _stats], label=_name, alpha=0.5)
        ax.set(title="mean", yscale=yscale)

        # gradient std
        ax = axs[3]
        for _name, _stats in self.gradients_history.stats.items():
            ax.plot([s.std for s in _stats], label=_name, alpha=0.5)
        ax.set(title="standard deviation", yscale=yscale)

        # abs_perc90
        ax = axs[2]
        for _name, _stats in self.gradients_history.stats.items():
            ax.plot([s.abs_perc90 for s in _stats], label=_name, alpha=0.5)
        ax.legend()
        ax.set(title="90%(abs)", yscale=yscale)

        # vanishing
        ax = axs[1]
        for _name, _stats in self.gradients_history.stats.items():
            ax.plot([s.frac_dead for s in _stats], label=_name, alpha=0.5)
        ax.set(title="frac(dead)")

        # exploding
        ax = axs[0]
        for _name, _stats in self.gradients_history.stats.items():
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
        if len(names) == 0:
            names = self.parameter_history.name_matches
        for name in names:
            try:
                self.parameter_history.draw(name, **kwargs)
            except HistoryException as e:
                logger.error(e)


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
