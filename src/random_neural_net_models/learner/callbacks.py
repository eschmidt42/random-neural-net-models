from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import torch.optim as optim
from plotnine import aes, geom_line, geom_point, ggplot, labs, scale_y_log10, ylim
from pydantic.dataclasses import dataclass
from torch.optim.lr_scheduler import LRScheduler

import random_neural_net_models.telemetry as rnnm_telemetry
from random_neural_net_models.learner.interfaces import (
    Callback,
    CallbackEnum,
    LearnerProtocol,
)


@dataclass
class Loss:
    iteration: int
    batch: int
    epoch: int
    loss: float
    loss_valid: float | None = None


class CancelFitException(Exception): ...


class TrainLossCallback(Callback):
    enum = CallbackEnum.train_loss
    losses: list[Loss]

    def __init__(self):
        self.losses = []

    def after_loss(self, learner: LearnerProtocol):
        self.losses.append(
            Loss(
                learner.iteration,
                learner.batch,
                learner.epoch,
                float(learner.loss.detach().cpu().numpy()),
            )
        )

    def before_train(self, learner: LearnerProtocol):
        pass

    def after_train(self, learner: LearnerProtocol):
        pass

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        pass

    def after_epoch(self, learner: LearnerProtocol):
        if hasattr(learner, "loss_valid"):
            loss_valid = float(learner.loss_valid.detach().numpy())
            self.losses[-1].loss_valid = loss_valid

    def get_losses(self) -> pl.DataFrame:
        return pl.from_dicts([asdict(loss) for loss in self.losses])

    def get_losses_valid(self) -> pl.DataFrame:
        return self.get_losses().filter(pl.col("loss_valid").is_not_nan())

    def plot(
        self,
        window: int = 10,
        window_valid: int = 5,
        log_yscale: bool = False,
        ylimits: tuple[int | float, int | float] | None = None,
    ):
        losses = self.get_losses()

        losses = losses.rename({"loss": "train loss", "loss_valid": "valid loss"})
        losses_train = (
            losses.select(["iteration", "train loss"])
            .filter(pl.col("train loss").is_not_null())
            .sort("iteration")
        )
        losses_valid = (
            losses.select(["iteration", "valid loss"])
            .filter(pl.col("valid loss").is_not_null())
            .sort("iteration")
        )

        losses_plot = pl.concat(
            [
                losses_train.rename({"train loss": "loss"}).with_columns(
                    **{"set": pl.lit("train")}
                ),
                losses_valid.rename({"valid loss": "loss"}).with_columns(
                    **{"set": pl.lit("valid")}
                ),
            ]
        )

        p = (
            ggplot(losses_plot, aes(x="iteration", y="loss", color="set"))
            + geom_line(alpha=0.5)
            + geom_point(size=1, alpha=0.5, fill="none")
            + labs(title="Training history: loss vs iteration", x="iteration")
        )
        if log_yscale:
            p = p + scale_y_log10()
        if ylimits:
            p = p + ylim(*ylimits)

        p.show()

        losses_train_rolling = losses_train.with_columns(
            **{"loss": pl.col("train loss").rolling_mean(window)}
        ).drop("train loss")
        losses_valid_rolling = losses_valid.with_columns(
            **{"loss": pl.col("valid loss").rolling_mean(window_valid)}
        ).drop("valid loss")
        losses_plot = pl.concat(
            [
                losses_train_rolling.with_columns(**{"set": pl.lit("train (rolling)")}),
                losses_valid_rolling.with_columns(**{"set": pl.lit("valid (rolling)")}),
            ]
        )
        losses_plot = losses_plot.drop_nulls(["loss"])

        p = (
            ggplot(losses_plot, aes(x="iteration", y="loss", color="set"))
            + geom_line(alpha=0.5)
            + geom_point(size=1, alpha=0.5, fill="none")
            + labs(
                title="Rolling mean training history: loss vs iteration", x="iteration"
            )
        )
        if log_yscale:
            p = p + scale_y_log10()
        if ylimits:
            p = p + ylim(*ylimits)

        p.show()


class TrainActivationsCallback(Callback):
    enum = CallbackEnum.train_activations
    activations_history: rnnm_telemetry.ActivationsHistory
    every_n: int
    name_patterns: tuple[str, ...] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str, ...] | None = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_train(self, learner: LearnerProtocol):
        self.activations_history = rnnm_telemetry.ActivationsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_train(self, learner: LearnerProtocol):
        self.activations_history.clean()

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        pass

    def after_epoch(self, learner: LearnerProtocol):
        pass

    def get_stats(self) -> pd.DataFrame:
        all_stats = []
        for name, stats in self.activations_history.stats.items():
            tmp = pd.DataFrame([asdict(v) for v in stats])
            tmp["name"] = name
            tmp["call"] = np.arange(len(tmp))
            all_stats.append(tmp)
        return pd.concat(all_stats, ignore_index=True)

    def plot(self):
        activations = self.get_stats()

        fig, axs = plt.subplots(figsize=(12, 12), nrows=3, sharex=True)

        ax = axs[0]
        sns.lineplot(data=activations, x="call", y="mean", hue="name", ax=ax)
        ax.get_legend().remove()

        ax = axs[1]
        sns.lineplot(data=activations, x="call", y="std", hue="name", ax=ax)

        ax = axs[2]
        sns.lineplot(data=activations, x="call", y="frac_dead", hue="name", ax=ax)
        ax.get_legend().remove()

        plt.tight_layout()


class TrainGradientsCallback(Callback):
    enum = CallbackEnum.train_gradients
    gradients_history: rnnm_telemetry.GradientsHistory
    every_n: int
    name_patterns: tuple[str, ...] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str, ...] | None = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_train(self, learner: LearnerProtocol):
        self.gradients_history = rnnm_telemetry.GradientsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_train(self, learner: LearnerProtocol):
        self.gradients_history.clean()

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        pass

    def after_epoch(self, learner: LearnerProtocol):
        pass

    def get_stats(self) -> pd.DataFrame:
        all_stats = []
        for name, stats in self.gradients_history.stats.items():
            tmp = pd.DataFrame([asdict(v) for v in stats])
            tmp["name"] = name
            tmp["call"] = np.arange(len(tmp))
            all_stats.append(tmp)
        return pd.concat(all_stats, ignore_index=True)

    def plot(self):
        gradients = self.get_stats()
        fig, axs = plt.subplots(figsize=(12, 12), nrows=4, sharex=True)

        ax = axs[0]
        sns.lineplot(data=gradients, x="call", y="mean", hue="name", ax=ax)
        ax.get_legend().remove()

        ax = axs[1]
        sns.lineplot(data=gradients, x="call", y="std", hue="name", ax=ax)

        ax = axs[2]
        sns.lineplot(data=gradients, x="call", y="frac_dead", hue="name", ax=ax)
        ax.get_legend().remove()

        ax = axs[3]
        sns.lineplot(data=gradients, x="call", y="abs_perc90", hue="name", ax=ax)
        ax.get_legend().remove()

        plt.tight_layout()


class TrainParametersCallback(Callback):
    enum = CallbackEnum.train_parameters
    parameters_history: rnnm_telemetry.ParametersHistory
    every_n: int
    name_patterns: tuple[str, ...] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str, ...] | None = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_train(self, learner: LearnerProtocol):
        self.parameters_history = rnnm_telemetry.ParametersHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_train(self, learner: LearnerProtocol):
        pass

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        self.parameters_history(learner.iteration)

    def after_epoch(self, learner: LearnerProtocol):
        pass

    def get_stats(self) -> pd.DataFrame:
        all_stats = []
        for name, stats in self.parameters_history.stats.items():
            if not (name.endswith("weight") or name.endswith("bias")):
                continue
            tmp = pd.DataFrame([asdict(v) for v in stats])
            tmp["name"] = name
            tmp["call"] = np.arange(len(tmp))
            all_stats.append(tmp)
        return pd.concat(all_stats, ignore_index=True)

    def plot(self):
        parameters = self.get_stats()
        fig, axs = plt.subplots(figsize=(10, 10), nrows=3, sharex=True)

        ax = axs[0]
        sns.lineplot(data=parameters, x="iter", y="mean", hue="name", ax=ax)
        ax.get_legend().remove()

        ax = axs[1]
        sns.lineplot(data=parameters, x="iter", y="std", hue="name", ax=ax)

        ax = axs[2]
        sns.lineplot(data=parameters, x="iter", y="abs_perc90", hue="name", ax=ax)
        ax.get_legend().remove()

        plt.tight_layout()


@dataclass
class LossWithLR:
    iteration: int
    batch: int
    epoch: int
    loss: float
    smooth_loss: float
    lr: float


class LRFinderCallback(Callback):
    enum = CallbackEnum.lr_finder
    losses: list[LossWithLR]

    def __init__(
        self,
        start_lr: float,
        end_lr: float,
        num_iterations: int,
        stop_at_jump: bool = True,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = num_iterations if num_iterations > 5 else 6
        self.stop_at_jump = stop_at_jump
        self.best_loss = float("inf")
        self.losses = []
        self.smooth_losses = []

    def schedule(self, start: float, stop: float, pct: float) -> float:
        return start * (stop / start) ** pct

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_batch(self, learner: LearnerProtocol):
        self.lr = self.schedule(
            self.start_lr, self.end_lr, learner.iteration / self.num_iterations
        )

        for param_group in learner.optimizer.param_groups:
            param_group["lr"] = self.lr

    def after_batch(self, learner: LearnerProtocol):
        current_loss = float(learner.loss.detach().cpu().numpy())
        current_smooth_loss = float(learner.smooth_loss.detach().cpu().numpy())
        self.losses.append(
            LossWithLR(
                learner.iteration,
                learner.batch,
                learner.epoch,
                current_loss,
                current_smooth_loss,
                self.lr,
            )
        )
        if current_smooth_loss < self.best_loss:
            self.best_loss = current_smooth_loss
        if current_smooth_loss > 4 * self.best_loss and self.stop_at_jump:
            raise CancelFitException()
        if learner.iteration >= self.num_iterations:
            raise CancelFitException()

    def before_train(self, learner: LearnerProtocol):
        # https://github.com/fastai/fastai/blob/43dbef38fe52b8b074d91ee1773e702a1401a486/fastai/callback/schedule.py#L180
        learner.save()

    def after_train(self, learner: LearnerProtocol):
        # https://github.com/fastai/fastai/blob/43dbef38fe52b8b074d91ee1773e702a1401a486/fastai/callback/schedule.py#L199
        learner.optimizer.zero_grad()
        learner.load()
        learner.learner_path.unlink()

    def after_epoch(self, learner: LearnerProtocol):
        pass

    def get_losses(self) -> pl.DataFrame:
        return pl.from_dicts([asdict(loss) for loss in self.losses])

    def plot(
        self,
        log_yscale: bool = False,
        ylimits: tuple[int | float, int | float] | None = None,
    ):
        lr_find_data = self.get_losses().rename(
            {"loss": "raw loss", "smooth_loss": "smooth loss"}
        )

        lr_find_data = lr_find_data.select(["lr", "smooth loss", "raw loss"]).unpivot(
            on=["smooth loss", "raw loss"],
            index="lr",
            variable_name="source",
            value_name="loss",
        )

        p = (
            ggplot(lr_find_data, aes(x="lr", y="loss", color="source"))
            + geom_line()
            + labs(
                title="learning rate finder: loss vs learning rate", x="learning rate"
            )
        )
        if log_yscale:
            p = p + scale_y_log10()
        if ylimits:
            p = p + ylim(*ylimits)

        p.show()


class EveryBatchSchedulerCallback(Callback):
    enum = CallbackEnum.every_batch_scheduler

    def __init__(self, scheduler: optim.lr_scheduler.LRScheduler):
        self.scheduler = scheduler

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_train(self, learner: LearnerProtocol):
        pass

    def after_train(self, learner: LearnerProtocol):
        pass

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        self.scheduler.step()

    def after_epoch(self, learner: LearnerProtocol):
        pass


class EarlyStoppingCallback(Callback):
    enum = CallbackEnum.early_stopping

    def __init__(self, patience: int):
        self.patience = patience
        self.n_epochs_worse = 0
        self.best_loss = torch.tensor(torch.inf)

    def after_loss(self, learner: LearnerProtocol):
        pass

    def before_train(self, learner: LearnerProtocol):
        pass

    def after_train(self, learner: LearnerProtocol):
        pass

    def before_batch(self, learner: LearnerProtocol):
        pass

    def after_batch(self, learner: LearnerProtocol):
        pass

    def after_epoch(self, learner: LearnerProtocol):
        if not hasattr(learner, "loss_valid"):
            msg = "Learner is missing attribute loss_valid. Did you pass dataloader_valid to Learner.fit?"
            raise ValueError(msg)

        if learner.loss_valid > self.best_loss.to(learner.device):
            self.n_epochs_worse += 1
            if self.n_epochs_worse > self.patience:
                raise CancelFitException()
        else:
            self.best_loss = learner.loss_valid
            self.n_epochs_worse = 0


@dataclass(frozen=True)
class TelemetryArgs:
    every_n: int
    max_depth_search: int
    name_patterns: tuple[str, ...]


def get_default_train_callbacks(
    activation_callback_args: TelemetryArgs = TelemetryArgs(10, 4, (".*act",)),
    use_activation_callback: bool = True,
    gradient_callback_args: TelemetryArgs = TelemetryArgs(10, 4, (".*lin",)),
    use_gradient_callback: bool = True,
    parameter_callback_args: TelemetryArgs = TelemetryArgs(10, 4, (".*lin",)),
    use_parameter_callback: bool = True,
    scheduler: LRScheduler | None = None,
) -> list[Callback]:
    callbacks: list[Callback] = [TrainLossCallback()]

    if use_activation_callback:
        activations_callback = TrainActivationsCallback(
            every_n=activation_callback_args.every_n,
            max_depth_search=activation_callback_args.max_depth_search,
            name_patterns=activation_callback_args.name_patterns,
        )
        callbacks.append(activations_callback)

    if use_gradient_callback:
        gradients_callback = TrainGradientsCallback(
            every_n=gradient_callback_args.every_n,
            max_depth_search=gradient_callback_args.max_depth_search,
            name_patterns=gradient_callback_args.name_patterns,
        )
        callbacks.append(gradients_callback)

    if use_parameter_callback:
        parameters_callback = TrainParametersCallback(
            every_n=parameter_callback_args.every_n,
            max_depth_search=parameter_callback_args.max_depth_search,
            name_patterns=parameter_callback_args.name_patterns,
        )
        callbacks.append(parameters_callback)

    if scheduler:
        scheduler_callback = EveryBatchSchedulerCallback(scheduler)
        callbacks.append(scheduler_callback)

    return callbacks


def plot(callbacks: list[Callback]):
    for cb in callbacks:
        match cb:
            case (
                TrainLossCallback()
                | TrainParametersCallback()
                | TrainGradientsCallback()
                | TrainActivationsCallback()
            ):
                cb.plot()
            case _:
                pass
