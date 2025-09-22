from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from pydantic.dataclasses import dataclass

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

    def get_losses(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(loss) for loss in self.losses])

    def get_losses_valid(self) -> pd.DataFrame:
        return self.get_losses().loc[self.get_losses()["loss_valid"].notna(), :]

    def plot(self, window: int = 10, window_valid: int = 5, yscale: str = "linear"):
        fig, ax = plt.subplots(figsize=(10, 4))
        losses = self.get_losses()
        sns.scatterplot(
            data=losses,
            x="iteration",
            y="loss",
            ax=ax,
            label="train",
            alpha=0.4,
            s=5,
        )
        losses["loss_rolling"] = losses.rolling(window=window)["loss"].mean()
        sns.lineplot(
            data=losses,
            x="iteration",
            y="loss_rolling",
            ax=ax,
            label="train (rolling)",
            color="black",
            alpha=0.5,
        )

        losses_valid = self.get_losses_valid()
        if len(losses_valid) > 0:
            losses_valid["loss_rolling"] = losses_valid.rolling(window=window_valid)[
                "loss_valid"
            ].mean()
            sns.scatterplot(
                data=losses_valid,
                x="iteration",
                y="loss_valid",
                ax=ax,
                label="valid",
                color="red",
                alpha=0.6,
                s=7,
            )
            sns.lineplot(
                data=losses_valid,
                x="iteration",
                y="loss_rolling",
                ax=ax,
                label="valid (rolling)",
                color="red",
                # alpha=0.5,
            )

        ax.legend(title="set")
        ax.set(yscale=yscale)
        plt.tight_layout()


class TrainActivationsCallback(Callback):
    enum = CallbackEnum.train_activations
    activations_history: rnnm_telemetry.ActivationsHistory
    every_n: int
    name_patterns: tuple[str] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
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
    name_patterns: tuple[str] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
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
    name_patterns: tuple[str] | None
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
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

    def get_losses(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(loss) for loss in self.losses])

    def plot(
        self,
        yscale: str = "linear",
        ylim: tuple[int | float, int | float] | None = None,
    ):
        fig, ax = plt.subplots(figsize=(12, 4))
        lr_find_data = self.get_losses()
        sns.lineplot(data=lr_find_data, x="lr", y="smooth_loss", ax=ax, label="smooth")
        sns.lineplot(
            data=lr_find_data,
            x="lr",
            y="loss",
            ax=ax,
            label="raw",
            alpha=0.5,
            linestyle="dashed",
        )
        ax.legend(title="loss type")
        ax.set(ylabel="loss", xscale="log", yscale=yscale, ylim=ylim)
        plt.tight_layout()


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
