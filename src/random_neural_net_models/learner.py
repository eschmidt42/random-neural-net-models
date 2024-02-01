# -*- coding: utf-8 -*-
import datetime
import typing as T
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.modules.loss as torch_loss
import torch.optim as optim
import tqdm
from pydantic.dataclasses import dataclass
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import random_neural_net_models.telemetry as rnnm_telemetry
import random_neural_net_models.utils as utils

logger = utils.get_logger("learner.py")


class Callback:
    ...


class CancelFitException(Exception):
    ...


class Events(Enum):
    # fastai callback events: https://docs.fast.ai/callback.core.html#events
    after_loss = "after_loss"
    before_train = "before_train"
    after_train = "after_train"
    before_batch = "before_batch"
    after_batch = "after_batch"
    after_epoch = "after_epoch"


# TODO: add logging of activations, weights, gradient and losses using wandb


def get_learner_name() -> str:
    return f"learner-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"


class Learner:
    model: nn.Module
    optimizer: Optimizer
    loss_func: torch_loss._Loss
    loss: torch.Tensor
    losses: torch.Tensor
    smooth_loss: torch.Tensor
    smooth_count: int
    save_dir: Path
    loss_valid: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_func: torch_loss._Loss,
        callbacks: T.List[Callback] = None,
        save_dir: Path = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        if callbacks is None:
            self.registered_callbacks = []
        else:
            self.registered_callbacks = callbacks

        self.save_dir = (
            save_dir.resolve().absolute() if save_dir is not None else None
        )

        self.iteration = 0
        self.smooth_count = 0
        self.smooth_val = torch.tensor(0.0)
        self.smooth_loss = torch.tensor(torch.inf)
        self.losses = torch.tensor([]).float()

    def callback(self, event: Events):
        relevant_callbacks = [
            c
            for c in self.registered_callbacks
            if isinstance(c, Callback) and hasattr(c, event.value)
        ]
        for callback in relevant_callbacks:
            getattr(callback, event.value)(self)

    # TODO: this does not produce the expected loss vs lr curve for rumelhart nb - is this correct?
    def _update_smooth_loss(self):
        self.losses = torch.cat(
            (self.losses, torch.tensor([self.loss.detach()]))
        )
        beta = 0.98
        self.smooth_count += 1
        self.smooth_val = torch.lerp(
            self.loss.detach().mean(), self.smooth_val, beta
        )
        self.smooth_loss = self.smooth_val / (1 - beta**self.smooth_count)

    def do_batch_train(self, tensordict: TensorDict):
        self.model.train()

        self.callback(Events.before_batch)
        self.optimizer.zero_grad()
        inference = self.model(tensordict)
        self.loss = self.loss_func(inference, tensordict)
        self.callback(Events.after_loss)
        self.loss.backward()
        self._update_smooth_loss()
        self.optimizer.step()
        self.callback(Events.after_batch)
        self.iteration += 1

    def do_batch_valid(self, tensordict: TensorDict) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            inference = self.model(tensordict)
            return self.loss_func(inference, tensordict)

    def do_epoch(
        self, dataloader_train: DataLoader, dataloader_valid: DataLoader = None
    ):
        for self.batch, tensordict in tqdm.tqdm(
            enumerate(dataloader_train),
            total=len(dataloader_train),
            desc="batch (train)",
        ):
            self.do_batch_train(tensordict)

        if dataloader_valid is not None:
            losses_valid = []
            for tensordict in tqdm.tqdm(
                dataloader_valid,
                desc="batch (valid)",
                total=len(dataloader_valid),
            ):
                losses_valid.append(self.do_batch_valid(tensordict))
            self.loss_valid = torch.tensor(losses_valid).mean()

        self.callback(Events.after_epoch)

    def fit(
        self,
        dataloader_train: DataLoader,
        n_epochs: int,
        dataloader_valid: DataLoader = None,
        callbacks: T.List[Callback] = None,
    ):
        if callbacks is not None:
            logger.info(
                f"replacing {self.registered_callbacks=} with {callbacks=}"
            )
            registered_callbacks = self.registered_callbacks
            self.registered_callbacks = callbacks

        self.callback(Events.before_train)
        for self.epoch in tqdm.tqdm(
            range(n_epochs), total=n_epochs, desc="epoch"
        ):
            try:
                self.do_epoch(
                    dataloader_train, dataloader_valid=dataloader_valid
                )
            except CancelFitException:
                break

        self.callback(Events.after_train)

        if callbacks is not None:
            logger.info(f"restoring {registered_callbacks=}")
            self.registered_callbacks = registered_callbacks

    def find_learning_rate(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        lr_find_callback: "LRFinderCallback",
    ):
        self.fit(
            dataloader_train=dataloader,
            n_epochs=n_epochs,
            dataloader_valid=None,
            callbacks=[lr_find_callback],
        )

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        inference = []
        for tensordict in tqdm.tqdm(
            dataloader, total=len(dataloader), desc="batch"
        ):
            inference.append(self.model(tensordict))

        return torch.concat(inference)

    def save(self):
        if self.save_dir is None:
            msg = f"In order to perform lr search `save_dir` needs to be passed to learner to write the model to and load backups from"
            raise ValueError(msg)
        if not self.save_dir.exists():
            msg = f"The path {self.save_dir=} does not exist"
            raise ValueError(msg)

        self.learner_path = self.save_dir / get_learner_name()
        if self.learner_path.exists():
            msg = f"The file {self.learner_path=} already exists."
            raise ValueError(msg)

        logger.info(f"writing learner to {self.learner_path=}")

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, self.learner_path, pickle_protocol=2)
        logger.info(f"done writing")

    def load(self):
        logger.info(f"reading learner from {self.learner_path=}")
        state = torch.load(self.learner_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        logger.info(f"done reading")


@dataclass
class Loss:
    iteration: int
    batch: int
    epoch: int
    loss: float
    loss_valid: float = None


class TrainLossCallback(Callback):
    losses: T.List[Loss]

    def __init__(self):
        self.losses = []

    def after_loss(self, learner: Learner):
        self.losses.append(
            Loss(
                learner.iteration,
                learner.batch,
                learner.epoch,
                float(learner.loss.detach().numpy()),
            )
        )

    def after_epoch(self, learner: Learner):
        if hasattr(learner, "loss_valid"):
            loss_valid = float(learner.loss_valid.detach().numpy())
            self.losses[-1].loss_valid = loss_valid

    def get_losses(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(l) for l in self.losses])

    def get_losses_valid(self) -> pd.DataFrame:
        return self.get_losses().loc[self.get_losses()["loss_valid"].notna(), :]

    def plot(self, window: int = 10):
        fig, ax = plt.subplots(figsize=(10, 4))
        losses = self.get_losses()
        sns.scatterplot(
            data=losses,
            x="iteration",
            y="loss",
            ax=ax,
            label="train",
            alpha=0.1,
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
            sns.lineplot(
                data=losses_valid,
                x="iteration",
                y="loss_valid",
                ax=ax,
                label="valid",
                color="red",
            )

        ax.legend(title="set")
        plt.tight_layout()


class TrainActivationsCallback(Callback):
    activations_history: rnnm_telemetry.ActivationsHistory
    every_n: int
    name_patterns: T.List[str]
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: T.List[str] = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def before_train(self, learner: Learner):
        self.activations_history = rnnm_telemetry.ActivationsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_train(self, learner: Learner):
        self.activations_history.clean()

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
        sns.lineplot(
            data=activations, x="call", y="frac_dead", hue="name", ax=ax
        )
        ax.get_legend().remove()

        plt.tight_layout()


class TrainGradientsCallback(Callback):
    gradients_history: rnnm_telemetry.GradientsHistory
    every_n: int
    name_patterns: T.List[str]
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: T.List[str] = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def before_train(self, learner: Learner):
        self.gradients_history = rnnm_telemetry.GradientsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_train(self, learner: Learner):
        self.gradients_history.clean()

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
        sns.lineplot(
            data=gradients, x="call", y="abs_perc90", hue="name", ax=ax
        )
        ax.get_legend().remove()

        plt.tight_layout()


class TrainParametersCallback(Callback):
    parameters_history: rnnm_telemetry.ParametersHistory
    every_n: int
    name_patterns: T.List[str]
    max_depth_search: int

    def __init__(
        self,
        every_n: int = 1,
        name_patterns: T.List[str] = None,
        max_depth_search: int = 1,
    ):
        self.every_n = every_n
        self.name_patterns = name_patterns
        self.max_depth_search = max_depth_search

    def before_train(self, learner: Learner):
        self.parameters_history = rnnm_telemetry.ParametersHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def after_batch(self, learner: Learner):
        self.parameters_history(learner.iteration)

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
        sns.lineplot(
            data=parameters, x="iter", y="abs_perc90", hue="name", ax=ax
        )
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
    losses: T.List[LossWithLR]

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

    def before_batch(self, learner: Learner):
        self.lr = self.schedule(
            self.start_lr, self.end_lr, learner.iteration / self.num_iterations
        )

        for param_group in learner.optimizer.param_groups:
            param_group["lr"] = self.lr

    def after_batch(self, learner: Learner):
        current_loss = float(learner.loss.detach().numpy())
        current_smooth_loss = float(learner.smooth_loss.detach().numpy())
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

    def before_train(self, learner: Learner):
        # https://github.com/fastai/fastai/blob/43dbef38fe52b8b074d91ee1773e702a1401a486/fastai/callback/schedule.py#L180
        learner.save()

    def after_train(self, learner: Learner):
        # https://github.com/fastai/fastai/blob/43dbef38fe52b8b074d91ee1773e702a1401a486/fastai/callback/schedule.py#L199
        learner.optimizer.zero_grad()
        learner.load()
        learner.learner_path.unlink()

    def get_losses(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(l) for l in self.losses])

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 4))
        lr_find_data = self.get_losses()
        sns.lineplot(
            data=lr_find_data, x="lr", y="smooth_loss", ax=ax, label="smooth"
        )
        sns.lineplot(data=lr_find_data, x="lr", y="loss", ax=ax, label="raw")
        ax.legend(title="loss type")
        ax.set(ylabel="loss")
        plt.tight_layout()


class EveryBatchSchedulerCallback(Callback):
    def __init__(self, scheduler: optim.lr_scheduler.LRScheduler):
        self.scheduler = scheduler

    def after_batch(self, learner: Learner):
        self.scheduler.step()


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int):
        self.patience = patience
        self.n_epochs_worse = 0
        self.best_loss = torch.tensor(torch.inf)

    def after_epoch(self, learner: Learner):
        if not hasattr(learner, "loss_valid"):
            msg = "Learner is missing attribute loss_valid. Did you pass dataloader_valid to Learner.fit?"
            raise ValueError(msg)

        if learner.loss_valid > self.best_loss:
            self.n_epochs_worse += 1
            if self.n_epochs_worse > self.patience:
                raise CancelFitException()
        else:
            self.best_loss = learner.loss_valid
            self.n_epochs_worse = 0
