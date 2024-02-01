# -*- coding: utf-8 -*-
import datetime
import shutil
import typing as T
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.loss as torch_loss
import tqdm
from pydantic.dataclasses import dataclass
from tensordict import TensorDict, tensorclass
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

import random_neural_net_models.telemetry as rnnm_telemetry

# fastai callback events: https://docs.fast.ai/callback.core.html#events


class Callback:
    ...


class CancelFitException(Exception):
    ...


class Events(Enum):
    after_loss = "after_loss"
    before_train = "before_train"
    after_train = "after_train"
    before_batch = "before_batch"
    after_batch = "after_batch"


# TODO: add hyperparam scheduler https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
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
            c for c in self.registered_callbacks if hasattr(c, event.value)
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

    def do_epoch(self, dataloader: DataLoader):
        for self.batch, tensordict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataloader), desc="batch"
        ):
            self.do_batch_train(tensordict)

    def fit(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        callbacks: T.List[Callback] = None,
    ):
        # TODO: add validation loop

        if callbacks is not None:
            print(f"replacing {self.registered_callbacks=} with {callbacks=}")
            registered_callbacks = self.registered_callbacks
            self.registered_callbacks = callbacks

        self.model.train()
        self.callback(Events.before_train)
        for self.epoch in tqdm.tqdm(
            range(n_epochs), total=n_epochs, desc="epoch"
        ):
            try:
                self.do_epoch(dataloader)
            except CancelFitException:
                break
            # for self.batch, tensordict in tqdm.tqdm(
            #     enumerate(dataloader), total=len(dataloader), desc="batch"
            # ):
            #     self.callback(Events.before_batch)
            #     self.optimizer.zero_grad()
            #     inference = self.model(tensordict)
            #     self.loss = self.loss_func(inference, tensordict)
            #     self.callback(Events.after_loss)
            #     self.loss.backward()
            #     self.optimizer.step()
            #     self.callback(Events.after_batch)
            #     self.iteration += 1

        self.callback(Events.after_train)

        if callbacks is not None:
            print(f"restoring {registered_callbacks=}")
            self.registered_callbacks = registered_callbacks

    def find_learning_rate(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        lr_find_callback: "LRFinderCallback",
    ):
        self.fit(dataloader, n_epochs, [lr_find_callback])

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

        print(f"writing learner to {self.learner_path=}")

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, self.learner_path, pickle_protocol=2)
        print(f"done writing")

    def load(self):
        print(f"reading learner from {self.learner_path=}")
        state = torch.load(self.learner_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        print(f"done reading")


@dataclass
class Loss:
    iteration: int
    batch: int
    epoch: int
    loss: float


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

    def get_losses(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(l) for l in self.losses])


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
            tmp = pd.DataFrame([asdict(v) for v in stats])
            tmp["name"] = name
            tmp["call"] = np.arange(len(tmp))
            all_stats.append(tmp)
        return pd.concat(all_stats, ignore_index=True)


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
