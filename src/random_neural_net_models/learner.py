# -*- coding: utf-8 -*-
import typing as T
from dataclasses import asdict
from enum import Enum

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


class Events(Enum):
    after_loss = "after_loss"
    on_train_begin = "on_train_begin"
    on_train_end = "on_train_end"
    on_batch_end = "on_batch_end"


class Learner:
    model: nn.Module
    optimizer: Optimizer
    loss_func: torch_loss._Loss

    # TODO: add hyperparam scheduler https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # TODO: add logging of activations, weights, gradient and losses using wandb

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_func: torch_loss._Loss,
        callbacks: T.List[Callback] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        if callbacks is None:
            self.registered_callbacks = []
        else:
            self.registered_callbacks = callbacks
        self.iteration = 0

    def callback(self, event: Events):
        relevant_callbacks = [
            c for c in self.registered_callbacks if hasattr(c, event.value)
        ]
        for callback in relevant_callbacks:
            getattr(callback, event.value)(self)

    def fit(self, dataloader: DataLoader, n_epochs: int):
        # TODO: add validation loop
        self.model.train()
        self.callback(Events.on_train_begin)
        for self.epoch in tqdm.tqdm(
            range(n_epochs), total=n_epochs, desc="epoch"
        ):
            for self.batch, tensordict in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), desc="batch"
            ):
                self.optimizer.zero_grad()
                inference = self.model(tensordict)
                self.loss = self.loss_func(inference, tensordict)
                self.callback(Events.after_loss)
                self.loss.backward()
                self.optimizer.step()
                self.iteration += 1
                self.callback(Events.on_batch_end)

        self.callback(Events.on_train_end)

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        inference = []
        for tensordict in tqdm.tqdm(
            dataloader, total=len(dataloader), desc="batch"
        ):
            inference.append(self.model(tensordict))

        return torch.concat(inference)


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

    def get_losses(self) -> np.ndarray:
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

    def on_train_begin(self, learner: Learner):
        self.activations_history = rnnm_telemetry.ActivationsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def on_train_end(self, learner: Learner):
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

    def on_train_begin(self, learner: Learner):
        self.gradients_history = rnnm_telemetry.GradientsHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def on_train_end(self, learner: Learner):
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

    def on_train_begin(self, learner: Learner):
        self.parameters_history = rnnm_telemetry.ParametersHistory(
            learner.model,
            every_n=self.every_n,
            name_patterns=self.name_patterns,
            max_depth_search=self.max_depth_search,
        )

    def on_batch_end(self, learner: Learner):
        self.parameters_history(learner.iteration)

    def get_stats(self) -> pd.DataFrame:
        all_stats = []
        for name, stats in self.parameters_history.stats.items():
            tmp = pd.DataFrame([asdict(v) for v in stats])
            tmp["name"] = name
            tmp["call"] = np.arange(len(tmp))
            all_stats.append(tmp)
        return pd.concat(all_stats, ignore_index=True)
