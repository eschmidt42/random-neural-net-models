from abc import ABC
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.modules.loss as torch_loss
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class CallbackEnum(StrEnum):
    train_loss = "train_loss"
    train_activations = "train_activations"
    train_gradients = "train_gradients"
    train_parameters = "train_parameters"
    lr_finder = "lr_finder"
    every_batch_scheduler = "every_batch_scheduler"
    early_stopping = "early_stopping"


class Callback(ABC):
    enum: CallbackEnum


class Events(StrEnum):
    # fastai callback events: https://docs.fast.ai/callback.core.html#events
    after_loss = "after_loss"
    before_train = "before_train"
    after_train = "after_train"
    before_batch = "before_batch"
    after_batch = "after_batch"
    after_epoch = "after_epoch"


class LearnerProtocol(Protocol):
    model: nn.Module
    optimizer: Optimizer
    loss_func: torch_loss._Loss
    loss: torch.Tensor
    losses: torch.Tensor
    smooth_loss: torch.Tensor
    smooth_count: int
    save_dir: Path | None
    loss_valid: torch.Tensor
    device: str

    def register_callback(self, callbacks: list[Callback]):
        pass

    def update_callback(self, events: Events):
        pass

    def call(self, events: Events):
        pass

    def do_batch_train(self, tensordict: TensorDict):
        pass

    def do_batch_valid(self, tensordict: TensorDict):
        pass

    def do_epoch(
        self, dataloader_train: DataLoader, dataloader_valid: DataLoader | None = None
    ):
        pass

    def fit(
        self,
        dataloader_train: DataLoader,
        n_epochs: int,
        dataloader_valid: DataLoader | None = None,
        callbacks: list[Callback] | None = None,
    ):
        pass

    def find_learning_rate(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        lr_find_callback: "LRFinderCallback",
    ):
        pass

    def predict(
        self,
        dataloader: DataLoader,
        component: int | tuple[int] | None = None,
        return_inputs: bool = False,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]
    ):
        pass

    def save(self):
        pass

    def load(self):
        pass
