import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.modules.loss as torch_loss
import tqdm
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import random_neural_net_models.utils as utils
from random_neural_net_models.learner.callbacks import (
    CancelFitException,
)
from random_neural_net_models.learner.interfaces import Callback, CallbackEnum, Events

logger = utils.get_logger("learner.py")


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
    save_dir: Path | None
    loss_valid: torch.Tensor
    device: str
    iteration: int
    batch: int
    epoch: int
    learner_path: Path
    registered_callbacks: dict[CallbackEnum, Callback]

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_func: torch_loss._Loss,
        callbacks: list[Callback] | None = None,
        save_dir: Path | None = None,
        device: str = "cpu",
        show_fit_progress: bool = True,
        show_epoch_progress: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.register_callbacks(callbacks)

        if save_dir is not None:
            self.save_dir = save_dir.resolve().absolute()
        else:
            self.save_dir = None
        self.device = device

        self.iteration = 0
        self.smooth_count = 0
        self.smooth_val = torch.tensor(0.0, device="cpu")
        self.smooth_loss = torch.tensor(torch.inf, device="cpu")
        self.losses = torch.tensor([], device="cpu").float()
        self.show_fit_progress = show_fit_progress
        self.show_epoch_progress = show_epoch_progress

    def register_callbacks(self, callbacks: list[Callback] | None = None):
        if callbacks is None:
            self.registered_callbacks = {}
        else:
            self.registered_callbacks = {cb.enum: cb for cb in callbacks}

    def update_callback(self, callback: Callback):
        if not hasattr(self, "registered_callbacks"):
            raise ValueError("'registered_callbacks' is not initialized yet")
        if callback.enum in self.registered_callbacks:
            logger.warning(
                f"overwriting previously registered callback {self.registered_callbacks[callback.enum]} with {callback}"
            )
        else:
            logger.info(f"adding {callback} to registered callbacks")

        self.registered_callbacks[callback.enum] = callback

    def callback(self, event: Events):
        match event:
            case Events.after_loss:
                for cb in self.registered_callbacks.values():
                    cb.after_loss(self)
            case Events.before_train:
                for cb in self.registered_callbacks.values():
                    cb.before_train(self)
            case Events.after_train:
                for cb in self.registered_callbacks.values():
                    cb.after_train(self)
            case Events.before_batch:
                for cb in self.registered_callbacks.values():
                    cb.before_batch(self)
            case Events.after_batch:
                for cb in self.registered_callbacks.values():
                    cb.after_batch(self)
            case Events.after_epoch:
                for cb in self.registered_callbacks.values():
                    cb.after_epoch(self)

    # TODO: this does not produce the expected loss vs lr curve for rumelhart nb - is this correct?
    def _update_smooth_loss(self):
        self.losses = torch.cat((self.losses, torch.tensor([self.loss.detach().cpu()])))

        beta = 0.98
        self.smooth_count += 1
        self.smooth_val = torch.lerp(
            self.loss.detach().mean().cpu(), self.smooth_val, beta
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
        self, dataloader_train: DataLoader, dataloader_valid: DataLoader | None = None
    ):
        self.batch = 0
        total = (
            None
            if not hasattr(dataloader_train.dataset, "__len__")
            else len(dataloader_train)
        )

        for tensordict in tqdm.tqdm(
            dataloader_train,
            total=total,
            desc="batch (train)",
            disable=not self.show_epoch_progress,
        ):
            self.do_batch_train(tensordict.to(self.device))
            self.batch += 1

        if dataloader_valid is not None:
            total = (
                None
                if not hasattr(dataloader_valid.dataset, "__len__")
                else len(dataloader_valid)
            )

            losses_valid = []
            for tensordict in tqdm.tqdm(
                dataloader_valid,
                desc="batch (valid)",
                total=total,
                disable=not self.show_epoch_progress,
            ):
                losses_valid.append(self.do_batch_valid(tensordict.to(self.device)))
            self.loss_valid = torch.tensor(losses_valid).mean().cpu()

        self.callback(Events.after_epoch)

    def fit(
        self,
        dataloader_train: DataLoader,
        n_epochs: int,
        dataloader_valid: DataLoader | None = None,
        callbacks: list[Callback] | None = None,
    ):
        use_callbacks = callbacks is not None
        registered_callbacks = {}
        if use_callbacks:
            logger.info(f"replacing {self.registered_callbacks=} with {callbacks=}")
            registered_callbacks = self.registered_callbacks
            self.register_callbacks(callbacks)

        self.model.to(self.device)
        self.callback(Events.before_train)
        for self.epoch in tqdm.tqdm(
            range(n_epochs),
            total=n_epochs,
            desc="epoch",
            disable=not self.show_fit_progress,
        ):
            try:
                self.do_epoch(dataloader_train, dataloader_valid=dataloader_valid)
            except CancelFitException:
                break

        self.callback(Events.after_train)

        if use_callbacks:
            logger.info(f"restoring {registered_callbacks=}")
            self.registered_callbacks = registered_callbacks

    def find_learning_rate(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        lr_find_callback: Callback,
    ):
        self.fit(
            dataloader_train=dataloader,
            n_epochs=n_epochs,
            dataloader_valid=None,
            callbacks=[lr_find_callback],
        )

    @torch.no_grad()
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
        self.model.eval()
        self.model.to(self.device)
        inference = []
        inputs = []
        total = None if not hasattr(dataloader.dataset, "__len__") else len(dataloader)
        for tensordict in tqdm.tqdm(dataloader, total=total, desc="batch"):
            inference.append(self.model(tensordict.to(self.device)))
            if return_inputs:
                inputs.append(tensordict)

        if isinstance(component, int):
            logger.info(
                f"only using {component=} of each model output (e.g. for VAE only the image)"
            )
            inference = [v[component] for v in inference]
        elif isinstance(component, (tuple, list)):
            logger.info(
                f"only using {component=} of each model output (e.g. for VAE only the image)"
            )
            inference = [[v[c] for v in inference] for c in component]

        if isinstance(component, int) or component is None:
            inference = torch.concat(inference).cpu()

        elif isinstance(component, (tuple, list)):
            inference = [torch.concat(inf).cpu() for inf in inference]

        if return_inputs:
            return inference, torch.cat(inputs)
        return inference

    def save(self):
        if self.save_dir is None:
            msg = "In order to perform lr search `save_dir` needs to be passed to learner to write the model to and load backups from"
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
        logger.info("done writing")

    def load(self):
        logger.info(f"reading learner from {self.learner_path=}")
        state = torch.load(self.learner_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        logger.info("done reading")
