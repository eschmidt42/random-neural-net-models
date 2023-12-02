# -*- coding: utf-8 -*-
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import typing as T
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import random_neural_net_models.mingpt.configs as configs
import random_neural_net_models.mingpt.model as gpt_model
import random_neural_net_models.utils as utils

logger = utils.get_logger("mingpt.trainer")


class Trainer:
    @staticmethod
    def get_config(**kwargs) -> configs.TrainerConfig:
        return configs.TrainerConfig(**kwargs)

    def __init__(
        self,
        config: configs.TrainerConfig,
        model: gpt_model.GPT,
        train_dataset: Dataset,
    ):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        logger.info(f"running on device: {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback: T.Callable):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback: T.Callable):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        # setup the optimizer
        self.optimizer = self.model.configure_optimizers(self.config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        self.do_train(train_loader)

    def _get_batch(
        self, train_loader: DataLoader
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(train_loader)
            batch = next(self.data_iter)
        batch = [t.to(self.device) for t in batch]
        x, y = batch
        return x, y

    def do_train(self, train_loader: DataLoader):
        self.model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        self.data_iter = iter(train_loader)
        while True:
            x, y = self._get_batch(train_loader)

            # forward the model
            _, self.loss = self.model(x, y)

            # backprop and update the parameters
            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_norm_clip
            )
            self.optimizer.step()

            # some bookkeeping
            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if (
                self.config.max_iters is not None
                and self.iter_num >= self.config.max_iters
            ):
                break
