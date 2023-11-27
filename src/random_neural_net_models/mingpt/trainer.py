# -*- coding: utf-8 -*-
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import typing as T
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

import random_neural_net_models.mingpt.utils as utils

CN = utils.CfgNode


class Trainer:
    @staticmethod
    def get_config(
        device: str = "auto",
        num_workers: int = 4,
        max_iters: int = None,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        betas: T.Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        grad_norm_clip: float = 1.0,
    ) -> CN:
        C = CN()
        # device to train on
        C.device = device
        # dataloder parameters
        C.num_workers = num_workers
        # optimizer parameters
        C.max_iters = max_iters
        C.batch_size = batch_size
        C.learning_rate = learning_rate
        C.betas = betas
        C.weight_decay = weight_decay  # only applied on matmul weights
        C.grad_norm_clip = grad_norm_clip
        return C

    def __init__(self, config, model, train_dataset):
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
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_norm_clip
            )
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if (
                config.max_iters is not None
                and self.iter_num >= config.max_iters
            ):
                break
