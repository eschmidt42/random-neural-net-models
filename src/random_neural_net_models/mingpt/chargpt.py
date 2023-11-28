# -*- coding: utf-8 -*-
"""
Trains a character-level language model.
"""

import torch
from pydantic.dataclasses import dataclass
from torch.utils.data import Dataset

import random_neural_net_models.mingpt.configs as configs
import random_neural_net_models.utils as utils

logger = utils.get_logger("mingpt.char")


@dataclass(frozen=True)
class DataConfig:
    block_size: int


@dataclass(frozen=True)
class CharConfig:
    system: configs.SystemConfig
    data: DataConfig
    model: configs.ModelConfig
    trainer: configs.TrainerConfig


def get_config(vocab_size: int, block_size: int, max_iters: int) -> CharConfig:
    return CharConfig(
        system=configs.SystemConfig(seed=3407, work_dir="./out/chargpt"),
        data=DataConfig(block_size=128),
        model=configs.ModelConfig(
            model_type="gpt-mini",
            vocab_size=vocab_size,
            block_size=block_size,
        ),
        trainer=configs.TrainerConfig(
            max_iters=max_iters,
            learning_rate=5e-4,  # the model we're using is so small that we can go a bit faster
        ),
    )


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_config() -> DataConfig:
        return DataConfig(block_size=128)

    def __init__(self, config: DataConfig, data: str):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        logger.info(
            f"data has {data_size:_d} characters, {vocab_size:_d} unique."
        )

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx: int):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
