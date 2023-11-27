# -*- coding: utf-8 -*-
"""
Trains a character-level language model.
"""

import torch
from torch.utils.data import Dataset

import random_neural_net_models.mingpt.model as gpt_model
import random_neural_net_models.mingpt.trainer as trainer
import random_neural_net_models.mingpt.utils as gpt_utils

GPT = gpt_model.GPT
Trainer = trainer.Trainer
set_seed = gpt_utils.set_seed
setup_logging = gpt_utils.setup_logging
CN = gpt_utils.CfgNode


# -----------------------------------------------------------------------------


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/chargpt"

    # data
    C.data = CharDataset.get_config()

    # model
    C.model = GPT.get_config()
    C.model.model_type = "gpt-mini"

    # trainer
    C.trainer = Trainer.get_config()
    C.trainer.learning_rate = (
        5e-4  # the model we're using is so small that we can go a bit faster
    )

    return C


# -----------------------------------------------------------------------------


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

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

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
