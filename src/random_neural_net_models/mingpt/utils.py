# -*- coding: utf-8 -*-

import random
from enum import StrEnum

import numpy as np
import torch


class SetsEnum(StrEnum):
    train = "train"
    test = "test"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
