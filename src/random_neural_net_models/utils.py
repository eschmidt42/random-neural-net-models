# -*- coding: utf-8 -*-
import logging
import random

import numpy as np
import torch
from rich.logging import RichHandler


def get_logger(name: str = "rich", level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(name)s: %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = get_logger("utils.py")


def make_deterministic(seed: int = 42):
    # from https://krokotsch.eu/posts/deep-learning-unit-tests/
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Built-in Python
    random.seed(seed)
