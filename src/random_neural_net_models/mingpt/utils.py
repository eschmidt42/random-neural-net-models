# -*- coding: utf-8 -*-

import json
import os
import random
import sys
from enum import Enum

import numpy as np
import torch

# Create a named Enum class so multiprocessing's pickler can import it by name.
# multiprocessing/pickle will try to look up the class as an attribute on the
# module (e.g. random_neural_net_models.mingpt.utils.Sets). Defining the
# class as `Sets` at module-level makes it importable/picklable. Keep the
# uppercase `SETS` alias for backwards compatibility with existing code.
Sets = Enum("Sets", "train test")
SETS = Sets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """monotonous bookkeeping"""
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))
