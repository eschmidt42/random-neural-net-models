# -*- coding: utf-8 -*-

import torch.optim as optim

from random_neural_net_models.learner.callbacks import (
    EarlyStoppingCallback,
    EveryBatchSchedulerCallback,
    LRFinderCallback,
    TrainActivationsCallback,
    TrainGradientsCallback,
    TrainLossCallback,
    TrainParametersCallback,
)
from random_neural_net_models.learner.core import Learner, logger
from random_neural_net_models.learner.interfaces import Events

__all__ = [
    "Learner",
    "LRFinderCallback",
    "TrainLossCallback",
    "EarlyStoppingCallback",
    "TrainGradientsCallback",
    "TrainParametersCallback",
    "TrainActivationsCallback",
    "EveryBatchSchedulerCallback",
    "Events",
    "logger",
]


def set_optimizer_hyperparameter(optimizer: optim.Optimizer, **hyper_params):
    # https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no-lr-sched

    for g in optimizer.param_groups:
        for param_name, param_value in hyper_params.items():
            if param_name not in g:
                raise ValueError(
                    f"was passed {hyper_params=} but did not find '{param_name}' in the parameter group {g=}. Maybe there is a typo in your hyperparameter name?"
                )
            g[param_name] = param_value
