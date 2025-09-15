# -*- coding: utf-8 -*-

import typing as T
from dataclasses import asdict

from pydantic.dataclasses import dataclass

import random_neural_net_models.utils as utils

logger = utils.get_logger("mingpt.utils")


@dataclass(frozen=True)
class ModelConfig:
    "Configuration for gpt model"

    model_type: str = "gpt"
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None
    vocab_size: int | None = None
    block_size: int | None = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


@dataclass(frozen=True)
class TrainerConfig:
    "Configuration for gpt trainer"

    device: str = "auto"
    num_workers: int = 4
    max_iters: int | None = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: T.Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0


def get_modified_config_dict(
    config: T.Union[ModelConfig, TrainerConfig], verbose: bool = False, **kwargs
) -> dict:
    logger.info(f"modifying: {config} with {kwargs}")
    vals = asdict(config)
    new_keys = set(kwargs.keys()).difference(set(vals.keys()))
    updated_keys = set(vals.keys()).intersection(set(kwargs.keys()))
    if verbose:
        if len(new_keys) > 0:
            msg = ", ".join([f"{k}: {kwargs[k]}" for k in new_keys])
            logger.info(f"adding: {msg}")
        if len(updated_keys) > 0:
            msg = ", ".join(
                [f"{k}: {config.__getattribute__(k)} -> {kwargs[k]}" for k in new_keys]
            )
            logger.info(f"updating: {msg}")

    vals.update(**kwargs)
    return vals


def get_modified_trainer_config(
    config: TrainerConfig, verbose: bool = False, **kwargs
) -> TrainerConfig:
    vals = get_modified_config_dict(config, verbose, **kwargs)
    return TrainerConfig(**vals)


def get_modified_model_config(
    config: ModelConfig, verbose: bool = False, **kwargs
) -> ModelConfig:
    vals = get_modified_config_dict(config, verbose, **kwargs)
    return ModelConfig(**vals)


@dataclass(frozen=True)
class SystemConfig:
    seed: int
    work_dir: str
