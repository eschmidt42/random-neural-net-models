# -*- coding: utf-8 -*-

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

import random_neural_net_models.search as search
import random_neural_net_models.utils as utils

logger = utils.get_logger("telemetry.py")


@dataclass
class ActivationStats:
    iteration: int
    mean: float
    std: float
    frac_dead: float


@dataclass
class ParameterStats:
    iteration: int
    mean: float
    std: float
    abs_perc90: float


@dataclass
class GradientStats:
    iteration: int
    mean: float
    std: float
    abs_perc90: float
    max: float
    frac_dead: float


class History:
    not_initialized: bool
    hooks: dict[str, RemovableHandle]
    child_search: search.ChildSearch
    every_n: int
    modules: list[search.NamedModule]

    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
        max_depth_search: int = 3,
    ):
        self.not_initialized = name_patterns is None or len(name_patterns) == 0
        if self.not_initialized:
            logger.info("Not collecting history.")
            return

        self.child_search = search.ChildSearch(model, max_depth=max_depth_search)
        self.every_n = every_n
        if name_patterns is None:
            raise ValueError(f"{name_patterns=} should not be None here.")
        self.modules = self.child_search(*name_patterns)

    @property
    def name_matches(self):
        if self.not_initialized:
            return []
        return [a.name for a in self.modules]

    def clean(self):
        if self.not_initialized:
            logger.info("No activation hooks to clean up.")
            return
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


class ActivationsHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
        max_depth_search: int = 3,
        stats: dict[str, list[ActivationStats]] | None = None,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)

        self.stats: dict[str, list[ActivationStats]] = (
            stats if stats else defaultdict(list)
        )
        self.register_hooks(model, name_patterns, max_depth_search)

    def register_hooks(
        self,
        model: nn.Module,
        name_patterns: tuple[str, ...] | None = None,
        max_depth_search: int = 3,
    ):
        self.hooks = defaultdict(None)
        if self.stats is None or (len(self.stats) == 0):
            iteration = 0
        else:
            k = list(self.stats.keys())[0]
            iteration = self.stats[k][-1].iteration

        for named_module in self.modules:
            cas = CollectorActivationStats(
                self, named_module.name, iteration, every_n=self.every_n
            )
            self.hooks[named_module.name] = named_module.module.register_forward_hook(
                cas  # type: ignore
            )

        logger.info(
            f"Will collect activiation history every {self.every_n}th iteration for: {self.name_matches=}"
        )


class GradientsHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
        max_depth_search: int = 3,
        stats: dict[str, list[GradientStats]] | None = None,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)
        self.stats: dict[str, list[GradientStats]] = (
            defaultdict(list) if stats is None else stats
        )
        self.register_hooks(model, name_patterns, max_depth_search)

    def register_hooks(
        self,
        model: nn.Module,
        name_patterns: tuple[str] | None = None,
        max_depth_search: int = 3,
    ):
        self.child_search = search.ChildSearch(model, max_depth=max_depth_search)
        if name_patterns is None:
            raise ValueError(f"{name_patterns=} should not be None here.")
        self.modules = self.child_search(*name_patterns)
        self.hooks = defaultdict(None)

        if self.stats is None or len(self.stats) == 0:
            iteration = 0
        else:
            k = list(self.stats.keys())[0]
            iteration = self.stats[k][-1].iteration

        for named_module in self.modules:
            cgs = CollectorGradientStats(
                self, named_module.name, iteration, every_n=self.every_n
            )
            self.hooks[named_module.name] = (
                named_module.module.register_full_backward_hook(cgs)  # type: ignore
            )

        logger.info(
            f"Will collect gradient history every {self.every_n}th iteration for: {self.name_matches=}"
        )


class HistoryException(Exception):
    pass


class ParametersHistory(History):
    def __init__(
        self,
        model: nn.Module,
        every_n: int = 1,
        name_patterns: tuple[str] | None = None,
        max_depth_search: int = 3,
        stats: dict[str, list[ParameterStats]] | None = None,
    ):
        super().__init__(model, every_n, name_patterns, max_depth_search)
        self.stats: dict[str, list[ParameterStats]] = (
            defaultdict(list) if stats is None else stats
        )
        self.every_n = every_n
        logger.info(
            f"Will collect parameter history every {self.every_n}th iteration for: {self.name_matches=}"
        )

    def __call__(self, _iter: int):
        if self.not_initialized or _iter % self.every_n != 0:
            return

        for named_module in self.modules:
            state_dict = named_module.module.state_dict()
            for name, parameter_values in state_dict.items():
                parameter_values = parameter_values.detach().flatten().float()
                mean = parameter_values.mean().cpu().item()
                std = parameter_values.std().cpu().item()
                abs_perc90 = parameter_values.abs().quantile(0.9).cpu().item()
                self.stats[f"{named_module.name}.{name}"].append(
                    ParameterStats(_iter, mean, std, abs_perc90)
                )


class CollectorActivationStats:
    def __init__(
        self,
        hook: ActivationsHistory,
        name: str,
        iteration: int,
        every_n: int = 1,
        threshold_dead=1e-6,
    ):
        self.hook = hook
        self.name = name
        self.every_n = every_n
        self.iteration = iteration
        self.threshold_dead = threshold_dead

    def __call__(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        self.iteration += 1
        if self.iteration % self.every_n != 0:
            return

        acts = output.detach().flatten()
        mean = acts.mean().cpu().item()
        std = acts.std().cpu().item()
        frac_dead = (acts.abs() < self.threshold_dead).sum().cpu().item() / len(acts)

        self.hook.stats[self.name].append(
            ActivationStats(int(self.iteration), mean, std, frac_dead)
        )


class CollectorGradientStats:
    def __init__(
        self,
        hook: GradientsHistory,
        name: str,
        iteration: int,
        every_n: int = 1,
        threshold_dead: float = 1e-8,
    ):
        self.hook = hook
        self.name = name
        self.every_n = every_n
        self.iteration = iteration
        self.threshold_dead = threshold_dead

    def __call__(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        self.iteration += 1
        if self.iteration % self.every_n != 0:
            return

        vals = output[0].detach().flatten()
        mean = vals.mean().cpu().item()
        std = vals.std().cpu().item()
        abs_perc90 = vals.abs().quantile(0.9).cpu().item()
        _max = vals.abs().max().cpu().item()
        frac_dead = (vals.abs() < self.threshold_dead).sum().cpu().item() / len(vals)

        self.hook.stats[self.name].append(
            GradientStats(int(self.iteration), mean, std, abs_perc90, _max, frac_dead)
        )
