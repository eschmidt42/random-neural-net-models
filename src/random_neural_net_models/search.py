# -*- coding: utf-8 -*-
import re
import typing as T
from dataclasses import dataclass

import torch.nn as nn

import random_neural_net_models.utils as utils

logger = utils.get_logger("search.py")


@dataclass
class NamedModule:
    name: str
    module: nn.Module


def collate_named_modules(
    module: nn.Module,
    module_name: str = None,
    depth: int = 0,
    max_depth: int = 3,
    collated_named_modules: T.List[NamedModule] = None,
) -> T.List[NamedModule]:
    if depth >= max_depth:
        return collated_named_modules
    elif len(list(module.children())) == 0:
        return collated_named_modules

    if collated_named_modules is None:
        collated_named_modules = []

    depth += 1

    for child_name, child_module in module.named_children():
        child_name = (
            f"{module_name}.{child_name}"
            if module_name is not None
            else child_name
        )
        collated_named_modules.append(NamedModule(child_name, child_module))

        collated_named_modules = collate_named_modules(
            child_module,
            module_name=child_name,
            depth=depth,
            max_depth=max_depth,
            collated_named_modules=collated_named_modules,
        )

    return collated_named_modules


def find_named_module(
    collated_named_modules: T.List[NamedModule], pattern: str
) -> T.List[NamedModule]:
    matches = []
    for named_module in collated_named_modules:
        if re.match(pattern, named_module.name) is not None:
            matches.append(named_module)
    if len(matches) > 0:
        return matches
    else:
        logger.error(f"Module with {pattern=} not found")
        return []


class ChildSearch:
    def __init__(self, module: nn.Module, max_depth: int = 3):
        self.module = module
        self.collated_named_modules = collate_named_modules(
            module, max_depth=max_depth
        )

    def __call__(self, *patterns: T.Tuple[str]) -> T.List[NamedModule]:
        matches = []
        for pattern in patterns:
            matches.extend(
                find_named_module(self.collated_named_modules, pattern)
            )
        return matches

    @property
    def names(self) -> T.List[str]:
        return [
            named_module.name for named_module in self.collated_named_modules
        ]
