# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import random_neural_net_models.telemetry as telemetry


class SimpleNet(nn.Module):
    def __init__(self, n_in: int = 4, n_out: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        # name the activation attribute 'act' so patterns like ".*act" will match
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(x))


def test_activations_history_collects_stats():
    model = SimpleNet()
    # collect every forward
    ah = telemetry.ActivationsHistory(model, every_n=1, name_patterns=(".*act",))

    # run a few forwards
    for _ in range(3):
        _ = model(torch.randn(5, 4))

    # the history should have entries for the activation module name
    # ChildSearch will name the module simply 'act' here
    assert any(name.endswith("act") for name in ah.stats.keys())
    # find the recorded list and ensure it has 3 entries (one per forward)
    key = [k for k in ah.stats.keys() if k.endswith("act")][0]
    assert len(ah.stats[key]) == 3

    ah.clean()


def test_gradients_history_collects_stats():
    model = SimpleNet()
    gh = telemetry.GradientsHistory(model, every_n=1, name_patterns=(".*lin",))

    # forward + simple loss + backward
    x = torch.randn(4, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # expect gradient stats recorded for the linear module
    assert any(name.endswith("lin") for name in gh.stats.keys())
    key = [k for k in gh.stats.keys() if k.endswith("lin")][0]
    assert len(gh.stats[key]) == 1

    gh.clean()


def test_parameters_history_collects_stats_and_not_initialized():
    model = SimpleNet()

    # parameter history when initialized with a pattern
    ph = telemetry.ParametersHistory(model, every_n=1, name_patterns=(".*lin",))
    ph(1)
    # should have entries for weight and bias under the linear module
    keys = list(ph.stats.keys())
    assert any(k.endswith("lin.weight") for k in keys)
    assert any(k.endswith("lin.bias") for k in keys)
    # values should match direct tensor statistics
    w_key = [k for k in keys if k.endswith("lin.weight")][0]
    stat = ph.stats[w_key][-1]
    direct_mean = model.lin.weight.detach().flatten().float().mean().cpu().item()
    assert abs(stat.mean - direct_mean) < 1e-6

    # parameter history when not initialized should be a no-op
    ph2 = telemetry.ParametersHistory(model, every_n=1, name_patterns=None)
    ph2(1)
    assert len(ph2.stats) == 0
