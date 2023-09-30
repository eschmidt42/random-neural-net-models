# -*- coding: utf-8 -*-
import pytest
import torch.nn as nn
from sklearn.datasets import fetch_openml
from torch.optim import SGD
from torch.utils.data import DataLoader

import random_neural_net_models.convolution_lecun1990 as conv_lecun1990
import random_neural_net_models.telemetry as telemetry


@pytest.fixture(scope="module")
def cnn_model_telemetry() -> telemetry.ModelTelemetry:
    """Telemetry for a CNN model."""

    mnist = fetch_openml("mnist_784", version=1, cache=True, parser="auto")

    X = mnist["data"].iloc[:100]
    y = mnist["target"].iloc[:100]

    ds = conv_lecun1990.DigitsDataset(X, y)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False)

    model = conv_lecun1990.Model(lecun_init=True, lecun_act=True)
    model = telemetry.ModelTelemetry(
        model,
        activations_every_n=1,
        gradients_every_n=1,
        parameters_every_n=1,
        activations_name_patterns=(".*act.*",),
        gradients_name_patterns=(r"conv\d$", r"lin\d"),
        parameters_name_patterns=(r"conv\d$", r"lin\d"),
    )
    model.double()

    opt = SGD(
        model.parameters(),
        lr=0.1,
    )
    loss_func = nn.MSELoss()

    _iter = 0
    xb, yb = next(iter(dataloader))
    xb = xb.cpu()
    yb = yb.cpu()
    yb = conv_lecun1990.densify_y(yb)
    loss = loss_func(model(xb), yb)

    opt.zero_grad()
    loss.backward()
    opt.step()

    model.loss_history_train(loss, _iter)
    model.parameter_history(_iter)

    return model


# test activations stats are present
def test_cnn_model_telemetry_activations(
    cnn_model_telemetry: telemetry.ModelTelemetry,
):
    assert cnn_model_telemetry.activations_history.stats is not None
    for _, stats in cnn_model_telemetry.activations_history.stats.items():
        assert len(stats) == 1


# test parameter stats are present
def test_cnn_model_telemetry_parameters(
    cnn_model_telemetry: telemetry.ModelTelemetry,
):
    assert cnn_model_telemetry.parameter_history.stats is not None
    for _, stats in cnn_model_telemetry.parameter_history.stats.items():
        assert len(stats) == 1


# test gradient stats are present
def test_cnn_model_telemetry_gradients(
    cnn_model_telemetry: telemetry.ModelTelemetry,
):
    assert cnn_model_telemetry.gradients_history.stats is not None
    for _, stats in cnn_model_telemetry.gradients_history.stats.items():
        assert len(stats) == 1
