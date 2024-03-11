# -*- coding: utf-8 -*-
import torch
import pytest
import random_neural_net_models.tabular as rnnm_tab
import random_neural_net_models.data as rnnm_data
import random_neural_net_models.losses as rnnm_loss
import numpy as np


@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("use_activation", [True, False])
def test_layer(use_batch_norm: bool, use_activation: bool):
    layer = rnnm_tab.Layer(
        n_in=10,
        n_out=5,
        use_batch_norm=use_batch_norm,
        use_activation=use_activation,
    )
    x = torch.randn(32, 10)
    output = layer(x)
    assert output.shape == (32, 5)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("n_features", [1, 4])
def test_tabular_model(
    use_batch_norm: bool, n_classes: int, n_features: int
):  # use_batch_norm: bool
    model = rnnm_tab.TabularModel(
        n_hidden=[n_features, 5, n_classes], use_batch_norm=use_batch_norm
    )
    bs = 32
    x = torch.randn(bs, n_features)
    y = torch.randint(low=0, high=n_classes, size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, n_classes)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.CrossEntropyXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)


@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("n_features", [1, 4])
def test_highlevel_tabular_model_for_classification(
    use_batch_norm: bool, n_classes: int, n_features: int
):
    model = rnnm_tab.TabularModelClassification(
        n_features=n_features,
        n_hidden=[5],
        n_classes=n_classes,
        use_batch_norm=use_batch_norm,
    )
    bs = 32
    x = torch.randn(bs, n_features)
    y = torch.randint(low=0, high=n_classes, size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, n_classes)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.CrossEntropyXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (-1.0, 5)])
def test_standard_normal_scaler(mean: float, std: float):
    scaler = rnnm_tab.StandardNormalScaler(mean=mean, std=std)
    x = torch.randn(32, 10)
    output = scaler(x)
    assert output.shape == (32, 10)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, x * std + mean)


@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("mean,std", [(-1, 1), (0, 10)])
@pytest.mark.parametrize("n_features", [1, 4])
def test_highlevel_tabular_model_for_regression(
    use_batch_norm: bool, mean: float, std: float, n_features: int
):
    model = rnnm_tab.TabularModelRegression(
        n_features=n_features,
        n_hidden=[5],
        mean=mean,
        std=std,
        use_batch_norm=use_batch_norm,
    )
    bs = 32
    x = torch.randn(bs, n_features)
    y = torch.randn(size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, 1)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.MSELossXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)


def test_impute_missingness():
    n_features = 3
    imputer = rnnm_tab.ImputeMissingness(
        n_features, bias_source=rnnm_tab.BiasSources.zero
    )

    X = torch.tensor(
        [[1, 2, float("inf")], [3, float("inf"), 6], [float("inf"), 5, 6]],
        dtype=torch.float32,
    )
    expected_output = torch.tensor(
        [[1, 2, 0, 0, 0, 1], [3, 0, 6, 0, 1, 0], [0, 5, 6, 1, 0, 0]],
        dtype=torch.float32,
    )

    output = imputer(X)

    assert torch.allclose(output, expected_output)
    assert output.shape == (3, 6)
    assert torch.isfinite(output[:, :n_features]).all()
    assert torch.isfinite(output).all()


def test_make_missing():
    X = np.random.randn(1_000, 10)
    p_missing = 0.1

    X_miss, mask = rnnm_tab.make_missing(X, p_missing)

    assert X_miss.shape == X.shape
    assert mask.shape == X.shape

    # Check if missing values are correctly set to infinity
    assert np.all(X_miss[mask] == float("inf"))

    # Check if non-missing values are unchanged
    assert np.all(X_miss[~mask] == X[~mask])

    # Check if the proportion of missing values is approximately equal to p_missing
    assert np.isclose(np.sum(mask) / X.size, p_missing, atol=0.01)


@pytest.mark.parametrize(
    "bias_source", [rnnm_tab.BiasSources.zero, rnnm_tab.BiasSources.normal]
)
def test_highlevel_tabular_model_for_classification_with_missingness(
    bias_source: rnnm_tab.BiasSources,
):
    n_classes = 2
    n_features = 5

    model = rnnm_tab.TabularModelClassification(
        n_features=n_features,
        n_hidden=[5],
        n_classes=n_classes,
        use_batch_norm=False,
        do_impute=True,
        impute_bias_source=bias_source,
    )
    bs = 32
    x = np.random.randn(bs, n_features)
    x, _ = rnnm_tab.make_missing(x, p_missing=0.5)
    x = torch.from_numpy(x).float()
    y = torch.randint(low=0, high=n_classes, size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, n_classes)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.CrossEntropyXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "bias_source", [rnnm_tab.BiasSources.zero, rnnm_tab.BiasSources.normal]
)
def test_highlevel_tabular_model_for_regression_with_missingness(
    bias_source: rnnm_tab.BiasSources,
):
    n_features = 5
    mean = 0.0
    std = 1.0

    model = rnnm_tab.TabularModelRegression(
        n_features=n_features,
        n_hidden=[5],
        mean=mean,
        std=std,
        use_batch_norm=False,
        do_impute=True,
        impute_bias_source=bias_source,
    )

    bs = 32
    x = np.random.randn(bs, n_features)
    x, _ = rnnm_tab.make_missing(x, p_missing=0.5)
    x = torch.from_numpy(x).float()
    y = torch.randn(size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, 1)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.MSELossXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)
