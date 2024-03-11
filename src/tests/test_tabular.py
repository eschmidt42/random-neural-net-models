# -*- coding: utf-8 -*-
import torch
import pytest
import random_neural_net_models.tabular as rnnm_tab
import random_neural_net_models.data as rnnm_data
import random_neural_net_models.losses as rnnm_loss


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
):  # use_batch_norm: bool
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
