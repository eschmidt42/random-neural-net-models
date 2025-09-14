# -*- coding: utf-8 -*-
import typing as T

import numpy as np
import pandas as pd
import pytest
import torch

import random_neural_net_models.data as rnnm_data
import random_neural_net_models.losses as rnnm_loss
import random_neural_net_models.tabular as rnnm_tab


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
def test_tabular_model(use_batch_norm: bool, n_classes: int, n_features: int):
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


@pytest.mark.parametrize(
    "n_features,cols_with_missing",
    [
        (1, (0,)),
        (4, (0,)),
        (4, (0, 3)),
        (4, (0, 1, 2, 3)),
    ],
)
def test_tabular_model_missingness(n_features: int, cols_with_missing: T.Tuple[int]):
    n_classes = 3
    model = rnnm_tab.TabularModel(
        n_hidden=[n_features, 5, n_classes],
        use_batch_norm=True,
        do_impute=True,
        cols_with_missing=cols_with_missing,
    )
    bs = 32
    x = torch.randn(bs, n_features)
    x[0, cols_with_missing] = float("inf")
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


@pytest.mark.parametrize(
    "cols_with_missing,expected_output",
    [
        (
            (0, 1, 2),
            torch.tensor(
                [[1, 2, 0, 0, 0, 1], [3, 0, 6, 0, 1, 0], [0, 5, 6, 1, 0, 0]],
                dtype=torch.float32,
            ),
        ),
        (
            (0, 1),
            torch.tensor(
                [[1, 2, float("inf"), 0, 0], [3, 0, 6, 0, 1], [0, 5, 6, 1, 0]],
                dtype=torch.float32,
            ),
        ),
        (
            (0,),
            torch.tensor(
                [
                    [1, 2, float("inf"), 0],
                    [3, float("inf"), 6, 0],
                    [0, 5, 6, 1],
                ],
                dtype=torch.float32,
            ),
        ),
        (
            (1,),
            torch.tensor(
                [
                    [1, 2, float("inf"), 0],
                    [3, 0, 6, 1],
                    [float("inf"), 5, 6, 0],
                ],
                dtype=torch.float32,
            ),
        ),
        (
            (2,),
            torch.tensor(
                [
                    [1, 2, 0, 1],
                    [3, float("inf"), 6, 0],
                    [float("inf"), 5, 6, 0],
                ],
                dtype=torch.float32,
            ),
        ),
    ],
)
def test_impute_missingness(
    cols_with_missing: T.Tuple[int], expected_output: torch.Tensor
):
    n_features = len(cols_with_missing)
    imputer = rnnm_tab.ImputeMissingness(
        cols_with_missing=cols_with_missing,
        bias_source=rnnm_tab.BiasSources.zero,
    )

    X = torch.tensor(
        [[1, 2, float("inf")], [3, float("inf"), 6], [float("inf"), 5, 6]],
        dtype=torch.float32,
    )

    output = imputer(X)

    assert output.shape == (3, X.shape[1] + n_features)
    assert torch.allclose(output, expected_output)
    assert torch.isfinite(output[:, cols_with_missing]).all()
    if n_features == X.shape[1]:
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("value", [float("inf"), -1])
def test_make_missing(value):
    X = np.random.randn(1_000, 10)
    p_missing = 0.1

    X_miss, mask = rnnm_tab.make_missing(X, value, p_missing)

    assert X_miss.shape == X.shape
    assert mask.shape == X.shape

    # Check if missing values are correctly set to infinity
    assert np.all(X_miss[mask] == value)

    # Check if non-missing values are unchanged
    assert np.all(X_miss[~mask] == X[~mask])

    # Check if the proportion of missing values is approximately equal to p_missing
    assert np.isclose(np.sum(mask) / X.size, p_missing, atol=0.01)


@pytest.mark.parametrize("cols", [(0,), (1, 2), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)])
def test_make_missing_given_cols(cols: T.Tuple[int]):
    X = np.random.randn(1_000, 10)
    p_missing = 0.1

    X_miss, mask = rnnm_tab.make_missing(
        X, float("inf"), p_missing, cols_with_missing=cols
    )

    assert X_miss.shape == X.shape
    assert mask.shape == (X.shape[0], len(cols))

    missinginess = np.logical_not(np.isfinite(X_miss)).any(axis=0)
    assert missinginess[np.array(cols)].all()
    other_cols = [i for i in list(range(X.shape[1])) if i not in cols]
    if len(other_cols) > 0:
        assert not missinginess[np.array(other_cols)].any()


@pytest.mark.parametrize(
    "bias_source", [rnnm_tab.BiasSources.zero, rnnm_tab.BiasSources.normal]
)
def test_highlevel_tabular_model_for_classification_with_missingness(
    bias_source: rnnm_tab.BiasSources,
):
    n_classes = 2
    n_features = 5
    cols_with_missing = (0, 2)

    model = rnnm_tab.TabularModelClassification(
        n_features=n_features,
        n_hidden=[5],
        n_classes=n_classes,
        use_batch_norm=False,
        do_impute=True,
        impute_bias_source=bias_source,
        cols_with_missing=cols_with_missing,
    )
    bs = 32
    x = np.random.randn(bs, n_features)

    x, _ = rnnm_tab.make_missing_numerical(
        x, p_missing=0.5, cols_with_missing=cols_with_missing
    )
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
    cols_with_missing = (0, 2)

    model = rnnm_tab.TabularModelRegression(
        n_features=n_features,
        n_hidden=[5],
        mean=mean,
        std=std,
        use_batch_norm=False,
        do_impute=True,
        impute_bias_source=bias_source,
        cols_with_missing=cols_with_missing,
    )

    bs = 32
    x = np.random.randn(bs, n_features)
    x, _ = rnnm_tab.make_missing_numerical(
        x, p_missing=0.5, cols_with_missing=cols_with_missing
    )
    x = torch.from_numpy(x).float()
    y = torch.randn(size=(bs,))
    input = rnnm_data.XyBlock(x=x, y=y, batch_size=[bs])
    inference = model(input)
    assert inference.shape == (bs, 1)
    assert torch.isfinite(inference).all()

    loss_fn = rnnm_loss.MSELossXy()
    loss = loss_fn(inference, input)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "s, expected_output, expected_mapping",
    [
        (
            pd.Series(["a", "b", "c", "d"]),
            pd.Series([0, 1, 2, 3]),
            {"a": 0, "b": 1, "c": 2, "d": 3},
        ),
        (
            pd.Series(["a", "b", "a", "c", "b"]),
            pd.Series([0, 1, 0, 2, 1]),
            {"a": 0, "b": 1, "c": 2},
        ),
    ],
)
def test_make_string_series_to_int(s, expected_output, expected_mapping):
    output, mapping = rnnm_tab.make_string_series_to_int(s)
    assert output.equals(expected_output)
    assert mapping == expected_mapping


def test_make_string_columns_to_int():
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c"],
            "col2": ["x", "y", "z"],
            "col3": [0.1, 0.2, -0.9],
        }
    )
    categorical_columns = ["col1", "col2"]

    df_int, maps_str2int = rnnm_tab.make_string_columns_to_int(df, categorical_columns)

    assert isinstance(df_int, pd.DataFrame)
    assert isinstance(maps_str2int, dict)

    assert df_int.shape == (3, 3)
    assert df_int["col1"].dtype == int
    assert df_int["col2"].dtype == int

    assert df_int["col1"].tolist() == [0, 1, 2]
    assert df_int["col2"].tolist() == [0, 1, 2]
    assert df_int["col3"].equals(df["col3"])

    assert maps_str2int == {
        "col1": {"a": 0, "b": 1, "c": 2},
        "col2": {"x": 0, "y": 1, "z": 2},
    }
