# -*- coding: utf-8 -*-
import pytest
import torch

from random_neural_net_models import tabular_vae


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]),
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        ),
    ],
)
def test_softmax_forward(x: torch.Tensor):
    n_categories_per_column = [3, 4, 2]
    model = tabular_vae.SoftmaxForCategoricalColumns(n_categories_per_column)

    output = model(x)
    for r in model.index_ranges:
        torch.allclose(output[:, r].sum(dim=1), torch.ones(x.shape[0]))


def test_transform_X_cat_probs_to_classes():
    X_cat_probs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    n_categories_per_column = [
        2,
    ]
    expected_output = torch.tensor([1, 0, 1]).reshape((-1, 1))

    output = tabular_vae.transform_X_cat_probs_to_classes(
        X_cat_probs, n_categories_per_column
    )

    assert torch.allclose(output, expected_output)
