# -*- coding: utf-8 -*-
import pytest

import random_neural_net_models.data as rnnm_data


@pytest.mark.parametrize(
    "n_categories_per_column, expected_ranges",
    [
        ([3], [(0, 1, 2)]),  # Test case 1: Single column with 3 categories
        (
            [2, 4, 3],
            [(0, 1), (2, 3, 4, 5), (6, 7, 8)],
        ),  # Test case 2: Multiple columns with different number of categories
    ],
)
def test_get_index_ranges_from_n_cats_per_col(
    n_categories_per_column, expected_ranges
):
    index_ranges = rnnm_data.get_index_ranges_from_n_cats_per_col(
        n_categories_per_column
    )
    assert index_ranges == expected_ranges


@pytest.mark.parametrize(
    "n_categories_per_column, expected_ranges",
    [
        ([], []),  # Test case 3: Empty input
        ([0], []),  # Test case 4: Single column with 0 categories
        (
            [-3],
            [],
        ),  # Test case 5: Single column with negative number of categories
    ],
)
def test_get_index_ranges_from_n_cats_per_col_fails(
    n_categories_per_column, expected_ranges
):
    with pytest.raises(ValueError):
        rnnm_data.get_index_ranges_from_n_cats_per_col(
            n_categories_per_column
        ) == expected_ranges
