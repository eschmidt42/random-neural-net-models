# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import random_neural_net_models.perceptron as perceptron


class TestPerceptronClassifier:
    model = perceptron.PerceptronClassifier()

    @pytest.mark.parametrize(
        "attribute",
        [
            "weights_",
            "bias_",
            "errors_",
            "classes_",
            "n_classes_",
            "is_multi_class_",
        ],
    )
    def test_attributes_(self, attribute):
        assert not hasattr(self.model, attribute)

    X = np.array(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]
    )
    y = np.array([False, False, True, True])

    def test_fit(self):
        model = perceptron.PerceptronClassifier()
        model.fit(self.X, self.y)

    def test_predict(self):
        model = perceptron.PerceptronClassifier()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert (predictions == self.y).all()


@pytest.mark.slow
@parametrize_with_checks(
    [
        perceptron.PerceptronClassifier(),
    ]
)
def test_perceptron_with_sklearn_checks(estimator, check):
    """Test of estimators using scikit-learn test suite

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks
    """
    check(estimator)
