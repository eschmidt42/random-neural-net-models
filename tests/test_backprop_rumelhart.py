from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from random_neural_net_models.backprop_rumelhart import (
    Rumelhart1986PerceptronClassifier,
)


def test_classifier():
    SEED = 42

    X, y = make_blobs(
        n_samples=1_000,
        n_features=2,
        centers=2,
        random_state=SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    model = Rumelhart1986PerceptronClassifier(
        n_hidden=(10, 5), epochs=10, verbose=True, eps=1e-3, alpha=1e-3
    )

    model.fit(X_train, y_train)
