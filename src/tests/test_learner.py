# -*- coding: utf-8 -*-
SEED = 42

import typing as T
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import sklearn.datasets as sk_datasets
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torch.nn.modules.loss as torch_loss
import torch.optim as optim
from einops import rearrange
from tensordict import tensorclass
from torch.utils.data import DataLoader, Dataset

import random_neural_net_models.data as rnnm_data
import random_neural_net_models.learner as rnnm_learner
import random_neural_net_models.utils as utils


class Layer(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        self.act = nn.Sigmoid()
        self.net = nn.Sequential(self.lin, self.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        n_hidden: T.Tuple[int] = (10, 5, 1),
    ):
        super().__init__()
        self.n_hidden = n_hidden

        components = [
            Layer(n_in, n_out)
            for (n_in, n_out) in zip(n_hidden[:-1], n_hidden[1:])
        ]

        self.net = nn.Sequential(*components)

    def forward(
        self, input: T.Union[rnnm_data.XyBlock, rnnm_data.XBlock]
    ) -> torch.Tensor:
        return self.net(input.x)


class BCELoss(torch_loss.BCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: rnnm_data.XyBlock
    ) -> torch.Tensor:
        return super().forward(inference, input.y)


@pytest.mark.parametrize("use_callbacks", [True, False])
def test_learner(use_callbacks: bool):
    "The test succeeds if the below executes without error"

    X, y = sk_datasets.make_blobs(
        n_samples=1_000,
        n_features=2,
        centers=2,
        random_state=SEED,
    )

    X0, X1, y0, y1 = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    device = utils.get_device()

    ds_train = rnnm_data.NumpyTrainingDataset(X0, y0)
    ds_val = rnnm_data.NumpyTrainingDataset(X1, y1)

    dl_train = DataLoader(
        ds_train,
        batch_size=10,
        collate_fn=rnnm_data.collate_numpy_dataset_to_xyblock,
        shuffle=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=10,
        collate_fn=rnnm_data.collate_numpy_dataset_to_xyblock,
        shuffle=False,
    )

    model = DenseNet(n_hidden=(2, 10, 5, 1))

    n_epochs = 2
    learning_rate = 1.0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=1e-3)
    loss = BCELoss()
    loss_callback = rnnm_learner.TrainLossCallback()

    save_dir = Path(
        f"./test-models-cb-{use_callbacks}"
    )  # location used by learner.find_learning_rate to store the model before the search

    # the following callbacks are not strictly necessary for learning rate search and
    # training, but may make debugging of slow / unexpected training easier

    if use_callbacks:
        # the name_patterns used below work only because of how DenseNet and Layer are defined, you may have to use different patterns
        activations_callback = rnnm_learner.TrainActivationsCallback(
            every_n=10, max_depth_search=4, name_patterns=(".*act",)
        )
        gradients_callback = rnnm_learner.TrainGradientsCallback(
            every_n=10, max_depth_search=4, name_patterns=(".*lin",)
        )
        parameters_callback = rnnm_learner.TrainParametersCallback(
            every_n=10, max_depth_search=4, name_patterns=(".*lin",)
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=learning_rate,
            epochs=n_epochs,
            steps_per_epoch=len(dl_train),
        )
        scheduler_callback = rnnm_learner.EveryBatchSchedulerCallback(scheduler)

        callbacks = [
            loss_callback,
            activations_callback,
            gradients_callback,
            parameters_callback,
            scheduler_callback,
        ]
    else:
        callbacks = [loss_callback]

    learner = rnnm_learner.Learner(
        model,
        optimizer,
        loss,
        callbacks=callbacks,
        save_dir=save_dir,
        device=device,
    )

    do_create_save_dir = not learner.save_dir.exists()
    if do_create_save_dir:
        print(f"{learner.save_dir=}")
        learner.save_dir.mkdir()

    lr_find_callback = rnnm_learner.LRFinderCallback(1e-5, 100, 100)

    learner.find_learning_rate(
        dl_train, n_epochs=2, lr_find_callback=lr_find_callback
    )

    lr_find_callback.plot()

    learner.fit(dl_train, n_epochs=n_epochs, dataloader_valid=dl_val)

    loss_callback.plot()

    if use_callbacks:
        parameters_callback.plot()

    if use_callbacks:
        gradients_callback.plot()

    if use_callbacks:
        activations_callback.plot()

    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X0, X1 = np.meshgrid(x0, x1)
    X_plot = np.array([X0.ravel(), X1.ravel()]).T
    X_plot[:4]

    ds_plot = rnnm_data.NumpyInferenceDataset(X_plot)
    dl_plot = DataLoader(
        ds_plot,
        batch_size=5,
        collate_fn=rnnm_data.collate_numpy_dataset_to_xblock,
    )

    y_prob = learner.predict(dl_plot)

    y_prob = y_prob.detach().numpy()
    y_prob

    fig, ax = plt.subplots()
    im = ax.pcolormesh(X0, X1, y_prob.reshape(X0.shape), alpha=0.2)
    fig.colorbar(im, ax=ax)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax, alpha=0.3)
    plt.show()

    if do_create_save_dir:
        learner.save_dir.rmdir()
