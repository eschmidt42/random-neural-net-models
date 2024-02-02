# -*- coding: utf-8 -*-
import torch
import torch.nn.modules.loss as torch_loss

import random_neural_net_models.data as rnnm_data


class MSELossMNIST1HotLabel(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: rnnm_data.MNISTDataTrain
    ) -> torch.Tensor:
        return super().forward(inference, input.label)


class MSELossMNISTAutoencoder(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: rnnm_data.MNISTDataTrain
    ) -> torch.Tensor:
        return super().forward(inference, input.image)


class CrossEntropyMNIST(torch_loss.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: rnnm_data.MNISTDataTrain
    ) -> torch.Tensor:
        return super().forward(inference, input.label)
