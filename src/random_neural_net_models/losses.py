# -*- coding: utf-8 -*-
import torch
import torch.nn.modules.loss as torch_loss

from random_neural_net_models.data import MNISTBlockWithLabels, XyBlock, XyBlock_numcat


class MSELossMNIST1HotLabel(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, input: torch.Tensor, target: MNISTBlockWithLabels | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(target, MNISTBlockWithLabels):
            return super().forward(input, target.label)
        return super().forward(input, target)


class MSELossMNISTAutoencoder(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, input: torch.Tensor, target: MNISTBlockWithLabels | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(target, MNISTBlockWithLabels):
            return super().forward(input, target.image)
        return super().forward(input, target)


class CrossEntropyMNIST(torch_loss.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, input: torch.Tensor, target: MNISTBlockWithLabels | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(target, MNISTBlockWithLabels):
            return super().forward(input, target.label)
        return super().forward(input, target)


class CrossEntropyXy(torch_loss.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: XyBlock | XyBlock_numcat | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(target, (XyBlock, XyBlock_numcat)):
            return super().forward(input, target.y.ravel())
        return super().forward(input, target.ravel())


class MSELossXy(torch_loss.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: XyBlock | XyBlock_numcat | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(target, (XyBlock, XyBlock_numcat)):
            return super().forward(input, target.y)
        return super().forward(input, target)
