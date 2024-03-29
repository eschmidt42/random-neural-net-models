# -*- coding: utf-8 -*-
import torch
import random_neural_net_models.unet_with_noise as rnnm_unet


def test_attention2d_forward():
    n_cnn_channels = 64
    n_channels_per_head = 16
    batch_size = 2
    height = 32
    width = 32

    attention = rnnm_unet.Attention2D(n_cnn_channels, n_channels_per_head)

    # Create random input tensor
    X = torch.randn(batch_size, n_cnn_channels, height, width)

    # Forward pass
    output = attention(X)

    # Check output shape
    assert output.shape == (batch_size, n_cnn_channels, height, width)

    # Check output values
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output), atol=1e-5)
