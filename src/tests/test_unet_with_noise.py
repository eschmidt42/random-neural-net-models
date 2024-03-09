# -*- coding: utf-8 -*-
import torch
import random_neural_net_models.unet_with_noise as rnnm_unet


def test_attention2d_forward():
    num_features_out = 64
    n_channels_per_head = 16
    batch_size = 2
    height = 32
    width = 32

    attention = rnnm_unet.Attention2D(num_features_out, n_channels_per_head)

    # Create random input tensor
    X = torch.randn(batch_size, num_features_out, height, width)

    # Forward pass
    output = attention(X)

    # Check output shape
    assert output.shape == (batch_size, num_features_out, height, width)

    # Check output values
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, torch.zeros_like(output), atol=1e-5)
