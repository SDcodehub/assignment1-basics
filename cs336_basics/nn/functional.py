"""
stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
"""

import torch

def linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies linear transformation to incoming data.
    Stateless function.

    Args:
        input(torch.Tensor): input tensor of shape (...., in_features)
        weight(torch.Tensor): weight matrix of shape (out_features, in_features)

    Returns:
    torch.Tensor: output tensor of shape (..., out_features)
    """
    # using einsum as its explicit and handles broadcasting over batch dimensions
    # This is the core computation of the linear layer
    return torch.einsum("...i,oi->...o", input, weight)
    