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
    

def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a lookup in an embedding matrix
    this is stateless function

    Args:
        input_ids(torch.Tensor): a tensor of integer token IDs of shape (...)
        weight (torch.Tensor): the embedding matrix of shape (vocab_size, embedding_dim)

    Returns:
        torch.Tensor: a looked up embedding vectors of shap (..., embedding_dim)
    """
    # Pytorchs indexing is higly optimised for this operation
    return weight[input_ids]


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    applies rms layer normalisation
    stateless function
    
    Args:
        input (torch.Tensor): input tensor of shape (..., d_model)
        weight (torch.Tensor): learnable gain parameter (gamma) of shape (d_model, )
        eps (float): a small value added for numerical stability

    Returns:
        torch.Tensor: normalised tensor of the same shape as input
    """
    # store original dtype to cast back at the end
    input_dtype = input.dtype

    # upcast to float32 for stable computation of squares
    x = input.to(torch.float32)

    # calculate the mean of the squares of the input along the last dimension
    variance = x.pow(2).mean(dim=-1, keepdim=True)

    # calculate the reciprocal fo the square root
    rsqrt = torch.rsqrt(variance + eps)

    # normalize the input and apply the learnable gain (weight)
    normalized_x = x * rsqrt

    # apply the gain and cast back to the original dtype
    return (weight * normalized_x ).to(input_dtype)