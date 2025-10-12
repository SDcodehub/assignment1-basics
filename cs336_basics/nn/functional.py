"""
stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
"""

import torch
import math
from einops import einsum

def linear(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies linear transformation to incoming data.
    Stateless function.

    Args:
        input(torch.Tensor): input tensor of shape (...., in_features)
        weight(torch.Tensor): weight matrix of shape (out_features, in_features)

    Returns:
    torch.Tensor: output tensor of shape (..., out_features)
    """
    # explicit einsum notation; supports arbitrary leading/batch dims
    return einsum(input_tensor, weight, "... in_features, out_features in_features -> ... out_features")
    

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


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
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
    input_dtype = input_tensor.dtype

    # upcast to float32 for stable computation of squares
    x = input_tensor.to(torch.float32)

    # calculate the mean of the squares of the input along the last dimension
    variance = x.pow(2).mean(dim=-1, keepdim=True)

    # calculate the reciprocal fo the square root
    rsqrt = torch.rsqrt(variance + eps)

    # normalize the input and apply the learnable gain (weight)
    normalized_x = x * rsqrt

    # apply the gain and cast back to the original dtype
    return (weight * normalized_x ).to(input_dtype)


def silu(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    applies the sigmoid weighted linear unit activation function
    also known as swish, x * sigmoid(x)

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor, of same as input
    """
    # addignment allows using torch.sigmoid for numerical stability
    return input_tensor * torch.sigmoid(input_tensor)

def swiglu_ffn(
    input_tensor: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """
    implementation of swiglu ffn 
    stateless function

    Args:
        input (torch.Tensor): input tensor of shape (...,d_model)
        w1 (torch.Tensor): weight matric for the first projection, shape (d_ff, d_model)
        w2 (torch.Tensor): weight matric for the output projection, shape (d_model, d_ff)
        w3 (torch.Tensor): weight matric for the gate projection, shape (d_ff, d_model)

    Returns:
        torch.Tensor: output tensor of shape (..., d_model)
    """
    # project up using w1, w3
    # einsum: "... d_model, d_ff d_model -> ... d_ff"
    x1 = linear(input_tensor, w1)
    # einsum: "... d_model, d_ff d_model -> ... d_ff"
    x3 = linear(input_tensor, w3)

    # apply silu activation and the gating mechanis (elementmise multiplication)
    gated_x = silu(x1) * x3

    # project back down using w2
    # einsum: "... d_ff, d_model d_ff -> ... d_model"
    return linear(gated_x, w2)


def softmax(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    applies numerically stable softmax function
    this is a stateless function

    Args: 
        input (torch.Tensor): input tensor of logits of shape
        dim (int): dimension along which softmax will be applied

    Returns:
        torch.Tensor: Tensor of probabilities, same shape as input.
    """
    # subtract the max for numerical stability
    # we use keepdim=true to ensure the result is broadccastable
    max_vals, _ = torch.max(input_tensor, dim=dim, keepdim=True)
    shifted_logits = input_tensor - max_vals

    # exponentiate
    exps = torch.exp(shifted_logits)

    # sum the exponents and divide
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    
    return exps / sum_exps


def scaled_dot_product_attention(
    query: torch.Tensor,
    key:torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    computes scaled dot product attenstion as a stateless function.
    formula: softmax( (query @ key.T) / sqrt(d_key) ) @ value

    Args:
        query (torch.Tensor): query tensor fo shape (..., seq_len, d_k)
        key (torch.Tensor): key tensor of shape (..., seq_len, d_k)
        value (torch.Tensor): value tensor of shape (..., seq_len, d_v)
        mask (torch.Tensor, optional): boolean mask of shape (..., seq_len_q, seq_len_k)
                                        if a value is flase, the corresponding attention 
                                        score is set to -inf. defaults to none
        Returns:
            torch.Tensor: the output of the attention mechanism, shape (..., seq_len, d_v)
    """
    # d_k is the dimension fo the key/query vectors
    d_k = key.shape[-1]

    # compute raw scores with matrix multiplication (Q @ K.T)
    scores = torch.einsum("...qd, ...kd-> ...qk", query, key)

    # scale the scores
    scaled_scores = scores / math.sqrt(d_k)

    # apply mask if provided
    if mask is not None:
        # we need to ensure the mask can be broadcasted to the scores shape
        # this is usually handled by how the mask is constructed, but a view can make is robust
        # for a mask of shape (T,T ) we might need to add batch/head dimensions
        while mask.dim() < scaled_scores.dim():
            mask = mask.unsqueeze(0)

        # set scores for a very large negative number where the mask is false
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.finfo(scaled_scores.dtype).max)

    # compute the attention weights using softmax
    # the softmax is applied on the last dimension (the keys)
    attention_weights = softmax(scaled_scores, dim=-1)

    # compute the weighted sum of values
    return torch.einsum("...qk, ...kd-> ...qd", attention_weights, value)
    
    

