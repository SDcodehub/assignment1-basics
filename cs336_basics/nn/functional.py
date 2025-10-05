"""
stateless kernels: linear, softmax, silu, rmsnorm, sdpa, rope, embedding_lookup
"""


# cs336_basics/nn/functional.py
def linear(x, weight, bias=None):
    # x: [..., d_in], weight: [d_out, d_in], bias: [d_out] or None
    y = x.matmul(weight.T)
    return y if bias is None else y + bias