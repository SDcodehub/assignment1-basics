import torch
import torch.nn as nn

from cs336_basics.nn.modules import MultiHeadAttention, SwiGLUFFN, RMSNorm

class Transformer(nn.Module):
    """
    implements a single pre-norm transformer block as a stateful module
    this block consists of a multi-head self-attention layer and a feed-forward network
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # first sub-layer components
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        # second sub-layer components
        self.ffn = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        defines the forward pass for the pre-norm Transformer block
        """
        # first sub-layer: multi-head self-attention
        # residual connection starts from the original input 'x'
        # the equation is: x + attention(Norm(x))

        attention_output = self.attn(self.norm1(x), token_positions)
        x = x + attention_output

        # second sub-layer feed forward network
        # residual connection starts from the output of the first sub-layer
        # the equation is x + ffn(norm(x))
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x