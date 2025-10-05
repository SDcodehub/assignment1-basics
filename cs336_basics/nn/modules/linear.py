# cs336_basics/nn/modules/linear.py
import torch
from cs336_basics.nn import functional as F

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_out, d_in))
        self.bias = torch.nn.Parameter(torch.empty(d_out)) if bias else None
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)