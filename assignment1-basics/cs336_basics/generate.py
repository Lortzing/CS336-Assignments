import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum

from cs336_basics.model import softmax

def temperature_logit(x: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.log_softmax(x / temperature, dim=-1)