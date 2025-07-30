import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(Linear, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: nn.Parameter = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._init_weight()
        
    def _init_weight(self):
        std = math.sqrt(2/(self.in_features+self.out_features))
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "o i, ... i -> ... o")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(Embedding, self).__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.weight: nn.Parameter = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._init_weight()
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(RMSNorm, self).__init__()
        self.d_model: int = d_model
        self.eps = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps)
        result = einsum(x, self.weight, "... d, d -> ... d") / rms
        
        return result.to(in_type)
    
class GLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(GLU, self).__init__()
        self.activation = activation
        self.d_model: int = d_model
        self.d_ff: int = int(8/3 * d_model) if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
    
    
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(GLU):
    def __init__(self, d_model: int, d_ff: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super(SwiGLU, self).__init__(d_model, d_ff, SiLU(), device, dtype)

class SiLU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SiLU, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)

class RoPE(nn.Module):
    """
    \\theta_{i,k} = \\frac{i}{\\Theta^{2k/d}}
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(RoPE, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        
        seq = torch.arange(0, max_seq_len, device=device, dtype=dtype).unsqueeze(1)
        dim = torch.arange(0, d_k // 2, device=device, dtype=dtype).unsqueeze(0)
        inv_freq = 1 / (theta ** (2 * (dim / d_k)))
        angle = seq * inv_freq
        
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        )
        return x_rotated.flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True)[0]
    exp = torch.exp(x - max_value)
    return exp / exp.sum(dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(K.shape[-1])
    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))
    attention = softmax(scores, dim=-1)
    return einsum(attention, V, "... q k, ... k d -> ... q d")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        Q, K, V = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.num_heads), rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.num_heads), rearrange(V, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        
        mask = torch.tril(torch.ones(Q.size(-2), Q.size(-2), dtype=torch.bool, device=x.device))
        
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h s d_k -> ... s (h d_k)")

        return self.output_proj(attention)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = rope

    @classmethod
    def from_theta(
        cls,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MultiHeadSelfAttentionWithRoPE":
        d_k = d_model // num_heads
        rope = RoPE(theta, d_k, max_seq_len, device=device, dtype=dtype)
        return cls(d_model, num_heads, rope, device=device, dtype=dtype)

        
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        Q, K, V = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.num_heads), rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.num_heads), rearrange(V, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        
        seq = Q.size(-2)
        
        if token_positions is None:
            token_positions = torch.arange(0, seq, device=x.device)
        
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h s d_k -> ... s (h d_k)")

        return self.output_proj(attention)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    @classmethod
    def from_theta(
        cls,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "TransformerBlock":
        d_k = d_model // num_heads
        rope = RoPE(theta, d_k, max_seq_len, device=device, dtype=dtype)
        return cls(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

if __name__ == '__main__':
    pass