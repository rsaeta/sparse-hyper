import torch
from torch import nn
import fast_transformers.attention as attn
from fast_transformers.attention import ClusteredAttention as _ClusteredAttention
from .config import ClusteredAttentionConfig


class ClusteredAttention(nn.Module):

    @classmethod
    def from_config(cls, config: ClusteredAttentionConfig):
        return cls(config.heads,
                   config.emb,
                   config.head_size,
                   config.num_clusters,
                   config.num_iterations,
                   config.hash_bits,
                   config.softmax_temp,
                   config.attention_dropout)

    def __init__(self,
                 num_heads: int,
                 embedding_dim: int,
                 head_size: int,
                 clusters: int,
                 iterations: int,
                 hash_bits: int,
                 softmax_temp: float,
                 dropout: float):
        super().__init__()
        _attention = _ClusteredAttention(clusters,
                                         iterations=iterations,
                                         bits=hash_bits,
                                         softmax_temp=softmax_temp,
                                         attention_dropout=dropout)
        self.head_size = head_size
        self.layer = attn.attention_layer.AttentionLayer(_attention, embedding_dim, num_heads, head_size, head_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask.all_ones = lambda: True
        return self.layer(x, x, x, attention_mask, self.head_size, self.head_size)
