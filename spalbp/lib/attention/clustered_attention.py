import torch
from torch import nn
# from routing_transformer.routing_transformer import SelfAttention
from routing_transformer import KmeansAttention
from .config import ClusteredAttentionConfig


class ClusteredAttention(nn.Module):

    @classmethod
    def from_config(cls, config: ClusteredAttentionConfig):
        return cls(config.heads,
                   config.emb,
                   config.context,
                   config.num_clusters,
                   config.head_size,
                   config.attention_dropout,
                   config.window_size)

    def __init__(self,
                 num_heads: int,
                 embedding_dim: int,
                 context: int,
                 clusters: int,
                 head_size: int,
                 dropout: float,
                 window_size: int):
        super().__init__()
        self.attention = KmeansAttention(clusters,
                                         window_size,  # window_size
                                         4,  # global attention
                                         head_size,
                                         causal=False,
                                         dropout=dropout)
        self.num_heads = num_heads
        self.head_size = head_size
        self.to_keys = nn.Linear(embedding_dim, head_size * num_heads)
        self.to_values = nn.Linear(embedding_dim, head_size * num_heads)
        self.to_queries = nn.Linear(embedding_dim, head_size * num_heads)
        self.unify = nn.Linear(head_size * num_heads, embedding_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        b, c, e = x.shape
        k = self.to_keys(x).view(b, c, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.to_values(x).view(b, c, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.to_queries(x).view(b, c, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        out, loss = self.attention(q, k, v)  # kind of annoying have to deal with aux-loss here
        out = self.unify(out.transpose(1, 2).reshape(b, c, self.num_heads * self.head_size))
        return out, loss
