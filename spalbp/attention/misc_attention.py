import torch
from torch import nn

import entmax

from spalbp.attention.config import AlphaEntmaxAttentionConfig


class AlphaEntmax(nn.Module):

    @classmethod
    def from_config(cls, config: AlphaEntmaxAttentionConfig):
        return cls(
            num_heads=config.heads,
            emb=config.emb,
            context=config.context,
            alpha=config.alpha,
        )

    def __init__(self, num_heads, emb, context, alpha=1.5):
        super().__init__()
        self.num_heads = num_heads
        self.emb = emb
        self.context = context
        self.alpha = torch.tensor(alpha, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

        self.to_keys = nn.Linear(emb, emb*num_heads)
        self.to_queries = nn.Linear(emb, emb * num_heads)
        self.to_values = nn.Linear(emb, emb * num_heads)

        self.unify_heads = nn.Linear(emb*num_heads, emb)

    def forward(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.to_keys(x).view(b, c, self.num_heads, e)
        q = self.to_queries(x).view(b, c, self.num_heads, e)
        v = self.to_values(x).view(b, c, self.num_heads, e)
        rank = k.size(-1)
        dot = (q.transpose(-2, -1) @ k) / (rank ** 0.5)  # scaled dot-product attention
        if self.mask:
            triu = torch.triu(torch.ones(dot.size(), device=x.device))
            dot = torch.masked_fill(dot, triu, -float('inf'))
        dot = entmax.entmax_bisect(dot, self.alpha)
        res = (dot @ v.transpose(-2, -1)).transpose(-2, -1).reshape(b, c, self.num_heads*e)
        return self.unify_heads(res)
