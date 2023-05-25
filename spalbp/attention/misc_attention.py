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
            head_size=config.head_size
        )

    def __init__(self, num_heads, emb, context, head_size, alpha=1.5):
        super().__init__()
        self.num_heads = num_heads
        self.emb = emb
        self.context = context
        self.head_size = head_size
        self.alpha = torch.tensor(alpha, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

        self.to_keys = nn.Linear(emb, head_size * num_heads)
        self.to_queries = nn.Linear(emb, head_size * num_heads)
        self.to_values = nn.Linear(emb, head_size * num_heads)

        self.unify_heads = nn.Linear(head_size * num_heads, emb)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        b, c, e = x.size()
        k = self.to_keys(x).view(b, c, self.num_heads, self.head_size)  # (B, C, nh, hs)
        q = self.to_queries(x).view(b, c, self.num_heads, self.head_size)  # (B, C, nh, hs)
        v = self.to_values(x).view(b, c, self.num_heads, self.head_size)  # (B, C, nh, hs)

        k = k.transpose(2, 1).reshape(-1, c, self.head_size)  # (B*nh, C, hs)
        q = q.transpose(2, 1).reshape(-1, c, self.head_size)  # (B*nh, C, hs)

        rank = k.size(-1)
        dot = (q @ k.transpose(-1, -2)) / (rank ** 0.5)  # scaled dot-product attention (B*nh, C, C)
        dot = dot.reshape(b, self.num_heads, c, c)

        dot = dot.masked_fill(~attention_mask[:, None].bool(), -float('inf'))  # mask out invalid positions
        dot = entmax.entmax_bisect(dot, self.alpha)
        mult = (dot @ v.transpose(1, 2))  # (B, nh, C, hs)
        mult = mult.transpose(1, 2).reshape(b, c, self.num_heads * self.head_size)  # (B, C, nh*hs)
        return self.unify_heads(mult)
