import torch
from torch import nn, Tensor

from spalbp.lib.attention.config import MultiHeadAttentionConfig, SlidingWindowConfig


# TAKEN FROM https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class _Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.0, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(~attention_mask.bool(), float('-inf'))  # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    @classmethod
    def from_config(cls, config: MultiHeadAttentionConfig):
        return cls(config.heads, config.head_size, config.emb, config.context, config.dropout)

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.heads = nn.ModuleList([_Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor):
        out = torch.cat([h(x, attention_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class NativeAttention(nn.Module):

    @classmethod
    def from_config(cls, config: MultiHeadAttentionConfig):
        return cls(config.heads, config.emb)

    def __init__(self, num_heads, emb):
        super().__init__()
        self.num_heads = num_heads
        self.native_attention = nn.MultiheadAttention(emb, num_heads, batch_first=True)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        b, c, e = attention_mask.shape
        expanded = ~(attention_mask[:, None, :, :]
                     .expand(b, self.num_heads, c, e)
                     .reshape(b * self.num_heads, c, e)
                     .bool())

        out, weights = self.native_attention(x, x, x, attn_mask=expanded, need_weights=True)
        return out


class EasySlidingWindowAttention(NativeAttention):

    @classmethod
    def from_config(cls, config: SlidingWindowConfig):
        return cls(config.heads, config.emb, config.window_size)

    def __init__(self, num_heads, emb, window_size):
        super().__init__(num_heads, emb)
        self.window_size = window_size

    @staticmethod
    def _get_sliding_window_mask(c: int, window_size: int):
        sliding_window_attn = torch.zeros((c, c))
        r = torch.arange(c)
        for step in range(1, 1+window_size):
            sliding_window_attn[r, torch.minimum(r + step, torch.tensor(c-1))] = 1
            sliding_window_attn[r, torch.maximum(r - step, torch.tensor(0))] = 1
        return sliding_window_attn

    def forward(self, x: Tensor, attention_mask: Tensor):
        c = x.shape[-2]
        sliding_window_attn = self._get_sliding_window_mask(c, self.window_size)
        sliding_window_attn = sliding_window_attn.to(x.device)
        sliding_window_attn = sliding_window_attn.expand_as(attention_mask)
        attention_mask = torch.logical_and(attention_mask, sliding_window_attn).long()
        return super().forward(x, attention_mask)


class SlidingWindowWithGlobalAttention(EasySlidingWindowAttention):

    def forward(self, x: Tensor, attention_mask: Tensor):
        c = x.shape[-2]
        sliding_window_attn = self._get_sliding_window_mask(c, self.window_size)
        sliding_window_attn = sliding_window_attn.to(x.device)
        sliding_window_attn[:, 0] = 1
        sliding_window_attn[0, :] = 1
        sliding_window_attn = sliding_window_attn.expand_as(attention_mask)
        attention_mask = torch.logical_and(attention_mask, sliding_window_attn).long()
        # Invoke grandparent's forward method
        return super(EasySlidingWindowAttention, self).forward(x, attention_mask)
