import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from attention_layers import (
    SparseSelfAttention, BlocksparseFixedSelfAttention, MultiHeadAttention, ReallySparseAttention
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class TransformerBlock(nn.Module):

    def __init__(self,
                 context: int,
                 emb: int,
                 heads: int = 4,
                 ff_hidden_mult: int = 4,
                 dropout: float = 0.0,
                 attention_type: Literal['dense', 'sparse', 'fixed'] = 'dense',
                 **kwargs):
        super().__init__()
        if attention_type == 'dense':
            self.attend = MultiHeadAttention(heads, emb, emb, context, **kwargs)
        elif attention_type == 'sparse':
            self.attend = SparseSelfAttention(emb, context, n_heads=heads, **kwargs)
        elif attention_type == 'fixed':
            self.attend = BlocksparseFixedSelfAttention(emb, t=context, **kwargs)
        elif attention_type == 'sparse2d':
            self.attend = ReallySparseAttention(emb, context, n_heads=heads, **kwargs)
        else:
            raise ValueError(f'attention_type {attention_type} not recognized')

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.register_buffer('attn_mask', torch.tril(torch.ones(context, context)).bool())
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x: Tensor) -> Tensor:
        normed1 = self.norm1(x)
        attended = self.attend(normed1)
        x = x + attended
        x = self.dropout(x)
        normed2 = self.norm2(x)
        x = x + self.ff(normed2)
        return self.dropout(x)


class SparseTransformer(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 context_len: int,
                 emb: int,
                 vocab_size: int,
                 *args,
                 **kwargs):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        # Here we add 1 to emb because an additional coord is used for positional embeddings but only added
        # here in the root Transformer
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb)
        self.pos_embedding = nn.Embedding(num_embeddings=context_len, embedding_dim=emb)
        t_blocks = [TransformerBlock(context_len, emb, *args, **kwargs) for _ in range(n_blocks)]
        self.t_blocks = nn.Sequential(*t_blocks)

    def embed(self, x: Tensor) -> Tensor:
        # Here we'll do some embedding addition
        x = self.token_embedding(x)
        b, c, e = x.size()
        positions = self.pos_embedding(torch.arange(self.context_len, dtype=torch.int, device=util.d(x)))[None, :, :] \
            .expand(b, -1, -1)
        return positions + x

    def forward(self, x: Tensor) -> Tensor:
        assert torch.isnan(self.token_embedding.weight).sum() == 0, "We got some nan embeddings"
        embedded = self.embed(x)  # (batch, context_len, emb)
        t_blocked = self.t_blocks(embedded)  # (batch, context_len, emb)
        done = self.post_tblocks(t_blocked)
        return done

    def post_tblocks(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ClassificationTransformer(SparseTransformer):

    def __init__(self,
                 n_blocks: int,
                 context_len: int,
                 emb: int,
                 vocab_size: int,
                 num_classes: int,
                 *args,
                 **kwargs):
        super().__init__(n_blocks, context_len, emb, vocab_size, *args, **kwargs)
        self.to_prob = nn.Linear(emb, num_classes)

    def post_tblocks(self, x: Tensor) -> Tensor:
        x = x.max(dim=1)[0]  # (batch, emb)
        x = self.to_prob(x)  # (batch, num_classes)
        x = torch.nn.functional.log_softmax(x, dim=1)  # (batch, num_classes) the probability distribution over classes
        return x


class GeneratingTransformer(SparseTransformer):
    def __init__(self,
                 n_blocks: int,
                 context_len: int,
                 emb: int,
                 vocab_size: int,
                 *args,
                 **kwargs):
        super().__init__(n_blocks, context_len, emb, vocab_size, *args, **kwargs)
        self.to_probs = nn.Linear(emb, vocab_size)

    def post_tblocks(self, x: Tensor) -> Tensor:
        b, c, e = x.size()  # batch, context, embed
        x = self.to_probs(x.view(b * c, e)).view(b, c, self.vocab_size)
        return torch.nn.functional.log_softmax(x, dim=2)

