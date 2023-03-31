import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from attention_layers import (
    SparseSelfAttention, 
    BlocksparseFixedSelfAttention, 
    MultiHeadAttention, 
    ReallySparseAttention,
    NativeAttention,
    DynamicDilatedAttention,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Tuple, List


class TransformerBlock(nn.Module):

    def __init__(self,
                 context: int,
                 emb: int,
                 heads: int = 4,
                 ff_hidden_mult: int = 4,
                 dropout: float = 0.0,
                 attention_type: Literal['dense', 'sparse', 'fixed', 'sparse2d', 'native'] = 'dense',
                 depth: int = 0,
                 shared_predictor: nn.Module = None,
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
        elif attention_type == 'native':
            self.attend = NativeAttention(heads, emb, context, **kwargs)
        elif attention_type == 'dilated':
            self.attend = DynamicDilatedAttention(shared_predictor, 
                                                  emb, 
                                                  layer=depth, 
                                                  n_heads=heads,
                                                  **kwargs)
        else:
            raise ValueError(f'attention_type {attention_type} not recognized')

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x + self.attend(self.norm1(x)))
        x = self.dropout(x + self.ff(self.norm2(x)))
        return x

    def forward_for_plot(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        normed1 = self.norm1(x)
        if isinstance(self.attend, MultiHeadAttention):
            m = torch.ones(x.size()).nonzero()
            s = torch.ones((m.size(0), ))
            v = s.clone()
        elif isinstance(self.attend, SparseSelfAttention):
            m, s, v = self.attend.hyper(normed1)
            # Single coord generated by hyper since 1-d case. Add x-coord for plotting
            x_coords = torch.arange(m.size(1), device=util.d(x))[None, :, None, None].expand_as(m)
            m = torch.cat([x_coords, m], axis=-1)
        elif isinstance(self.attend, ReallySparseAttention):
            m, s, v = self.attend.hyper(normed1)
        elif isinstance(self.attend, BlocksparseFixedSelfAttention):
            head1_att = self.attend.head1.indices
            head2_att = self.attend.head2.indices
            m = torch.cat([head1_att, head2_att])
            s = torch.ones((m.size(0), ))
            v = s.clone()
        attended = self.attend(normed1)
        x = x + attended
        x = self.dropout(x)
        normed2 = self.norm2(x)
        x = x + self.ff(normed2)
        return self.dropout(x), (m, s, v)


class SparseTransformer(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 context_len: int,
                 emb: int,
                 vocab_size: int,
                 attention_type: str = None,
                 *args,
                 **kwargs):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb)
        self.pos_embedding = nn.Embedding(num_embeddings=context_len, embedding_dim=emb)
        if attention_type == 'dilated':
            self.shared_predictor = nn.Sequential(nn.Linear(1, 10), nn.Sigmoid(), nn.Linear(4, 2), nn.Softplus())
        else:
            self.shared_predictor = None
        t_blocks = [TransformerBlock(context_len,
                                      emb, 
                                      *args, 
                                      depth=depth, 
                                      shared_predictor=self.shared_predictor, 
                                      attention_type=attention_type,
                                      **kwargs) 
                        for depth in range(n_blocks)]
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

    def forward_for_plot(self, x: Tensor) -> Tuple[Tensor, Tuple[List, List, List]]:
        ms, ss, vs = [], [], []
        x = self.embed(x)
        for t_block in self.t_blocks:
            x, (m, s, v) = t_block.forward_for_plot(x)
            ms.append(m)
            ss.append(s)
            vs.append(v)
        done = self.post_tblocks(x)
        return done, (ms, ss, vs)

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
        return x

