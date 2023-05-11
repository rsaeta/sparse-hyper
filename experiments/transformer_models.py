import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util
from bigbird import BigBirdBlockSparseAttention, BigBirdConfig
from smallbird import SmallBirdSparseAttention, SmallBirdConfig
from smallest_bird import SmallerBirdConfig, SmallerBirdSparseAttention
# from longformer_sliding_window import LongformerSelfAttention
# from sliding_window import SlidingWindowAttention, SlidingWindowConfig
from bigbird_mod import BigBirdModSelfAttention
from attention_layers import (
    SparseSelfAttention,
    BlocksparseFixedSelfAttention,
    MultiHeadAttention,
    ReallySparseAttention,
    NativeAttention,
    DynamicDilatedAttention,
    AlphaEntmax,
    NonadaptiveSparseAttention,
    KnowingSparseAttention,
    UnknowingSparseAttention,
    EasySlidingWindowAttention,
)
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Tuple, List


attention_types = Literal[
    'dense', 
    'sparse', 
    'fixed', 
    'sparse2d', 
    'native',
    'simple-sparse',
    'knowing',
    'unknowing',
    'dilated',
    'bigbird',
    'smallbird',
    'smallerbird',
    'sliding-window',
    'bigbird-mod',
]

pos_encodings = Literal[
    'learned',
    'sinusoidal',
    'easy',
]


class TransformerBlock(nn.Module):

    def __init__(self,
                 context: int,
                 emb: int,
                 heads: int = 4,
                 ff_hidden_mult: int = 4,
                 dropout: float = 0.0,
                 attention_type: attention_types = 'dense',
                 depth: int = 0,
                 shared_predictor: nn.Module = None,
                 k = 4,
                 **kwargs):
        super().__init__()  
        if attention_type == 'dense':
            self.attend = MultiHeadAttention(heads, emb, emb, context, **kwargs)
        elif attention_type == 'sliding-window':
            self.attend = EasySlidingWindowAttention(heads, emb, emb, context, **kwargs)
        elif attention_type == 'sparse':
            self.attend = SparseSelfAttention(emb, context, n_heads=heads, k=k, **kwargs)
        elif attention_type == 'fixed':
            self.attend = BlocksparseFixedSelfAttention(emb, k=k, t=context, **kwargs)
        elif attention_type == 'sparse2d':
            self.attend = ReallySparseAttention(emb, context, k=k, n_heads=heads, **kwargs)
        elif attention_type == 'native':
            self.attend = NativeAttention(heads, emb, context, **kwargs)
        elif attention_type == 'dilated':
            self.attend = DynamicDilatedAttention(shared_predictor, 
                                                  emb,
                                                  k=k,
                                                  layer=depth, 
                                                  n_heads=heads,
                                                  **kwargs)
        elif attention_type == 'entmax':
            self.attend = AlphaEntmax(heads, emb, context, **kwargs)
        elif attention_type == 'simple-sparse':
            self.attend = NonadaptiveSparseAttention(emb, context, k=k, n_heads=heads, **kwargs)
        elif attention_type == 'knowing':
            self.attend = KnowingSparseAttention(emb, context, k=k, n_heads=heads, **kwargs)
        elif attention_type == 'unknowing':
            self.attend = UnknowingSparseAttention(emb, context, k=k, n_heads=heads, **kwargs)
        elif attention_type == 'bigbird':
            cfg = BigBirdConfig(context, heads, emb, k, 1)
            self.attend = BigBirdBlockSparseAttention(cfg)
        elif attention_type == 'smallbird':
            d = {
                'max_position_embeddings': context,
                'num_attention_heads': heads,
                'hidden_size': emb,
                'num_random_blocks': k,
                'block_size': 1,
                **kwargs,
            }
            cfg = SmallBirdConfig.from_dict(d)
            self.attend = SmallBirdSparseAttention(cfg)
        elif attention_type == 'smallerbird':
            d = {
                'max_position_embeddings': context,
                'num_attention_heads': heads,
                'hidden_size': emb,
                'num_random_blocks': k,
                'block_size': 1,
                **kwargs,
            }
            cfg = SmallerBirdConfig.from_dict(d)
            self.attend = SmallerBirdSparseAttention(cfg)
        elif attention_type == 'bigbird-mod':
            d = {
                'max_position_embeddings': context,
                'num_attention_heads': heads,
                'hidden_size': emb,
                'num_random_blocks': k,
                'block_size': 1,
                **kwargs,
            }
            cfg = SmallerBirdConfig.from_dict(d)
            self.attend = BigBirdModSelfAttention(cfg)
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

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.dropout(x + self.attend(self.norm1(x), attention_mask))
        x = self.dropout(x + self.ff(self.norm2(x)))
        return x

    def forward_for_plot(self, x: Tensor, attention_mask) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        normed1 = self.norm1(x)
        if isinstance(self.attend, MultiHeadAttention):
            m = torch.ones(x.size()).nonzero()
            s = torch.ones((m.size(0), ))
            v = s.clone()
        elif isinstance(self.attend, SparseSelfAttention) or isinstance(self.attend, DynamicDilatedAttention):
            m, s, v = self.attend.hyper(normed1)
            indices: Tensor = sparse.ngenerate(m,
                                               self.attend.gadditional,
                                               self.attend.nadditional,
                                               rng=(x.size(1),),
                                               relative_range=(2,),
                                               cuda='cuda' in util.d(x))
            # Single coord generated by hyper since 1-d case. Add x-coord for plotting
            x_coords = torch.arange(m.size(1), device=util.d(x))[None, :, None, None].expand_as(indices)
            m = torch.cat([x_coords, indices], dim=-1)
            m = torch.unique(m, dim=0)
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


class MaskedSequential(nn.Sequential):
    """To handle passing multiple values to the forward function"""
    def forward(self, x: Tensor, mask: Tensor):
        for module in self._modules.values():
            x = module(x, mask)
        return x


class LearnedPosEmbedding(nn.Module):

    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=emb_dim)

    def forward(self, seq: Tensor) -> Tensor:
        """Takes an embedded sequence and adds learned pos embeddings to it"""
        c = seq.shape[-2]
        pos = torch.arange(c, device=util.d(seq))
        pos_embeds = self.embedding(pos).expand_as(seq)
        return seq + pos_embeds


class EasyPosEmbedding(nn.Module):
    """
    Takes an embedding and adds a positional embedding to it.
    """
    def forward(self, seq: Tensor) -> Tensor:
        b, c, e = seq.size()
        pos = torch.arange(c, device=util.d(seq))[None, :].expand(b, -1)
        pos = pos[:, :, None]
        return torch.cat([seq, pos], dim=-1)


class SparseTransformer(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 context_len: int,
                 emb: int,
                 vocab_size: int,
                 attention_type: str = None,
                 attentions: List[str] = None,
                 pos_embedding: str = 'learned',
                 *args,
                 **kwargs):
        super().__init__()
        self.context_len = context_len
        self.vocab_size = vocab_size
        if pos_embedding == 'learned':
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb)
            self.pos_embedding = LearnedPosEmbedding(context_len, emb)
        elif pos_embedding == 'easy':
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb-1)
            self.pos_embedding = EasyPosEmbedding()
        elif pos_embedding == 'sinusoidal':
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb)
            self.pos_embedding = Summer(PositionalEncoding1D(emb))

        if (attentions is not None and 'dilated' in attentions) or (attention_type == 'dilated'):
            self.shared_predictor = nn.Sequential(nn.Linear(1, 10),
                                                  nn.ReLU(),
                                                  nn.Linear(10, 10),
                                                  nn.Tanh(),
                                                  nn.Linear(10, 2),
                                                  nn.Sigmoid())
        else:
            self.shared_predictor = None
        if attentions is not None:
            t_blocks = [TransformerBlock(context_len,
                                         emb, 
                                         *args, 
                                         depth=depth, 
                                         shared_predictor=self.shared_predictor, 
                                         attention_type=at,
                                         **kwargs) 
                            for depth, at in enumerate(attentions)]
        else:
            t_blocks = [TransformerBlock(context_len,
                                         emb, 
                                         *args, 
                                         depth=depth, 
                                         shared_predictor=self.shared_predictor, 
                                         attention_type=attention_type,
                                         **kwargs) 
                            for depth in range(n_blocks)]
        self.t_blocks = MaskedSequential(*t_blocks)

    def embed(self, x: Tensor) -> Tensor:
        # Here we'll do some embedding addition
        x = self.token_embedding(x)
        return self.pos_embedding(x)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        embedded = self.embed(x)  # (batch, context_len, emb)
        t_blocked = self.t_blocks(embedded, attention_mask)  # (batch, context_len, emb)
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
