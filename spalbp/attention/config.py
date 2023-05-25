from dataclasses import dataclass
from spalbp.config_base import _ConfigBase


@dataclass
class _AttentionConfig(_ConfigBase):
    emb: int
    heads: int
    context: int
    head_size: int


@dataclass
class MultiHeadAttentionConfig(_AttentionConfig):
    dropout: float = 0.0


@dataclass
class AlphaEntmaxAttentionConfig(_AttentionConfig):
    alpha: float = 1.5


@dataclass
class _OneDSparseAttention(_AttentionConfig):
    k: int
    gadditional: int
    nadditional: int
    sigma_scale: float
    transformation_method: str

@dataclass
class AdaptiveSparseAttentionConfig(_OneDSparseAttention):
    hyper_hidden_dim: int
