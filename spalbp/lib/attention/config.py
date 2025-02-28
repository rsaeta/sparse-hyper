from dataclasses import dataclass
from config_base import _ConfigBase


@dataclass
class _AttentionConfig(_ConfigBase):
    emb: int
    heads: int
    context: int
    head_size: int


@dataclass
class BigBirdConfig(_AttentionConfig):
    num_random_blocks: int
    block_size: int = 1
    use_bias: bool = False


@dataclass
class MultiHeadAttentionConfig(_AttentionConfig):
    dropout: float = 0.0


@dataclass
class SlidingWindowConfig(MultiHeadAttentionConfig):
    window_size: int = 1


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
    head_size: int
    remove_rand_on_eval: bool
    bias_kv: bool
    densities_buffer: float


@dataclass
class NonAdaptiveSparseAttentionConfig(_OneDSparseAttention):
    means_init_method: str


@dataclass
class KnowingSparseAttentionConfig(NonAdaptiveSparseAttentionConfig):
    learn_means: bool


@dataclass
class AdaptiveSparseAttentionConfig(_OneDSparseAttention):
    hyper_hidden_dim: int
    hyper_hidden_depth: int


@dataclass
class ClusteredAttentionConfig(_AttentionConfig):
    num_clusters: int
    num_iterations: int
    window_size: int
    attention_dropout: float = 0.0
