from . import config

from .dynamic_sparse_attentions import (
    SparseSelfAttention, KnowingSparseAttention, UnknowingSparseAttention, NonadaptiveSparseAttention
)
from .bigbird import BigBirdBlockSparseAttention
from .misc_attention import AlphaEntmax
from .native_attentions import (
    MultiHeadAttention, NativeAttention, EasySlidingWindowAttention, SlidingWindowWithGlobalAttention
)
from .clustered_attention import ClusteredAttention


# Stores a mapping between attention types and their corresponding config and impl classes
type_to_classes = {
    'dense': (config.MultiHeadAttentionConfig, MultiHeadAttention),
    'native': (config.MultiHeadAttentionConfig, NativeAttention),
    'sparse': (config.AdaptiveSparseAttentionConfig, SparseSelfAttention),
    'knowing': (config.AdaptiveSparseAttentionConfig, KnowingSparseAttention),
    'unknowing': (config.AdaptiveSparseAttentionConfig, UnknowingSparseAttention),
    'simple_sparse': (config.NonAdaptiveSparseAttentionConfig, NonadaptiveSparseAttention),
    'entmax': (config.AlphaEntmaxAttentionConfig, AlphaEntmax),
    'sliding-window': (config.SlidingWindowConfig, EasySlidingWindowAttention),
    'bigbird': (config.BigBirdConfig, BigBirdBlockSparseAttention),
    'sliding-window-with-global': (config.SlidingWindowConfig, SlidingWindowWithGlobalAttention),
    'clustered': (config.ClusteredAttentionConfig, ClusteredAttention),
}


def from_config(d: dict):
    attention_type = d['attention_type']
    if attention_type not in type_to_classes:
        raise ValueError(f'Unknown attention type: {attention_type}')
    config_cls, impl_cls = type_to_classes[attention_type]
    cfg = config_cls.from_dict(d)
    return impl_cls.from_config(cfg)
