from dataclasses import dataclass

import torch
from torch import nn, Tensor

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from ..attention import from_config, _AttentionConfig

from sparse import util

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Tuple, List


attention_types = Literal[
    "dense",
    "sparse",
    "fixed",
    "sparse2d",
    "native",
    "simple-sparse",
    "knowing",
    "unknowing",
    "dilated",
    "bigbird",
    "smallbird",
    "smallerbird",
    "sliding-window",
    "bigbird-mod",
]

pos_encodings = Literal["learned", "sinusoidal", "easy"]


@dataclass
class TransformerBlockConfig:
    attention: _AttentionConfig
    dropout: float
    embedding_dim: int
    ff_hidden_mult: int
    repeat: int


@dataclass
class ModelConfig:
    type: str
    embedding_dim: int
    context_size: int
    positional_encoding_type: str
    t_blocks: List[TransformerBlockConfig]
    vocab_size: int
    num_classes: int


class TransformerBlock(nn.Module):
    @classmethod
    def from_config(cls, cfg: TransformerBlockConfig):
        return cls(cfg.embedding_dim, cfg.attention, cfg.ff_hidden_mult, cfg.dropout)

    def __init__(
        self,
        emb: int,
        attention_config: _AttentionConfig,
        ff_hidden_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attend = from_config(attention_config)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        attended = self.attend(self.norm1(x), attention_mask)
        if type(attended) is tuple:
            attended, loss = attended
        else:
            loss = None
        x = self.dropout(x + attended)
        x = self.dropout(x + self.ff(self.norm2(x)))
        return x, loss


class MaskedSequential(nn.Sequential):
    """To handle passing multiple values to the forward function"""

    def forward(self, x: Tensor, mask: Tensor):
        loss_acc = torch.tensor(0.0, requires_grad=True, device=x.device)
        for module in self._modules.values():
            x, aux_loss = module(x, mask)
            if aux_loss is not None:
                loss_acc = loss_acc + aux_loss
        return x, loss_acc


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


class SinusoidalPosEncoding(nn.Module):
    def __init__(self, emb: int):
        super().__init__()
        self.encoder = Summer(PositionalEncoding1D(emb))

    def forward(self, seq: Tensor) -> Tensor:
        return self.encoder(seq)


class TransformerModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        context_len = cfg.context_size
        vocab_size = cfg.vocab_size
        emb = cfg.embedding_dim
        if cfg.positional_encoding_type == "learned":
            self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=emb
            )
            self.pos_embedding = LearnedPosEmbedding(context_len, emb)
        elif cfg.positional_encoding_type == "easy":
            self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=emb - 1
            )
            self.pos_embedding = EasyPosEmbedding()
        elif cfg.positional_encoding_type == "sinusoidal":
            self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=emb
            )
            self.pos_embedding = SinusoidalPosEncoding(emb)
        else:
            raise ValueError(
                f"Unknown positional encoding type: {cfg.positional_encoding_type}"
            )
        t_blocks = []
        for tblock_cfg in cfg.t_blocks:
            sub_tblocks = [
                TransformerBlock.from_config(tblock_cfg)
                for _ in range(tblock_cfg.repeat)
            ]
            for t_block in sub_tblocks:
                t_blocks.append(t_block)
        self.t_blocks = MaskedSequential(*t_blocks)

    def embed(self, x: Tensor) -> Tensor:
        # Here we'll do some embedding addition
        x = self.token_embedding(x)
        return self.pos_embedding(x)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.embed(x)  # (batch, context_len, emb)
        t_blocked, aux_loss = self.t_blocks(
            embedded, attention_mask
        )  # (batch, context_len, emb)
        done = self.post_tblocks(t_blocked)
        return done, aux_loss

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


class ClassificationTransformer(TransformerModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.to_prob = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def post_tblocks(self, x: Tensor) -> Tensor:
        x = x[:, 0, :]  # (batch, embed) just take the first token
        x = self.to_prob(x)  # (batch, num_classes)
        x = torch.nn.functional.log_softmax(
            x, dim=1
        )  # (batch, num_classes) the probability distribution over classes
        return x


class GeneratingTransformer(TransformerModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.vocab_size = cfg.vocab_size
        self.to_probs = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def post_tblocks(self, x: Tensor) -> Tensor:
        b, c, e = x.size()  # batch, context, embed
        x = self.to_probs(x.view(b * c, e)).view(b, c, self.vocab_size)
        return x
