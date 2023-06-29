from dataclasses import dataclass
from torch import nn, Tensor

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from .transformer_models import LearnedPosEmbedding


@dataclass
class NativeTransformerConfig:
    emb: int
    heads: int
    ff_hidden_mult: int
    dropout: float
    depth: int
    context: int
    vocab: int
    pos_embedding: str


class NativeTransformer(nn.Module):
    @classmethod
    def from_config(cls, config: NativeTransformerConfig):
        return cls(
            emb=config.emb,
            heads=config.heads,
            ff_hidden_mult=config.ff_hidden_mult,
            dropout=config.dropout,
            depth=config.depth,
            context=config.context,
            vocab=config.vocab,
            pos_embedding=config.pos_embedding,
        )

    def __init__(
        self,
        emb: int,
        heads: int = 4,
        ff_hidden_mult: int = 4,
        dropout: float = 0.0,
        depth: int = 0,
        context: int = 0,
        vocab: int = 32000,
        pos_embedding: str = "learned",
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=emb)
        if pos_embedding == "learned":
            self.pos_embedding = LearnedPosEmbedding(context, emb)
        elif pos_embedding == "sinusoidal":
            self.pos_embedding = Summer(PositionalEncoding1D(emb))
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_embedding}")
        self.nheads = heads
        self.context = context
        encoder_layer = nn.TransformerEncoderLayer(
            emb,
            heads,
            ff_hidden_mult * emb,
            dropout,
            batch_first=True,
            norm_first=True,
            device="cuda",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.to_vocab = nn.Linear(emb, vocab)

    def embed(self, x):
        # Here we'll do some embedding addition
        x = self.token_embedding(x)
        return self.pos_embedding(x)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        emb = self.embed(x)
        mask = (
            mask[:, None, :, :]
            .expand(-1, self.nheads, -1, -1)
            .reshape(-1, self.context, self.context)
        )
        mask = mask[0]
        out = self.transformer(emb, (~(mask.bool())))
        return self.to_vocab(out), 0
