from dataclasses import dataclass
from torch import nn, Tensor

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


@dataclass
class NativeTransformerConfig:
    emb: int
    heads: int
    ff_hidden_mult: int
    dropout: float
    depth: int
    context: int
    vocab: int


class NativeTransformer(nn.Module):
    @classmethod
    def from_config(cls, config: NativeTransformerConfig):
        return cls(**config.__dict__)

    def __init__(
        self,
        emb: int,
        heads: int = 4,
        ff_hidden_mult: int = 4,
        dropout: float = 0.0,
        depth: int = 0,
        context: int = 0,
        vocab: int = 32000,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=emb)
        self.pos_embedding = Summer(PositionalEncoding1D(emb))
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
        out = self.transformer(emb, (~(mask.bool())))
        return self.to_vocab(out)
