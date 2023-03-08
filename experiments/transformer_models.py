import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from typing import Tuple


class SparseSelfAttention(nn.Module):
    """Do I need emb? Not sure"""

    def __init__(self,
                 emb: int,
                 context_len: int,
                 k: int,
                 hidden: int,
                 n_heads: int = 4,
                 gadditional: int = 2,
                 nadditional: int = 2,
                 mask: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.context_len = context_len
        self.emb = emb
        self.k = k
        self.gadditional = gadditional
        self.nadditional = nadditional
        self.mask = mask
        self.to_keys = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_queries = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_values = nn.Linear(emb, emb * n_heads, bias=False)
        self.unify = nn.Linear(emb * n_heads, emb)
        self.register_buffer('mvalues', torch.ones((k,)))
        self.to_param = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * k)  # One mean and one sigma
        )

    def hyper(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, context_len, emb = x.size()
        assert context_len == self.context_len, f'Expected contextlen equal to {self.context_len}. Got {context_len}'
        assert emb == self.emb, f'Expected embedded equal to {self.emb}. Got {emb}'

        # Add positional encoding at the attention layer
        # coords = torch.arange(context_len, dtype=torch.float, device=util.d(x)) / context_len
        # coords = coords[None, :, None, ].expand(batch_size, context_len, 1)

        # inp = torch.cat([x, coords], dim=2)
        params = self.to_param(x)  # (B, C, k*2) k means and sigmas for each point (1 degree of freedom)

        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(context_len, device=util.d(x), dtype=torch.float)
        diags = util.inv(diags, mx=context_len)
        diags = diags[None, :, None, None].expand(batch_size, -1, self.k, 1)  # (B, C, K, 1)

        means = params[:, :, :self.k].view(batch_size, -1, self.k, 1)  # Single mean for current point  (B, C, K, 1)
        sigmas = params[:, :, self.k:].view(batch_size, -1, self.k)  # (B, C, K)
        values = self.mvalues[None, None, :].expand(batch_size, context_len, -1)  # Expand to all points (B, C, K)

        means = diags - torch.nn.functional.softplus(means)
        means = sparse.transform_means(means, (context_len,))
        sigmas = sparse.transform_sigmas(sigmas, (context_len,))
        return means, sigmas, values

    def forward(self, x: Tensor) -> Tensor:
        means, sigmas, values = self.hyper(x)  # (B, C, k, 1); (B, C, k, 1); (B, C, k)
        batch, context, emb = x.size()  # (B, C, E)
        rank = means.size(-1)
        indices: Tensor = sparse.ngenerate(means,
                                           self.gadditional,
                                           self.nadditional,
                                           rng=(context,),
                                           relative_range=(2,),
                                           cuda='cuda' in util.d(x))  # (B, C, P, 1)
        assert ((indices < 0).sum().item() == 0) and ((indices >= context).sum().item() == 0), \
            f'Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; ' \
            f'indices >= {context}: {(indices >= context).sum().item()}'
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k * (2 ** rank + self.gadditional + self.nadditional)
        assert indices.size() == (batch, context, num_points, 1)
        densities = sparse.densities(indices_fl, means, sigmas).clone()  # (B, C, P, self.k)
        duplicates = util.nduplicates(indices).to(torch.bool)  # (B, C, P) boolean mask of duplicates all-but-one
        densities[duplicates, :] = 0  # Removes all duplicates

        # Normalize densities across all K probability distributions by summing
        densities = densities / densities.sum(dim=2, keepdim=True)

        weights = values[:, :, None, :].expand_as(densities)
        weights = weights * densities
        weights = weights.sum(dim=3)  # I don't get this at all (sum out the MVNs)

        # Because we have 1 degree of freedom, this adds the first index of the attention mask, while the second
        # is generated by our hyper network
        out = torch.arange(context, device=util.d(x))[None, :, None, None].expand(batch, -1, num_points, 1)
        indices = torch.cat([out, indices], dim=3)

        # Here we expand the indicies for each head in this transformer block
        indices = indices[:, None, :, :, :].expand(-1, self.n_heads, -1, -1, -1) \
            .contiguous() \
            .view(batch * self.n_heads, context * num_points, -1)

        weights = weights[:, None, :, :].expand(-1, self.n_heads, -1, -1) \
            .contiguous() \
            .view(batch * self.n_heads, context * num_points)

        # Perform key, query, value transformation
        keys = self.to_keys(x).view(batch, context, self.n_heads, emb)
        queries = self.to_queries(x).view(batch, context, self.n_heads, emb)
        values = self.to_values(x).view(batch, context, self.n_heads, emb)

        # Because the KQV tensors have head dimension, we need to fold them back to single
        keys = keys.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)
        queries = queries.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)
        values = values.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)

        queries = queries / (emb ** (1 / 4))  # Normalize along the embedding dimension
        keys = keys / (emb ** (1 / 4))

        indices_flattened = indices.view(batch * self.n_heads * context * num_points, -1)
        ar = torch.arange(batch * self.n_heads, device=util.d(x), dtype=torch.long)[:, None] \
            .expand(batch * self.n_heads, context * num_points) \
            .contiguous() \
            .view(batch * self.n_heads * context * num_points)

        squeries = queries[ar, indices_flattened[:, 0], :]
        skeys = keys[ar, indices_flattened[:, 1], :]
        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(batch * self.n_heads, context * num_points)
        # assert dot.size() == (batch * self.n_heads, context, context), \
        #     f'Expected dot to be of size {(batch * self.n_heads, context, context)}; got {dot.size()}'

        if self.mask:
            # breakpoint()
            util.mask_(dot, maskval=0.0, mask_diagonal=False)

            dot = sparse.logsoftmax(indices, weights * dot, (context, context)).exp()

        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=values)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads * emb)
        return self.unify(out)


class TransformerBlock(nn.Module):

    def __init__(self,
                 context: int,
                 emb: int,
                 k: int = 2,
                 heads: int = 4,
                 ff_hidden_mult: int = 4,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        self.attend = SparseSelfAttention(emb, context, k, ff_hidden_mult, heads, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x: Tensor) -> Tensor:
        attended = self.attend(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)

        ff = self.ff(x)
        x = self.norm2(ff + x)
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
        # self.to_prob = nn.Linear(emb, num_classes)

    def embed(self, x: Tensor) -> Tensor:
        # Here we'll do some embedding addition
        x = self.token_embedding(x)
        b, c, e = x.size()
        positions = self.pos_embedding(torch.arange(self.context_len, dtype=torch.int, device=util.d(x)))[None, :, :] \
            .expand(b, -1, -1)
        return positions + x

    def forward(self, x: Tensor) -> Tensor:
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
