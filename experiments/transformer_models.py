import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from typing import Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def get_eyes(j, l):
    return torch.arange((j % l) + 1) + ((j // l) * l)

def get_coords(t, k):
    num_coords = (torch.arange(k) + 1).sum() * (t // k) + torch.arange(t % k).sum()
    coords = torch.empty((num_coords, 2))
    i = 0
    for j in range(t):
        eyes = get_eyes(j, k)
        for eye in eyes:
            coords[i, 0] = j
            coords[i, 1] = eye
            i += 1
    return coords


class SparseHead1(nn.Module):

    def __init__(self, k=8):
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        b, t, e = x.size()
        coords = get_coords(t, self.k)


class BlocksparseSelfAttention(nn.Module):

    def __init__(self, emb: int):
        super().__init__()
        self.emb = emb
        self.to_keys = nn.Linear(emb, emb)
        self.to_queries = nn.Linear(emb, emb)
        self.to_values = nn.Linear(emb, emb)

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.emb, f'Input tensor must have embedding size {self.emb}. Got {x.size(-1)}'
        K, Q, V = self.to_keys(x), self.to_queries(x), self.to_values(x)



class SparseSelfAttention(nn.Module):

    def __init__(self,
                 emb: int,
                 context_len: int,
                 *,
                 k: int = 8,
                 hidden: int = 4,
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
        # Here indices has a bunch of matrices where are just lists of coordinates.
        # One matrix for each head for the whole input

        # Now expand weights (= values * densities) for each head
        weights = weights[:, None, :, :].expand(-1, self.n_heads, -1, -1) \
            .contiguous() \
            .view(batch * self.n_heads, context * num_points)

        # Perform key, query, value transformation
        K = self.to_keys(x).view(batch, context, self.n_heads, emb)
        Q = self.to_queries(x).view(batch, context, self.n_heads, emb)
        V = self.to_values(x).view(batch, context, self.n_heads, emb)

        # Because the KQV tensors have head dimension, we need to fold them back to single
        # First, transpose the head and context dimension to make it (batch, heads, context, emb)
        # Then we just get a list of matrices of size (context, emb) which would essentially be the
        # encoded sentences for each head, for each element in batch.
        K = K.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)
        Q = Q.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)
        V = V.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)

        Q = Q / (emb ** (1 / 4))  # Normalize along the embedding dimension
        K = K / (emb ** (1 / 4))

        # Now this is just a list of coordinates arranged by batch,head,context
        # indices_flattened = indices.view(batch * self.n_heads * context * num_points, -1)
        # This is an array that refers the coordinates in the above tensor back to the original values in KQV matrices
        # ar = torch.arange(batch * self.n_heads, device=util.d(x), dtype=torch.long)[:, None] \
        #     .expand(batch * self.n_heads, context * num_points) \
        #     .contiguous() \
        #     .view(batch * self.n_heads * context * num_points)

        # It's important to note we have 1 degree of freedom (see out (torch.cat([out, indices])) was generated via
        # torch.arange in order to have "k" points for every token in the attention matrix. This means that every
        # element will be represented as a query. However, since indices were generated via the hyper network, the keys
        # are actually going to sparse (80 vs 200). This is what indices_flattened[:, {0,1}] refers to.
        # NOTE: This explodes the size of Q or K since there's a lot of repeats in ar.
        # squeries = Q[ar, indices_flattened[:, 1], :]
        # skeys = K[ar, indices_flattened[:, 0], :]

        # In order to get the dot between QK for each of the elements acting as Q and K, somehow this works out
        # to calculating that dot product in a flattened form
        # dot_ish = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(batch * self.n_heads, context * num_points)
        # breakpoint()
        batch2, np, _ = indices.shape
        batch_is = torch.arange(batch2, dtype=torch.long, device=util.d(x))[None, :].expand(np, -1).t().reshape(-1)
        indices2 = torch.cat([batch_is[:, None], indices.view(-1, 2)], dim=-1)
        dot = util.calc_vals(Q, K.transpose(-2, -1), indices2).view(batch2, -1)
        if self.mask:
            util.mask_(dot, maskval=0.0, mask_diagonal=False)
            dot = sparse.logsoftmax(indices, weights * dot, (context, context), method='naive').exp()
        # breakpoint()
        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=V)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads * emb)
        return self.unify(out)


class TransformerBlock(nn.Module):

    def __init__(self,
                 context: int,
                 emb: int,
                 heads: int = 4,
                 ff_hidden_mult: int = 4,
                 dropout: float = 0.0,
                 attention_type: Literal['dense', 'sparse'] = 'dense',
                 **kwargs):
        super().__init__()
        if attention_type == 'dense':
            self.attend = MultiHeadAttention(heads, emb, emb, context)
        elif attention_type == 'sparse':
            self.attend = SparseSelfAttention(emb, context, n_heads=heads, **kwargs)
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

# TAKEN FROM https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py 
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

