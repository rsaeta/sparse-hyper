from typing import Tuple
import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from spalbp.lib.attention.config import AdaptiveSparseAttentionConfig


class _OneDimensionalSparseAttention(nn.Module):

    def __init__(self,
                 emb: int,
                 n_heads: int,
                 *,
                 head_size: int = 16,
                 k: int = 4,
                 gadditional: int = 1,
                 nadditional: int = 4):
        super().__init__()
        self.emb = emb
        self.n_heads = n_heads
        self.k = k
        self.gadditional = gadditional
        self.nadditional = nadditional
        self.head_size = head_size

        self.to_keys = nn.Linear(emb, head_size * n_heads, bias=False)
        self.to_queries = nn.Linear(emb, head_size * n_heads, bias=False)
        self.to_values = nn.Linear(emb, head_size * n_heads, bias=False)
        self.unify = nn.Linear(head_size * n_heads, emb)

        self.register_buffer('mvalues', torch.ones((k,)))

    def hyper(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("You must implement this yourself")

    def batched_index_select(self, values: Tensor, indices: Tensor):
        last_dim = values.shape[-1]
        return values.gather(2, self.expand_dim(indices, -1, last_dim))

    def expand_dim(self, t, dim, k):
        t = t.unsqueeze(dim)
        expand_shape = [-1] * len(t.shape)
        expand_shape[dim] = k
        return t.expand(*expand_shape)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False) -> torch.Tensor:
        means, sigmas, values = self.hyper(x)  # (B, H, C, k, 1); (B, H, C, k, 1); (B, H, C, k)
        batch, context, emb = x.size()  # (B, C, E)
        rank = means.size(-1)
        indices: Tensor = sparse.ngenerate(means,
                                           self.gadditional if self.training else 0,  # For evaluation, only get nearest
                                           self.nadditional if self.training else 0,  # index for each point
                                           rng=(context,),
                                           relative_range=(3,),
                                           cuda='cuda' in util.d(x))  # (B, H, C, P, 1)
        if not ((indices < 0).sum().item() == 0) and ((indices >= context).sum().item() == 0):
            print(f'Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; '
                  f'indices >= {context}: {(indices >= context).sum().item()}')
            breakpoint()
        assert ((indices < 0).sum().item() == 0) and ((indices >= context).sum().item() == 0), \
            f'Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; ' \
            f'indices >= {context}: {(indices >= context).sum().item()}'
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k * (2 ** rank + ((self.gadditional + self.nadditional) if self.training else 0))
        assert indices.size() == (
            batch, self.n_heads, context, num_points, 1), f'Expected size {(batch, context, num_points, 1)}. ' \
                                                          f'Got {indices.size()}'
        densities = sparse.densities(indices_fl, means, sigmas).clone()  # (B, H, C, P, self.k)
        duplicates = util.nduplicates(indices).to(torch.bool)  # (B, C, P) boolean mask of duplicates all-but-one
        densities[duplicates, :] = 0  # Removes all duplicates

        # Normalize densities across all K probability distributions by summing
        densities = densities / (densities.sum(dim=2, keepdim=True) + 1e-8)  # (B, H, C, P, self.k)

        weights = values[:, :, :, None, :].expand_as(densities)
        weights = weights * densities
        weights = weights.sum(dim=4)

        # Perform key, query, value transformation
        K = self.to_keys(x).view(batch, context, self.n_heads, self.head_size)
        Q = self.to_queries(x).view(batch, context, self.n_heads, self.head_size)
        V = self.to_values(x).view(batch, context, self.n_heads, self.head_size)

        K = K / (self.head_size ** (1 / 2))
        Q = Q / (self.head_size ** (1 / 2))

        flat_indices = indices.view(batch, self.n_heads, -1)  # [b, h, c * p]
        gathered_keys = self.batched_index_select(K.transpose(1, 2), flat_indices)  # [b, h, c * p, d]
        dots = torch.einsum('bhcd,bhcqd -> bhcq',
                            Q.transpose(1, 2),
                            gathered_keys.view(batch, self.n_heads, context, num_points, -1))  # [b, h, c, p]
        gathered_values = self.batched_index_select(V.transpose(1, 2), flat_indices)  # [b, h, c * p, d]
        gathered_values = gathered_values.view(batch, self.n_heads, context, num_points, -1)  # [b, h, c, p, d]
        new_weights = weights * dots  # Weigh attention scores by densities
        new_weights = new_weights.softmax(dim=-1)  # Normalize attention scores
        head_reps = gathered_values * new_weights.unsqueeze(-1)  # Weigh values by attention scores
        head_reps = head_reps.sum(dim=-2)  # Sum over weighted values [b, h, c, d]
        united = head_reps.view(batch, context, -1)  # [b, c, h * d]
        new_out = self.unify(united)  # [b, c, e]
        return new_out
        """
        # Because we have 1 degree of freedom, this adds the first index of the attention mask, while the second
        # is generated by our hyper network
        out = torch.arange(context, device=util.d(x))[None, None, :, None, None]\
            .expand(batch, self.n_heads, -1, num_points, 1)
        indices = torch.cat([out, indices], dim=-1)

        # Here we expand the indicies for each head in this models block
        indices = indices.contiguous() \
            .view(batch * self.n_heads, context * num_points, -1)
        # Here indices has a bunch of matrices where are just lists of coordinates.
        # One matrix for each head for the whole input

        # Now expand weights (= values * densities) for each head
        weights = weights.contiguous() \
            .view(batch * self.n_heads, context * num_points)

        
        # Because the KQV tensors have head dimension, we need to fold them back to single
        # First, transpose the head and context dimension to make it (batch, heads, context, emb)
        # Then we just get a list of matrices of size (context, emb) which would essentially be the
        # encoded sentences for each head, for each element in batch.

        K = K.transpose(1, 2).contiguous().view(batch * self.n_heads, context, -1)
        Q = Q.transpose(1, 2).contiguous().view(batch * self.n_heads, context, -1)
        V = V.transpose(1, 2).contiguous().view(batch * self.n_heads, context, -1)
        batch2, np, _ = indices.shape
        batch_is = torch.arange(batch2, dtype=torch.long, device=util.d(x))[None, :].expand(np, -1).t().reshape(-1)
        indices2 = torch.cat([batch_is[:, None], indices.view(-1, 2)], dim=-1)
        dot = util.calc_vals(Q, K.transpose(-2, -1), indices2).view(batch2, -1)
        smax_dot = sparse.logsoftmax(indices, weights * dot, (context, context)).exp()
        out = sparse.batchmm(indices, smax_dot, size=(context, context), xmatrix=V)  # [B * H, C, E]
        out = out.view(batch, context, self.n_heads * self.head_size)
        to_return = self.unify(out)
#        breakpoint()
        if output_attentions:
            coo = torch.sparse_coo_tensor(indices2.t(), dot.view(-1), (batch*self.n_heads, context, context))
            dense_attention = coo.to_dense()
            dense_attention = dense_attention.view(batch, self.n_heads, self.context_len, -1)  # [B, H, C, C]
            return to_return, dense_attention
        
        return to_return"""


class NonadaptiveSparseAttention(_OneDimensionalSparseAttention):

    @classmethod
    def from_config(cls, config: AdaptiveSparseAttentionConfig):
        return cls(config.emb,
                   config.context,
                   k=config.k,
                   n_heads=config.heads,
                   gadditional=config.gadditional,
                   nadditional=config.nadditional,
                   sigma_scale=config.sigma_scale,
                   transformation_method=config.transformation_method)

    def __init__(self,
                 emb: int,
                 context_len: int,
                 *,
                 k: int = 8,
                 n_heads: int = 4,
                 gadditional: int = 2,
                 nadditional: int = 2,
                 sigma_scale: float = 1.,
                 transformation_method: str = 'sigmoid'):
        super().__init__(emb, n_heads, k=k, gadditional=gadditional, nadditional=nadditional)
        self.pmeans = torch.nn.Parameter(torch.rand((context_len, k, 1)))
        self.psigmas = torch.nn.Parameter(torch.rand((context_len, k)))
        # Non-learnabe
        self.register_buffer('pvalues', torch.ones(k))
        self.sigma_scale = sigma_scale
        self.transformation_method = transformation_method

    def hyper(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.k
        means = self.pmeans[None, :, :, :].expand(b, c, k, 1)
        sigmas = self.psigmas[None, :, :].expand(b, c, k)
        values = self.pvalues[None, None, :].expand(b, c, k)

        means = sparse.transform_means(means, (c,), method=self.transformation_method)
        sigmas = sparse.transform_sigmas(sigmas, (c,)) * self.sigma_scale

        return means, sigmas, values


class UnknowingSparseAttention(_OneDimensionalSparseAttention):
    def __init__(self,
                 emb: int,
                 context_len: int,
                 *,
                 n_heads: int = 4,
                 gadditional: int = 2,
                 nadditional: int = 2,
                 sigma_scale: float = 1.,
                 transformation_method: str = 'modulo'):
        k = 1
        super().__init__(emb, n_heads, k=k, gadditional=gadditional, nadditional=nadditional)

        self.pmeans = torch.rand((context_len, k, 1), device='cuda')
        self.pmeans[45, 0, 0] = 112.

        self.psigmas = torch.nn.Parameter(torch.rand((context_len, k)))
        # Non-learnabe
        self.register_buffer('pvalues', torch.ones(k))
        self.sigma_scale = sigma_scale
        self.transformation_method = transformation_method

    def hyper(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.k
        means = self.pmeans[None, :, :, :].expand(b, c, k, 1)
        sigmas = self.psigmas[None, :, :].expand(b, c, k)
        values = self.pvalues[None, None, :].expand(b, c, k)

        means = sparse.transform_means(means, (c,), method=self.transformation_method)
        sigmas = sparse.transform_sigmas(sigmas, (c,)) * self.sigma_scale
        return means, sigmas, values


class KnowingSparseAttention(_OneDimensionalSparseAttention):
    def __init__(self,
                 emb: int,
                 context_len: int,
                 *,
                 n_heads: int = 4,
                 gadditional: int = 2,
                 nadditional: int = 2,
                 sigma_scale: float = 1.,
                 transformation_method: str = 'modulo'):
        k = 1
        super().__init__(emb, n_heads, k=k, gadditional=gadditional, nadditional=nadditional)

        self.pmeans = torch.randint(100, 32000, (context_len, k, 1), device='cuda')
        for i in range(45, 105):
            self.pmeans[i, 0, 0] = i + 70

        self.psigmas = torch.nn.Parameter(torch.rand((context_len, k)))
        # Non-learnabe
        self.register_buffer('pvalues', torch.ones(k))
        self.sigma_scale = sigma_scale
        self.transformation_method = transformation_method

    def hyper(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.k
        means = self.pmeans[None, :, :, :].expand(b, c, k, 1)
        sigmas = self.psigmas[None, :, :].expand(b, c, k)
        values = self.pvalues[None, None, :].expand(b, c, k)

        means = sparse.transform_means(means, (c,), method=self.transformation_method)
        sigmas = sparse.transform_sigmas(sigmas, (c,)) * self.sigma_scale
        return means, sigmas, values


class SparseSelfAttention(_OneDimensionalSparseAttention):

    @classmethod
    def from_config(cls, config: AdaptiveSparseAttentionConfig):
        return cls(config.emb,
                   config.context,
                   head_size=config.head_size,
                   k=config.k,
                   hidden=config.hyper_hidden_dim,
                   n_heads=config.heads,
                   gadditional=config.gadditional,
                   nadditional=config.nadditional)

    def __init__(self,
                 emb: int,
                 context_len: int,
                 head_size: int,
                 *,
                 k: int = 8,
                 hidden: int = 4,
                 n_heads: int = 4,
                 gadditional: int = 2,
                 nadditional: int = 2):
        super().__init__(emb,
                         n_heads,
                         head_size=head_size,
                         k=k,
                         gadditional=gadditional,
                         nadditional=nadditional)
        self.context_len = context_len
        self.to_param = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * k * n_heads)  # One mean and one sigma per head
        )

    def hyper(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, context_len, emb = x.size()
        assert context_len == self.context_len, f'Expected contextlen equal to {self.context_len}. Got {context_len}'
        assert emb == self.emb, f'Expected embedded equal to {self.emb}. Got {emb}'

        params = self.to_param(x)  # (B, C, h*k*2) k means and sigmas for each point (1 degree of freedom)

        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(context_len, device=util.d(x), dtype=torch.float)
        diags = util.inv(diags, mx=context_len)
        diags = diags[None, :, None, None].expand(batch_size, -1, self.k, 1)  # (B, C, K, 1)
        means = params[:, :, :self.n_heads * self.k, None] \
            .view(batch_size, context_len, self.n_heads, self.k, 1) \
            .permute(0, 2, 1, 3, 4)  # Single mean for current point  (B, H, C, K, 1)
        sigmas = params[:, :, self.n_heads * self.k:] \
            .view(batch_size, context_len, self.n_heads, self.k) \
            .permute(0, 2, 1, 3)  # (B, H, C, K)
        values = self.mvalues[None, None, None, :].expand(batch_size, self.n_heads, context_len,
                                                          -1)  # Expand to all points (B, C, K)

        # means = diags - torch.nn.functional.softplus(means)
        means = sparse.transform_means(means, (context_len,))
        sigmas = sparse.transform_sigmas(sigmas, (context_len,))
        return means, sigmas, values
