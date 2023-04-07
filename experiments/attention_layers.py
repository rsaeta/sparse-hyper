import torch
from torch import nn, Tensor
import wandb
import entmax

from _context import sparse
from sparse import util

from typing import Tuple


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


def get_coords3(t, k, dilation: float):
    offsets = torch.arange(-k, k+1)*dilation
    derp = torch.arange(t)[None,:].expand(offsets.size(0), -1).t()
    derp = (offsets.expand_as(derp) + derp)[:,:,None]
    firsts = torch.arange(derp.size(0))[:, None, None].expand_as(derp)
    coords = torch.cat([firsts, derp], dim=-1)
    return coords
    # num_coords = (torch.arange(k) + 1).sum() * (t // k) + torch.arange(t % k).sum()
    # coords = torch.empty((num_coords, 2))
    # i = 0
    # for j in range(t):
    #     eyes = get_eyes(j, k)
    #     for eye in eyes:
    #         coords[i, 0] = j
    #         coords[i, 1] = eye
    #         i += 1
    # return coords


class SparseHead1(nn.Module):

    def __init__(self, t=200, k=8):
        super().__init__()
        self.k = k
        self.register_buffer('indices', get_coords(t, k))

    def forward(self, k: Tensor, q: Tensor, v: Tensor) -> Tensor:
        b, t, e = k.size()
        expanded_coords = self.indices[None, :].expand(b, -1, -1).long()
        _, np, dims = expanded_coords.size()
        bcoords = torch.arange(b, device=util.d(k))[:, None].expand(-1, np).reshape(-1)[:, None]
        coords = expanded_coords.reshape(b*np, -1)
        indices = torch.cat([bcoords, coords], dim=-1).long()
        dot = util.calc_vals(k, q.transpose(-2, -1), indices).view(b, -1)
        out = sparse.batchmm(expanded_coords, dot, size=(t, t), xmatrix=v)
        return out


def get_coords2(t, k):
    coords = torch.arange(t//k) * k
    np = (t - coords).sum()
    toret = torch.empty((np, 2))
    i = 0
    for c in coords:
        for y in torch.arange(c, t):
            toret[i, 0] = y
            toret[i, 1] = c
            i += 1
    assert i == np
    return toret


class SparseHead2(nn.Module):

    def __init__(self, t=200, k=8):
        super().__init__()
        self.register_buffer('indices', get_coords2(t, k))

    def forward(self, k: Tensor, q: Tensor, v: Tensor) -> Tensor:
        b, t, e = k.size()
        expanded_coords = self.indices[None, :].expand(b, -1, -1).long()
        _, np, dims = expanded_coords.size()
        bcoords = torch.arange(b, device=util.d(k))[:, None].expand(-1, np).reshape(-1)[:, None]
        coords = expanded_coords.reshape(b * np, -1)
        indices = torch.cat([bcoords, coords], dim=-1).long()
        dot = util.calc_vals(k, q.transpose(-2, -1), indices).view(b, -1)
        out = sparse.batchmm(expanded_coords, dot, size=(t, t), xmatrix=v)
        return out


class BlocksparseFixedSelfAttention(nn.Module):

    def __init__(self, emb: int, t: int, k: int, **kwargs):
        super().__init__()
        self.emb = emb
        self.to_keys = nn.Linear(emb, emb)
        self.to_queries = nn.Linear(emb, emb)
        self.to_values = nn.Linear(emb, emb)
        self.head1 = SparseHead1(t, k)
        self.head2 = SparseHead2(t, k)
        self.unify = nn.Linear(emb*2, emb)

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.emb, f'Input tensor must have embedding size {self.emb}. Got {x.size(-1)}'
        K, Q, V = self.to_keys(x), self.to_queries(x), self.to_values(x)
        h1 = self.head1(K, Q, V)
        h2 = self.head2(K, Q, V)
        comb = torch.cat([h1, h2], dim=-1)
        return self.unify(comb)


class DynamicDilatedAttention(nn.Module):
    def __init__(self, 
                 stride_predictor: nn.Module, 
                 emb: int, 
                 k: int = 4, 
                 layer: int = 0, 
                 gadditional: int = 2,
                 nadditional: int = 0,
                 n_heads: int = 4,
                 **kwargs):
        super().__init__()
        self.emb = emb
        self.k = k
        self.layer = layer
        self.gadditional = gadditional
        self.nadditional = nadditional
        self.to_keys = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_queries = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_values = nn.Linear(emb, emb * n_heads, bias=False)
        self.unify = nn.Linear(emb * n_heads, emb)
        self.n_heads = n_heads
        # parameter sharing module that predicts the dilation and sigma given the layer number
        self.stride_predictor = stride_predictor
        self.register_buffer('mvalues',torch.ones(2*k+1))

    def hyper(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        params = self.stride_predictor(torch.tensor([self.layer], device=util.d(x)).float())
        dilation, sigma = params[0], params[1]
        wandb.log({f'{self.layer}.dilation': dilation, f'{self.layer}.sigma': sigma}, commit=False)
        b, t = x.size(0), x.size(-2)
        offsets = torch.arange(-self.k, self.k+1, device=util.d(x))*dilation
        means = torch.arange(t, device=util.d(x))[None,:].expand(offsets.size(0), -1).t()
        means = (offsets.expand_as(means) + means)[None,:,:].expand(b, -1, -1)
        sigmas = sigma[None,None].expand_as(means).squeeze()
        mvalues = self.mvalues[None, :].expand_as(sigmas)
        means = sparse.transform_means(means, (t,), 'clamp')
        sigmas = sparse.transform_sigmas(sigmas, (t,))
        return means[..., None], sigmas, mvalues

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
        num_points = (2*self.k+1) * (2 ** rank + self.gadditional + self.nadditional)
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
        # if self.mask:
            # indices = util.flip(indices)
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

        batch2, np, _ = indices.shape
        batch_is = torch.arange(batch2, dtype=torch.long, device=util.d(x))[None, :].expand(np, -1).t().reshape(-1)
        indices2 = torch.cat([batch_is[:, None], indices.view(-1, 2)], dim=-1)
        dot = util.calc_vals(Q, K.transpose(-2, -1), indices2).view(batch2, -1)
        # dot = weights * dot
        dot = sparse.logsoftmax(indices, weights * dot, (context, context)).exp()
        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=V)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads * emb)
        return self.unify(out)


class ReallySparseAttention(nn.Module):

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
        self.to_keys = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_queries = nn.Linear(emb, emb * n_heads, bias=False)
        self.to_values = nn.Linear(emb, emb * n_heads, bias=False)
        self.mask = mask
        self.unify = nn.Linear(emb * n_heads, emb)
        self.register_buffer('mvalues', torch.ones((k,)))
        self.to_param = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 * k)  # Two mean and one sigma
        )

    def hyper(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, context_len, emb = x.size()
        assert context_len == self.context_len, f'Expected contextlen equal to {self.context_len}. Got {context_len}'
        assert emb == self.emb, f'Expected embedded equal to {self.emb}. Got {emb}'

        # inp = torch.cat([x, coords], dim=2)
        params = self.to_param(x)  # (B, C, k*3) k means and sigmas for each point (2 degrees of freedom)

        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(context_len, device=util.d(x), dtype=torch.float)
        # diags = util.inv(diags, mx=context_len)
        diags = diags[None, :, None, None].expand(batch_size, -1, self.k, 1)  # (B, C, K, 1)

        means = params[:, :, :self.k*2].view(batch_size, -1, self.k, 2)  # mean for current point  (B, C, K, 2)
        sigmas = params[:, :, self.k*2:].view(batch_size, -1, self.k)  # (B, C, K)  # 1 sigma for current point
        values = self.mvalues[None, None, :].expand(batch_size, context_len, -1)  # Expand to all points (B, C, K)
        means = diags + means
        if self.mask:
            means = util.flip(means)
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
        if self.mask:
            indices = util.flip(indices)
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k * (2 ** rank + self.gadditional + self.nadditional)
        assert indices.size() == (batch, context, num_points, 2)
        densities = sparse.densities(indices_fl, means, sigmas).clone()  # (B, C, P, self.k)
        duplicates = util.nduplicates(indices).to(torch.bool)  # (B, C, P) boolean mask of duplicates all-but-one
        densities[duplicates, :] = 0  # Removes all duplicates

        # Normalize densities across all K probability distributions by summing
        densities = densities / densities.sum(dim=2, keepdim=True)

        weights = values[:, :, None, :].expand_as(densities)
        weights = weights * densities
        weights = weights.sum(dim=3)  # I don't get this at all (sum out the MVNs)
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

        batch2, np, _ = indices.shape
        batch_is = torch.arange(batch2, dtype=torch.long, device=util.d(x))[None, :].expand(np, -1).t().reshape(-1)
        indices2 = torch.cat([batch_is[:, None], indices.view(-1, 2)], dim=-1)
        dot = util.calc_vals(Q, K.transpose(-2, -1), indices2).view(batch2, -1)
        dot = weights * dot
        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=V)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads * emb)
        return self.unify(out)


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

        params = self.to_param(x)  # (B, C, k*2) k means and sigmas for each point (1 degree of freedom)

        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(context_len, device=util.d(x), dtype=torch.float)
        diags = util.inv(diags, mx=context_len)
        diags = diags[None, :, None, None].expand(batch_size, -1, self.k, 1)  # (B, C, K, 1)

        means = params[:, :, :self.k].view(batch_size, -1, self.k, 1)  # Single mean for current point  (B, C, K, 1)
        sigmas = params[:, :, self.k:].view(batch_size, -1, self.k)  # (B, C, K)
        values = self.mvalues[None, None, :].expand(batch_size, context_len, -1)  # Expand to all points (B, C, K)

        # means = diags - torch.nn.functional.softplus(means)
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
        if self.mask:
            indices = util.flip(indices)
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

        batch2, np, _ = indices.shape
        batch_is = torch.arange(batch2, dtype=torch.long, device=util.d(x))[None, :].expand(np, -1).t().reshape(-1)
        indices2 = torch.cat([batch_is[:, None], indices.view(-1, 2)], dim=-1)
        dot = util.calc_vals(Q, K.transpose(-2, -1), indices2).view(batch2, -1)
        # dot = weights * dot
        dot = sparse.logsoftmax(indices, weights * dot, (context, context)).exp()
        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=V)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads * emb)
        return self.unify(out)


# TAKEN FROM https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.0, mask=True, **kwargs):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, **kwargs) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class NativeAttention(nn.Module):
    def __init__(self, num_heads, emb, context, mask, **kwargs):
        super().__init__()
        self.mask = mask
        self.native_attention = nn.MultiheadAttention(emb, num_heads)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.mask:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(None, x.size(1)).to(util.d(x))
            out, _ = self.native_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1), attn_mask=mask)
        else:
            out, _ = self.native_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        return out.transpose(0, 1)


class AlphaEntmax(nn.Module):

    def __init__(self, num_heads, emb, context, mask, alpha=1.5, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.emb = emb
        self.context = context
        self.mask = mask
        self.alpha = torch.tensor(alpha, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)

        self.to_keys = nn.Linear(emb, emb*num_heads)
        self.to_queries = nn.Linear(emb, emb * num_heads)
        self.to_values = nn.Linear(emb, emb * num_heads)

        self.unify_heads = nn.Linear(emb*num_heads, emb)

    def forward(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.to_keys(x).view(b, c, self.num_heads, e)
        q = self.to_queries(x).view(b, c, self.num_heads, e)
        v = self.to_values(x).view(b, c, self.num_heads, e)
        rank = k.size(-1)
        dot = (q.transpose(-2, -1) @ k) / (rank ** 0.5)  # scaled dot-product attention
        if self.mask:
            triu = torch.triu(torch.ones(dot.size(), device=util.d(x)))
            dot = torch.masked_fill(dot, triu, -float('inf'))
        dot = entmax.entmax_bisect(dot, self.alpha)
        res = (dot @ v.transpose(-2, -1)).transpose(-2, -1).reshape(b, c, self.num_heads*e)
        return self.unify_heads(res)
