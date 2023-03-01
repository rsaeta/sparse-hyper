"""
In this file we will explore using the adaptive sparsity for next-bit-prediction
"""

import torch
from torch import nn
from _context import sparse
from sparse import util


class SparseSelfAttention(nn.Module):
    """Do I need emb? Not sure"""
    def __init__(self, emb, context_len, k, hidden, n_heads=4, gadditional=2, nadditional=2):
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
        self.unify = nn.Linear(emb*n_heads, emb)
        self.register_buffer('mvalues', torch.ones((context_len, )))
        self.to_param = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*k)  # One mean and one sigma
        )

    def hyper(self, x):
        batch_size, context_len, emb = x.size()  # batch, context_len, embed_size
        assert context_len == self.context_len, f'Expected contextlen equal to {self.context_len}. Got {context_len}'
        assert emb == self.emb, f'Expected embedded equal to {self.emb}. Got {emb}'

        params = self.to_param(x)
        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(context_len, dtype=torch.float)
        diags = diags[None, :, None, None].expand(batch_size, context_len, self.k, 2)

        means = params[:, :, :self.k].view(batch_size, -1, self.k, 1)
        sigmas = params[:, :, self.k:].view(batch_size, -1, self.k)
        values = self.mvalues[None, :, None].expand(batch_size, -1, self.k)

        means = diags - torch.nn.functional.softplus(means)
        means = sparse.transform_means(means, (context_len, ))
        sigmas = sparse.transform_sigmas(sigmas, (context_len, ))
        return means, sigmas, values

    def forward(self, x):
        means, sigmas, values = self.hyper(x)
        batch, context, emb = x.size()
        rank = means.size(-1)
        indices = sparse.ngenerate(means,
                                   self.gadditional,
                                   self.nadditional,
                                   rng=(context, ),
                                   relative_range=(2, 2))
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k*(2**rank+self.gadditional+self.nadditional)

        assert indices.size() == (batch, context, num_points, 2)
        densities = sparse.densities(indices_fl, means, sigmas).clone()
        duplicates = util.nduplicates(indices).to(torch.bool)
        densities[duplicates, :] = 0
        densities = densities / densities.sum(dim=2, keepdim=True)
        weights = self.mvalues[None, :, None, None].expand_as(densities)
        weights = weights * densities
        weights = weights.sum(dim=3)  # I don't get this at all (sum out the MVNs)

        # I don't get what this is doing either
        # out = torch.arange(context)[None, :, None, None].expand(batch, -1, num_points, 1)
        # indices = torch.cat([out, indices], dim=3)

        # Here we expand the indicies for each head in this transformer block
        indices = indices[:, None, :, :, :].expand(-1, self.n_heads, -1, -1, -1)\
            .contiguous()\
            .view(batch*self.n_heads, context*num_points, -1)

        weights = weights[:, None, :, :].expand(-1, self.n_heads, -1, -1)\
            .contiguous()\
            .view(batch*self.n_heads, context*num_points)

        # Perform key, query, value transformation
        keys = self.to_keys(x).view(batch, context, self.n_heads, emb)
        queries = self.to_queries(x).view(batch, context, self.n_heads, emb)
        values = self.to_values(x).view(batch, context, self.n_heads, emb)

        # Because the KQV tensors have head dimension, we need to fold them back to single
        keys = keys.transpose(1, 2).contiguous().view(batch * self.n_heads, context, emb)
        queries = queries.transpose(1, 2).contiguous().view(batch*self.n_heads, context, emb)
        values = values.transpose(1, 2).contiguous().view(batch*self.n_heads, context, emb)

        queries = queries / (emb ** (1/4))  # Normalize along the embedding dimension
        keys = keys / (emb ** (1/4))

        indices_flattened = indices.view(batch*self.n_heads*context*num_points, -1)
        ar = torch.arange(batch*self.n_heads, dtype=torch.long)[:, None].expand(batch*self.n_heads, context*num_points)\
            .contiguous()\
            .view(batch*self.n_heads*context*num_points)

        squeries = queries[ar, indices_flattened[:, 0], :]
        skeys = keys[ar, indices_flattened[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(batch*self.n_heads, context*num_points)
        dot = sparse.logsoftmax(indices, weights*dot, (context, context)).exp()

        out = sparse.batchmm(indices, dot, size=(context, context), xmatrix=values)
        out = out.transpose(1, 2).contiguous().view(batch, context, self.n_heads*emb)
        breakpoint()
        return self.unify(out)


class TransformerBlock(nn.Module):

    def __init__(self, context, emb, k, heads, ff_hidden_mult=4, dropout=0.0, type='dense', ones=True, **kwargs):
        super().__init__()
        self.attend = SparseSelfAttention(emb, context, k, ff_hidden_mult, heads, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult*emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult*emb, emb)
        )

    def forward(self, x):
        attended = self.attend(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)

        ff = self.ff(x)
        x = self.norm2(ff+x)
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, n_blocks, context_len, emb, *args, **kwargs):
        super().__init__()
        self.context_len = context_len
        # Here we add 1 to emb because an additional coord is used for positional embeddings but only added
        # here in the root Transformer
        t_blocks = [TransformerBlock(context_len, emb+1, *args, **kwargs) for _ in range(n_blocks)]
        self.t_blocks = nn.Sequential(*t_blocks)
        self.to_prob = nn.Linear(1, 1)

    def pos_embed(self, x):
        b, c, e = x.size()
        positions = torch.arange(self.context_len, dtype=torch.float) / self.context_len
        positions = positions[None, :, None].expand(b, c, 1)
        return torch.cat([x, positions], dim=2)

    def forward(self, x):
        x = self.pos_embed(x)
        x = self.t_blocks(x)
        x = self.to_prob(x)
        return torch.nn.functional.log_softmax(x, dim=2)


def get_answers(batch, nums):
    indices = bin2dec(batch[:, -nums:], nums).float()
    return torch.index_select(batch, 1, indices).diag()


def train(model, iters=10, batch_size=1, context_len=100, nums_to_decide=6):
    for i in range(iters):
        batch = torch.rand((batch_size, context_len)).round()[:, :, None]
        y_hat = model(batch)
        y = get_answers(batch, nums_to_decide)[:, None]
        print(y - y_hat)


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def get_model(*args):
    return Transformer(*args)


def main():
    n_blocks = 4
    context = 200
    emb = 1
    k = 1
    n_heads = 4
    model = get_model(n_blocks, context, emb, k, n_heads)
    train(model, context_len=context)


if __name__ == '__main__':
    main()