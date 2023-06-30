from typing import Tuple
import torch
from torch import nn, Tensor

from _context import sparse
from sparse import util

from spalbp.lib.attention.config import (
    AdaptiveSparseAttentionConfig,
    NonAdaptiveSparseAttentionConfig,
)


class _OneDimensionalSparseAttention(nn.Module):
    def __init__(
        self,
        emb: int,
        n_heads: int,
        *,
        head_size: int = 16,
        k: int = 4,
        gadditional: int = 1,
        nadditional: int = 4,
        remove_rand_on_eval: bool = True,
        bias_kv: bool = False,
    ):
        super().__init__()
        self.emb = emb
        self.n_heads = n_heads
        self.k = k
        self.gadditional = gadditional
        self.nadditional = nadditional
        self.head_size = head_size

        self.to_keys = nn.Linear(emb, head_size * n_heads, bias=bias_kv)
        self.to_queries = nn.Linear(emb, head_size * n_heads, bias=bias_kv)
        self.to_values = nn.Linear(emb, head_size * n_heads, bias=bias_kv)
        self.unify = nn.Linear(head_size * n_heads, emb)
        self.remove_rand_on_eval = remove_rand_on_eval

        self.register_buffer("mvalues", torch.ones((k,)))

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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        means, sigmas, values = self.hyper(
            x
        )  # (B, H, C, k, 1); (B, H, C, k, 1); (B, H, C, k)
        batch, context, emb = x.size()  # (B, C, E)
        rank = means.size(-1)
        gadditional = self.gadditional if self.training or not self.remove_rand_on_eval else 0
        nadditional = self.nadditional if self.training or not self.remove_rand_on_eval else 0
        indices: Tensor = sparse.ngenerate(
            means,
            # For evaluation, only get nearest
            gadditional,
            nadditional,
            rng=(context,),
            relative_range=(3,),
            cuda="cuda" in util.d(x),
        )  # (B, H, C, P, 1)
        if not ((indices < 0).sum().item() == 0) and (
            (indices >= context).sum().item() == 0
        ):
            print(
                f"Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; "
                f"indices >= {context}: {(indices >= context).sum().item()}"
            )
            breakpoint()
        assert ((indices < 0).sum().item() == 0) and (
            (indices >= context).sum().item() == 0
        ), (
            f"Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; "
            f"indices >= {context}: {(indices >= context).sum().item()}"
        )
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k * (
            2**rank
            + (
                (self.gadditional + self.nadditional)
                if self.training or not self.remove_rand_on_eval
                else 0
            )
        )
        assert indices.size() == (batch, self.n_heads, context, num_points, 1), (
            f"Expected size {(batch, context, num_points, 1)}. " f"Got {indices.size()}"
        )

        densities = sparse.densities(
            indices_fl, means, sigmas
        ).clone()  # (B, H, C, P, self.k)
        duplicates = util.nduplicates(indices).to(
            torch.bool
        )  # (B, C, P) boolean mask of duplicates all-but-one
        densities[duplicates, :] = 0  # Removes all duplicates
        # Normalize densities across all K probability distributions by summing
        densities = densities / (
            densities.sum(dim=2, keepdim=True) + 1e-8
        )  # (B, H, C, P, self.k)

        weights = values[:, :, :, None, :].expand_as(densities)  # (B, H, C, P, self.k)
        weights = weights * densities  # (B, H, C, P, self.k)
        weights = weights.sum(dim=4)  # (B, H, C, P)

        # Perform key, query, value transformation

        K = self.to_keys(x).view(batch, context, self.n_heads, self.head_size)
        Q = self.to_queries(x).view(batch, context, self.n_heads, self.head_size)
        V = self.to_values(x).view(batch, context, self.n_heads, self.head_size)

        K = K / (self.head_size ** (1 / 2))
        Q = Q / (self.head_size ** (1 / 2))

        flat_indices = indices.view(batch, self.n_heads, -1)  # [b, h, c * p]
        gathered_keys = self.batched_index_select(
            K.transpose(1, 2), flat_indices
        )  # [b, h, c * p, d]
        dots = torch.einsum(
            "bhcd,bhcqd -> bhcq",
            Q.transpose(1, 2),
            gathered_keys.view(batch, self.n_heads, context, num_points, -1),
        )  # [b, h, c, p]
        gathered_values = self.batched_index_select(
            V.transpose(1, 2), flat_indices
        )  # [b, h, c * p, d]
        gathered_values = gathered_values.view(
            batch, self.n_heads, context, num_points, -1
        )  # [b, h, c, p, d]
        new_weights = weights * dots  # Weigh attention scores by densities
        new_weights = new_weights.softmax(dim=-1)  # Normalize attention scores
        head_reps = gathered_values * new_weights.unsqueeze(
            -1
        )  # Weigh values by attention scores
        head_reps = head_reps.sum(dim=-2)  # Sum over weighted values [b, h, c, d]
        # united = head_reps.view(batch, context, -1)  # [b, c, h * d]
        united = (
            head_reps.transpose(1, 2).contiguous().view(batch, context, -1)
        )  # [b, c, h * d]
        new_out = self.unify(united)  # [b, c, e]

        attentions = torch.zeros(batch, self.n_heads, context, context, device=x.device)
        attentions.scatter_add_(dim=3, index=indices.squeeze(-1), src=new_weights)
        if output_attentions:
            return new_out, attentions
        return new_out


class NonadaptiveSparseAttention(_OneDimensionalSparseAttention):
    @classmethod
    def from_config(cls, config: NonAdaptiveSparseAttentionConfig):
        return cls(
            config.emb,
            config.context,
            k=config.k,
            n_heads=config.heads,
            gadditional=config.gadditional,
            nadditional=config.nadditional,
            sigma_scale=config.sigma_scale,
            transformation_method=config.transformation_method,
            means_init_method=config.means_init_method,
            remove_rand_on_eval=config.remove_rand_on_eval,
            bias_kv=config.bias_kv,
        )

    @staticmethod
    def _init_means(context: int, k: int, means_init_method: str):
        if means_init_method == "random":
            means = torch.rand((context, k, 1)) * 2 - 1
        elif means_init_method == "uniform":
            means = (
                torch.linspace(0, context - 1, k + 2)[None, 1:-1, None]
                .expand(context, k, 1)
                .clone()
            )
        else:
            raise ValueError(
                f"Unknown means initialization method: {means_init_method}"
            )
        return means

    def __init__(
        self,
        emb: int,
        context_len: int,
        *,
        k: int = 8,
        n_heads: int = 4,
        gadditional: int = 2,
        nadditional: int = 2,
        sigma_scale: float = 1.0,
        transformation_method: str = "sigmoid",
        means_init_method: str = "random",
        remove_rand_on_eval: bool = False,
        bias_kv: bool = False,
    ):
        super().__init__(
            emb,
            n_heads,
            k=k,
            gadditional=gadditional,
            nadditional=nadditional,
            remove_rand_on_eval=remove_rand_on_eval,
            bias_kv=bias_kv,
        )
        means = self._init_means(context_len, k, means_init_method)
        self.pmeans = torch.nn.Parameter(means)
        self.psigmas = torch.nn.Parameter(torch.rand((context_len, k)))
        # Non-learnabe
        self.register_buffer("pvalues", torch.ones(k))
        self.sigma_scale = sigma_scale
        self.transformation_method = transformation_method

    def hyper(self, x: torch.Tensor):
        b, c, e = x.size()
        k = self.k
        means = self.pmeans[None, None, :, :, :].expand(b, self.n_heads, c, k, 1)
        sigmas = self.psigmas[None, None, :, :].expand(b, self.n_heads, c, k)
        values = self.pvalues[None, None, None, :].expand(b, self.n_heads, c, k)

        means = sparse.transform_means(means, (c,), method=self.transformation_method)
        sigmas = sparse.transform_sigmas(sigmas, (c,)) * self.sigma_scale

        return means, sigmas, values


class UnknowingSparseAttention(_OneDimensionalSparseAttention):
    def __init__(
        self,
        emb: int,
        context_len: int,
        *,
        n_heads: int = 4,
        gadditional: int = 2,
        nadditional: int = 2,
        sigma_scale: float = 1.0,
        transformation_method: str = "modulo",
    ):
        k = 1
        super().__init__(
            emb, n_heads, k=k, gadditional=gadditional, nadditional=nadditional
        )

        self.pmeans = torch.rand((context_len, k, 1), device="cuda")
        self.pmeans[45, 0, 0] = 112.0

        self.psigmas = torch.nn.Parameter(torch.rand((context_len, k)))
        # Non-learnabe
        self.register_buffer("pvalues", torch.ones(k))
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
    @classmethod
    def from_config(cls, config: NonAdaptiveSparseAttentionConfig):
        return cls(
            config.emb,
            config.context,
            n_heads=config.heads,
            gadditional=config.gadditional,
            nadditional=config.nadditional,
            sigma_scale=config.sigma_scale,
            k=config.k,
            transformation_method=config.transformation_method,
        )

    def __init__(
        self,
        emb: int,
        context_len: int,
        *,
        n_heads: int = 4,
        gadditional: int = 2,
        nadditional: int = 2,
        sigma_scale: float = 1.0,
        k: int = 1,
        transformation_method: str = "modulo",
    ):
        super().__init__(
            emb, n_heads, k=k, gadditional=gadditional, nadditional=nadditional
        )
        answers = torch.remainder(
            torch.arange(0, context_len, device="cuda") + 70, context_len
        ).float()
        self.pmeans = torch.randint(
            0, context_len, (n_heads, context_len, k, 1), device="cuda"
        ).float()
        self.pmeans[0, :, 0, :] = answers[:, None]
        self.psigmas = torch.nn.Parameter(torch.rand((n_heads, context_len, k)))
        # Non-learnabe
        self.register_buffer("pvalues", torch.ones(k))
        self.sigma_scale = sigma_scale
        self.transformation_method = transformation_method

    def hyper(self, x: torch.Tensor):
        b, c, e = x.size()
        k, h = self.k, self.n_heads
        means = self.pmeans[None].expand(b, h, c, k, 1)
        sigmas = self.psigmas[None].expand(b, h, c, k)
        values = self.pvalues[None, None, None, :].expand(b, h, c, k)

        means = sparse.transform_means(means, (c,), method=self.transformation_method)
        sigmas = sparse.transform_sigmas(sigmas, (c,)) * self.sigma_scale
        return means, sigmas, values


class SparseSelfAttention(_OneDimensionalSparseAttention):
    @classmethod
    def from_config(cls, config: AdaptiveSparseAttentionConfig):
        return cls(
            config.emb,
            config.context,
            head_size=config.head_size,
            k=config.k,
            hidden=config.hyper_hidden_dim,
            n_heads=config.heads,
            gadditional=config.gadditional,
            nadditional=config.nadditional,
            remove_rand_on_eval=config.remove_rand_on_eval,
            bias_kv=config.bias_kv,
        )

    def __init__(
        self,
        emb: int,
        context_len: int,
        head_size: int,
        *,
        k: int = 8,
        hidden: int = 4,
        n_heads: int = 4,
        gadditional: int = 2,
        nadditional: int = 2,
        remove_rand_on_eval: bool = True,
        bias_kv: bool = False,
    ):
        super().__init__(
            emb,
            n_heads,
            head_size=head_size,
            k=k,
            gadditional=gadditional,
            nadditional=nadditional,
            remove_rand_on_eval=remove_rand_on_eval,
            bias_kv=bias_kv,
        )
        self.context_len = context_len
        self.to_param = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
#            nn.Linear(hidden, hidden),
#            nn.ReLU(),
            nn.Linear(hidden, 2 * k * n_heads),  # One mean and one sigma per head
#            nn.Sigmoid(),
        )

    def hyper(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, context_len, emb = x.size()
        assert (
            context_len == self.context_len
        ), f"Expected contextlen equal to {self.context_len}. Got {context_len}"
        assert emb == self.emb, f"Expected embedded equal to {self.emb}. Got {emb}"

        params = self.to_param(
            x
        )  # (B, C, h*k*2) k means and sigmas for each point (1 degree of freedom)

        # Generate the logits that correspond to the horizontal coordinate of the current word
        # diags = torch.arange(context_len, device=util.d(x), dtype=torch.float)
        # diags = util.inv(diags, mx=context_len)
        # diags = diags[None, None, :, None, None].expand(
        #    batch_size, self.n_heads, -1, self.k, 1
        # )  # (B, H, C, K, 1)
        means = (
            params[:, :, : self.n_heads * self.k, None]
            .view(batch_size, context_len, self.n_heads, self.k, 1)
            .permute(0, 2, 1, 3, 4)
        )  # Single mean for current point  (B, H, C, K, 1)
        sigmas = (
            params[:, :, self.n_heads * self.k :]
            .view(batch_size, context_len, self.n_heads, self.k)
            .permute(0, 2, 1, 3)
        )  # (B, H, C, K)
        values = self.mvalues[None, None, None, :].expand(
            batch_size, self.n_heads, context_len, -1
        )  # Expand to all points (B, C, K)

        # means = diags - torch.nn.functional.softplus(means)
        means = sparse.transform_means(means, (context_len,))
        sigmas = sparse.transform_sigmas(sigmas, (context_len,))

        return means, sigmas, values
