import math
import inspect

from _context import sparse
from sparse import util

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np
from dataclasses import dataclass
from attention_layers import SparseSelfAttention


@dataclass
class SmallerBirdConfig:
    max_position_embeddings: int
    num_attention_heads: int
    hidden_size: int
    num_random_blocks: int
    block_size: int
    nadditional: int = 4
    gadditional: int = 4
    use_bias: bool = False

    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })


def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
    batch_size, seq_length = attention_mask.size()
    assert (
        seq_length % block_size == 0
    ), f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}."

    def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
        """
        Create 3D attention mask from a 2D tensor mask.
        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
        Returns:
            float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
            3*to_block_size].
        """
        exp_blocked_to_pad = torch.cat(
            [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
        )
        band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
        band_mask.unsqueeze_(1)
        return band_mask

    blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
    band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

    from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
    to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

    return blocked_encoder_mask, band_mask, from_mask, to_mask


class SmallerBirdSparseAttention(SparseSelfAttention):
    def __init__(self, config: SmallerBirdConfig, seed=None):
        super().__init__(config.hidden_size,
                         config.max_position_embeddings,
                         k=config.num_random_blocks,
                         gadditional=config.gadditional,
                         nadditional=config.nadditional,
                         mask=False)

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def hyper_to_weights(self, x, means, sigmas, values):
        batch, context, emb = x.size()  # (B, C, E)
        rank = means.size(-1)
        indices: Tensor = sparse.ngenerate(means,
                                           self.gadditional,
                                           self.nadditional,
                                           rng=(context,),
                                           relative_range=(2,),
                                           cuda='cuda' in util.d(x),
                                           train=self.train)  # (B, C, P, 1)
        assert ((indices < 0).sum().item() == 0) and ((indices >= context).sum().item() == 0), \
            f'Found some indices out of bounds: indices < 0: {(indices < 0).sum().item()}; ' \
            f'indices >= {context}: {(indices >= context).sum().item()}'
        indices_fl = indices.float()
        # For each point (self.k), we expect to sample the 2**rank closest points from the first set of sampling,
        # then self.gadditional globally-sampled indices, and self.nadditional neighborhood-sampled indices.
        num_points = self.k * (2 ** rank + self.gadditional + self.nadditional)
        assert indices.size() == (batch, context, num_points, 1), f'Expected size {(batch, context, num_points, 1)}. ' \
                                                                  f'Got {indices.size()}'
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
        # out = torch.arange(context, device=util.d(x))[None, :, None, None].expand(batch, -1, num_points, 1)
        # indices = torch.cat([out, indices], dim=3)
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
        
        return weights, indices

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
        # Useless arguments: `attention_mask`, `head_mask`, `encoder_hidden_states`, `encoder_attention_mask`, `past_key_value`
        # Currently this `class` can't be used in decoder.
        attention_mask = (attention_mask[:,0,:]).long()
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        assert from_seq_length % from_block_size == 0, "Query sided sequence length must be multiple of block size"
        assert to_seq_length % to_block_size == 0, "Key/Value sided sequence length must be multiple of block size"

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        means, sigmas, values = self.hyper(hidden_states) # Getting adaptive sparse locations
        weights, indices = self.hyper_to_weights(hidden_states, means, sigmas, values)

        indreshape = indices.squeeze().reshape(hidden_states.size(0), self.n_heads, hidden_states.size(1), -1)[:,:,1:-1,:]
        weightsreshape = weights.reshape(hidden_states.size(0), self.n_heads, hidden_states.size(1), -1)

        blocked_encoder_mask, band_mask, from_mask, to_mask = create_masks_for_block_sparse_attn(attention_mask, 1)
        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            blocked_encoder_mask,
            blocked_encoder_mask,
            self.num_attention_heads,
            indreshape,
            weightsreshape,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs[0]

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication """
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication with transpose """
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        num_attention_heads,
        adaptive_attn_inds,
        adaptive_attn_weights,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_length,
        to_seq_length,
        seed=None,
        plan_from_length=None,
        plan_num_rand_blocks=None,
        output_attentions=None,
    ):

        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        # Define shorthands
        h = num_attention_heads
        r = adaptive_attn_inds.size(-1)
        rsqrt_d = 1 / math.sqrt(attention_head_size)
        b = batch_size
        m = from_seq_length
        n = to_seq_length
        wm = from_block_size
        wn = to_block_size

        # generate random attention and corresponding masks
        np.random.seed(seed)
        # if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper
        #     rand_attn = [
        #         self._bigbird_block_rand_mask(self.max_seqlen, self.max_seqlen, wm, wn, r, last_idx=1024)[
        #             : (from_seq_length // wm - 2)
        #         ]
        #         for _ in range(h)
        #     ]
        # else:
        #     if plan_from_length is None:
        #         plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(from_seq_length, wm, r)

        #     rand_attn = self._bigbird_block_rand_mask_with_head(
        #         from_seq_length=from_seq_length,
        #         to_seq_length=to_seq_length,
        #         from_block_size=wm,
        #         to_block_size=wn,
        #         num_heads=h,
        #         plan_from_length=plan_from_length,
        #         plan_num_rand_blocks=plan_num_rand_blocks,
        #     )

        # rand_attn = np.stack(rand_attn, axis=0)
        # rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
        # rand_attn.unsqueeze_(0)
        # rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
        rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, adaptive_attn_inds, h, r, b, m, wm)

        blocked_query_matrix = query_layer.view(b, h, m // wm, wm, -1)
        blocked_key_matrix = key_layer.view(b, h, n // wn, wn, -1)
        blocked_value_matrix = value_layer.view(b, h, n // wn, wn, -1)

        # preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, adaptive_attn_inds)
        gathered_key = gathered_key.view(b, h, n // wn - 2, r * wn, -1)  # [b, h, n//wn-2, r, wn, -1]
        gathered_value = self.torch_gather_b2(blocked_value_matrix, adaptive_attn_inds)
        gathered_value = gathered_value.view(b, h, n // wn - 2, r * wn, -1)  # [b, h, n//wn-2, r, wn, -1]

        # 1st block is global q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * -10000.0
        first_attn_weights = F.softmax(first_product, dim=-1)  # [b, h, wm, n]

        # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # q[1] x (sliding_keys, random_keys, global_keys)
        second_key_mat = torch.cat(
            [
                # blocked_key_matrix[:, :, 0],
                # blocked_key_matrix[:, :, 1],
                # blocked_key_matrix[:, :, 2],
                # blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]
        second_value_mat = torch.cat(
            [
                # blocked_value_matrix[:, :, 0],
                # blocked_value_matrix[:, :, 1],
                # blocked_value_matrix[:, :, 2],
                # blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]

        # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * wn],
                to_mask[:, :, :, -wn:],
                torch.ones(b, 1, 1, r * wn, device=second_value_mat.device, dtype=second_value_mat.dtype),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                torch.ones(b, h, wm, 4 * wn, device=second_value_mat.device, dtype=second_value_mat.dtype),
                rand_mask[:, :, 0],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad))[:,:,:,2:-2] * -10000.0
        second_attn_weights = F.softmax(second_product, dim=-1)  # [b , h, wm, (4+r)*wn]

        # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # q[-2:2] x (sliding_keys, random_keys, global_keys)

        # initialize q,k,v->q,k,v[-2:2]
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [b, h, m//wm-4, 3*wn, -1]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [b, h, m//wm-4, 3*wn, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        # ==> [b, h, m//wm-4, wm, 3*wn]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        adaptive_rand_weights = adaptive_attn_weights[:, :, 2:-2, None, :]
        #     ==> [b, h, m//wm-4, wm, r*wn]
        rand_band_product = rand_band_product * rsqrt_d * adaptive_rand_weights

        # 1st block is global
        # first_band_product = torch.einsum(
        #     "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        # )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        # first_band_product = first_band_product * rsqrt_d

        # last block is global
        # last_band_product = torch.einsum(
        #     "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        # )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        # last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * -10000.0
        # first_band_product += (1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
        # last_band_product += (1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [inner_band_product, rand_band_product], dim=-1
        )  # [b, h, m//wm-4, wm, (5+r)*wn]

        # safely doing softmax since attention matrix is completed
        attn_weights = F.softmax(band_product, dim=-1)  # [b, h, m//wm-4, wm, (5+r)*wn]

        # contibution of sliding keys
        # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, wn : 4 * wn], exp_blocked_value_matrix, ndim=5)
        #     ==> [b, h, m//wm-4, wm, -1]

        # adding contribution of random keys
        # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
        context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 4 * wn : -wn], gathered_value[:, :, 1:-1, 1:-1], ndim=5)
        #     ==> [b, h, m//wm-4, wm, -1]

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :wn], blocked_value_matrix[:, :, 0]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -wn:], blocked_value_matrix[:, :, -1]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

        # q[-2] x (sliding_keys, random_keys, global_keys)

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]

        # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :wn],
                to_mask[:, :, :, -3 * wn :],
                torch.ones([b, 1, 1, r * wn], device=context_layer.device, dtype=context_layer.dtype),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                torch.ones([b, h, wm, 4 * wn], device=context_layer.device, dtype=context_layer.dtype),
                rand_mask[:, :, -1],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
        second_last_attn_weights = F.softmax(second_last_product, dim=-1)  # [b, h, wm, (4+r)*wn]

        # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)

        # last block is global q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * -10000.0
        last_attn_weights = F.softmax(last_product, dim=-1)  # [b, h, wm, n]

        # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)
        context_layer = torch.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],   
            dim=2,
        )
        context_layer = context_layer.view((b, h, m, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)

        if output_attentions:
            # TODO(PVP): need to verify if below code is correct
            attention_probs = torch.zeros(b, h, m, n, dtype=torch.float, device=context_layer.device)

            # corresponding to `first_context_layer`
            attention_probs[:, :, :wm, :] = first_attn_weights

            # corresponding to `second_context_layer`
            attention_probs[:, :, wm : 2 * wm, : 3 * wn] = second_attn_weights[:, :, :, : 3 * wn]
            attention_probs[:, :, wm : 2 * wm, -wn:] = second_attn_weights[:, :, :, 3 * wn : 4 * wn]
            for p1, i1, w1 in zip(range(b), adaptive_attn_inds, second_attn_weights):
                for p2, i2, w2 in zip(range(h), i1, w1):
                    attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, 1, :, i2[0]] = w2[:, 4 * wn :].view(
                        wm, r, wn
                    )

            # corresponding to `context_layer`
            for q_idx in range(m // wm - 4):
                slice = attention_probs.view(b, h, m // wm, wm, n // wn, wn)[:, :, 2:-2, :, 1:-1, :]
                slice[:, :, q_idx, :, q_idx : q_idx + 3, :] = attn_weights[:, :, q_idx, :, wn : 4 * wn].view(
                    b, h, wm, 3, wn
                )  # inner_band_product
            attention_probs[:, :, 2 * wm : -2 * wm, :wn] = attn_weights[:, :, :, :, :wn].view(
                b, h, -1, wn
            )  # first_band_product
            attention_probs[:, :, 2 * wm : -2 * wm, -wn:] = attn_weights[:, :, :, :, -wn:].view(
                b, h, -1, wn
            )  # last_band_product
            for p1, i1, w1 in zip(range(b), adaptive_attn_inds, attn_weights):
                for p2, i2, w2 in zip(range(h), i1, w1):
                    for q_idx in range(1, len(i2) - 1):
                        attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, q_idx + 1, :, i2[q_idx]] = w2[
                            q_idx - 1, :, 4 * wn : -wn
                        ].view(wm, r, wn)

            # corresponding to `second_last_context_layer`
            attention_probs[:, :, -2 * wm : -wm, :wn] = second_last_attn_weights[:, :, :, :wn]
            attention_probs[:, :, -2 * wm : -wm, -3 * wn :] = second_last_attn_weights[:, :, :, wn : 4 * wn]
            for p1, i1, w1 in zip(range(b), adaptive_attn_inds, second_last_attn_weights):
                for p2, i2, w2 in zip(range(h), i1, w1):
                    attention_probs.view(b, h, m // wm, wm, n // wn, wn)[p1, p2, -2, :, i2[-1]] = w2[:, 4 * wn :].view(
                        wm, r, wn
                    )

            # corresponding to `last_context_layer`
            attention_probs[:, :, -wm:, :] = last_attn_weights

        else:
            attention_probs = None

        return context_layer, attention_probs
    
    @staticmethod
    def torch_gather_b2(params, indices):

        assert (
            params.shape[:2] == indices.shape[:2]
        ), f"Make sure that the first two dimensions of params and indices are identical, but they are params: {params.shape[:2]} vs. indices: {params.shape[:2]}"
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]

        indices_shift = (
            torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
            // num_indices_to_gather
            * num_indices_to_pick_from
        )


        flattened_indices = indices.contiguous().view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        out_flattened = flattened_params.index_select(0, flattened_indices)

        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        rand_attn,
        num_attention_heads,
        num_rand_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):

        # general plan
        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        assert from_seq_length in plan_from_length, "Error from sequence length not in plan!"

        # Total number of blocks in the mmask
        num_blocks = from_seq_length // from_block_size
        # Number of blocks per plan
        plan_block_length = np.array(plan_from_length) // from_block_size
        # till when to follow plan
        max_plan_idx = plan_from_length.index(from_seq_length)
        # Random Attention adjajency list
        rand_attn = [
            np.zeros((num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])), dtype=np.int32)
            for i in range(num_heads)
        ]

        # We will go iteratively over the plan blocks and pick random number of
        # Attention blocks from the legally allowed blocks
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                # set the row for all from_blocks starting from 0 to
                # plan_block_length[plan_idx-1]
                # column indx start fromm plan_block_length[plan_idx-1] and ends at
                # plan_block_length[plan_idx]
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]

            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )

        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]

        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):

        # list of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)


class BigBirdSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
