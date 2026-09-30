# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""FoldingTrunk + PairUpdateBlock + Transition + PairTransition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .primitives import LayerNorm
from .swiglu import SwiGLU
from .triangle import TriangleMultiplicativeUpdate


class Transition(AbstractFromTorch):
    """LN -> SwiGLU residual."""

    norm: LayerNorm
    ffn: SwiGLU

    def __call__(self, x):
        # Pair transitions act independently on each (i, j) row. Keep the
        # expanded SwiGLU activations to a bounded token-row band, including
        # during backward, rather than materialising them for the whole pair.
        chunk_rows = 128
        if x.ndim != 4 or x.shape[1] <= chunk_rows:
            return x + self.ffn(self.norm(x))

        batch, n_rows, n_cols, channels = x.shape
        pad_rows = (-n_rows) % chunk_rows
        padded = jnp.pad(x, ((0, 0), (0, pad_rows), (0, 0), (0, 0)))
        n_chunks = (n_rows + pad_rows) // chunk_rows
        chunks = jnp.moveaxis(
            padded.reshape(batch, n_chunks, chunk_rows, n_cols, channels), 1, 0
        )

        @jax.checkpoint
        def chunk_forward(chunk):
            return chunk + self.ffn(self.norm(chunk))

        out = jax.lax.map(chunk_forward, chunks)
        out = jnp.moveaxis(out, 0, 1).reshape(batch, n_rows + pad_rows, n_cols, channels)
        return out[:, :n_rows]


class PairTransition(AbstractFromTorch):
    """Same structure as Transition but without residual (used in MSA encoder)."""

    norm: LayerNorm
    ffn: SwiGLU

    def __call__(self, x):
        return self.ffn(self.norm(x))


class PairUpdateBlock(AbstractFromTorch):
    """One folding-trunk block: trimul-out, trimul-in, pair transition residual."""

    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    pair_transition: Transition

    def __call__(self, pair, pair_attention_mask=None):
        # row_drop has r=0 in inference → identity residual sum.
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class FoldingTrunk(eqx.Module):
    """Stack of identical PairUpdateBlocks → scan.

    Stores (block_params, block_static) where block_params has arrays stacked
    along the leading dim. Built via from_torch.
    """

    block_params: PairUpdateBlock
    block_static: PairUpdateBlock

    def __call__(self, pair, pair_attention_mask=None):
        @jax.checkpoint
        def body(p, params):
            block = eqx.combine(self.block_static, params)
            return block(p, pair_attention_mask=pair_attention_mask), None

        # A scan's transpose retains its carry at every block even when the
        # body is checkpointed. Checkpoint groups of blocks so backward only
        # retains the carry at group boundaries, then recomputes each group.
        n_blocks = jax.tree.leaves(self.block_params)[0].shape[0]
        group_size = 4
        if n_blocks <= group_size:
            return jax.lax.scan(body, pair, self.block_params)[0]

        n_groups = n_blocks // group_size
        grouped_count = n_groups * group_size
        grouped = jax.tree.map(
            lambda x: x[:grouped_count].reshape((n_groups, group_size) + x.shape[1:]),
            self.block_params,
        )

        @jax.checkpoint
        def group_body(p, params):
            p, _ = jax.lax.scan(body, p, params)
            return p, None

        pair, _ = jax.lax.scan(group_body, pair, grouped)
        if grouped_count < n_blocks:
            tail = jax.tree.map(lambda x: x[grouped_count:], self.block_params)
            pair, _ = jax.lax.scan(body, pair, tail)
        return pair

    @classmethod
    def from_torch(cls, m):
        blocks = [from_torch(b) for b in m.blocks]
        if not blocks:
            raise ValueError("Empty FoldingTrunk.blocks")
        # Partition arrays/static using the first block as the static skeleton.
        _, static = eqx.partition(blocks[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, 0),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        return cls(block_params=stacked, block_static=static)


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    # Transition / PairTransition both wrap a LN + SwiGLU. Tell apart by class.
    from_torch.register(common.Transition, Transition.from_torch)
    # PairTransition lives in the model module.
    from esm.models.esmfold2 import model
    from_torch.register(model.PairTransition, PairTransition.from_torch)
    from_torch.register(common.PairUpdateBlock, PairUpdateBlock.from_torch)
    from_torch.register(common.FoldingTrunk, FoldingTrunk.from_torch)
