"""Forward and backward parity for the channel-major triangle contraction."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esmjfold2.primitives import LayerNorm, Linear
from esmjfold2.triangle import TriangleMultiplicativeBlock


def make_block(*, channels=16, latent=8, flow="outgoing", with_bias=False):
    keys = iter(jax.random.split(jax.random.key(19), 3))

    def linear(out, inp):
        weight = jax.random.normal(next(keys), (out, inp)) * 0.1
        return Linear(weight=weight, bias=jnp.zeros((out,)) if with_bias else None)

    return TriangleMultiplicativeBlock(
        norm_start=LayerNorm(
            weight=1 + jnp.arange(channels) / channels * 0.1,
            bias=jnp.arange(channels) / channels * 0.01,
        ),
        norm_mix=LayerNorm(
            weight=1 + jnp.arange(latent) / latent * 0.1,
            bias=jnp.arange(latent) / latent * 0.01,
        ),
        proj_bundle=linear(4 * latent, channels),
        proj_emit=linear(channels, latent),
        proj_gate=linear(channels, channels),
        flow=flow,
        input_channels=channels,
        latent_channels=latent,
    )


def reference(block, pair_grid, mask=None):
    """Original channel-last implementation, kept here for parity checks."""
    if mask is None:
        mask = jnp.ones(pair_grid.shape[:-1], dtype=pair_grid.dtype)
    normalized = block.norm_start(pair_grid)
    signal, gate_logits = jnp.split(block.proj_bundle(normalized), 2, axis=-1)
    routed = signal * jax.nn.sigmoid(gate_logits) * mask[..., None]
    left, right = jnp.split(routed, 2, axis=-1)
    left, right = left.astype(jnp.float32), right.astype(jnp.float32)
    if block.flow == "outgoing":
        contracted = jnp.einsum("bikd,bjkd->bijd", left, right)
    else:
        contracted = jnp.einsum("bkid,bkjd->bijd", left, right)
    contracted = contracted.astype(pair_grid.dtype)
    mixed = block.proj_emit(block.norm_mix(contracted))
    return mixed * jax.nn.sigmoid(block.proj_gate(normalized))


@pytest.mark.parametrize("flow", ["outgoing", "incoming"])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_triangle_channel_major_forward_and_gradient(flow, with_mask, batch, with_bias):
    block = make_block(flow=flow, with_bias=with_bias)
    pair = jax.random.normal(jax.random.key(20), (batch, 7, 7, 16))
    mask = (
        (jnp.arange(batch)[:, None, None]
         + jnp.arange(7)[None, :, None]
         + jnp.arange(7)[None, None, :]) % 3 != 0
        if with_mask else None
    )
    cotangent = jax.random.normal(jax.random.key(21), pair.shape)

    def evaluate(fn):
        return eqx.filter_jit(jax.value_and_grad(
            lambda x: jnp.sum(fn(block, x, mask) * cotangent)
        ))(pair)

    old_value, old_grad = evaluate(reference)
    new_value, new_grad = evaluate(lambda model, x, m: model(x, m))
    np.testing.assert_allclose(new_value, old_value, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(new_grad, old_grad, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        eqx.filter_jit(block)(pair, mask), eqx.filter_jit(reference)(block, pair, mask),
        rtol=3e-5, atol=3e-5,
    )


@pytest.mark.parametrize("flow", ["outgoing", "incoming"])
def test_triangle_channel_major_model_gradient_at_real_width(flow):
    block = make_block(channels=256, latent=256, flow=flow, with_bias=False)
    pair = jax.random.normal(jax.random.key(22), (1, 16, 16, 256)) * 0.2
    cotangent = jax.random.normal(jax.random.key(23), pair.shape)

    def evaluate(fn):
        return eqx.filter_jit(eqx.filter_value_and_grad(
            lambda model: jnp.sum(fn(model, pair, None) * cotangent)
        ))(block)

    old_value, old_grad = evaluate(reference)
    new_value, new_grad = evaluate(lambda model, x, m: model(x, m))
    np.testing.assert_allclose(new_value, old_value, rtol=3e-5, atol=3e-5)
    for old, new in zip(jax.tree.leaves(old_grad), jax.tree.leaves(new_grad)):
        np.testing.assert_allclose(new, old, rtol=3e-5, atol=3e-5)
