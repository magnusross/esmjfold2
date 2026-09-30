"""Parity checks for chunked pair transitions, including backward passes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from esmjfold2.primitives import LayerNorm, Linear
from esmjfold2.swiglu import SwiGLU
from esmjfold2.trunk import Transition


def make_transition():
    first, second = jax.random.split(jax.random.key(31))
    return Transition(
        norm=LayerNorm(weight=jnp.ones(16), bias=jnp.zeros(16)),
        ffn=SwiGLU(
            w12=Linear(weight=jax.random.normal(first, (64, 16)) * 0.1),
            w3=Linear(weight=jax.random.normal(second, (16, 32)) * 0.1),
            hidden_features=32,
        ),
    )


def reference(model, pair):
    return pair + model.ffn(model.norm(pair))


@pytest.mark.parametrize("n_rows", [127, 133])
def test_transition_chunking_forward_and_gradients(n_rows):
    model = make_transition()
    pair = jax.random.normal(jax.random.key(32), (2, n_rows, 7, 16))
    cotangent = jax.random.normal(jax.random.key(33), pair.shape)

    def loss(fn, module, x):
        return jnp.sum(fn(module, x) * cotangent)

    def evaluate(fn):
        value, input_grad = jax.value_and_grad(lambda x: loss(fn, model, x))(pair)
        _, model_grad = eqx.filter_value_and_grad(lambda m: loss(fn, m, pair))(model)
        return value, input_grad, model_grad

    expected_value, expected_input_grad, expected_model_grad = evaluate(reference)
    value, input_grad, model_grad = evaluate(lambda module, x: module(x))
    np.testing.assert_allclose(value, expected_value, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(input_grad, expected_input_grad, rtol=3e-5, atol=3e-5)
    for actual, expected in zip(
        jax.tree.leaves(model_grad), jax.tree.leaves(expected_model_grad)
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
