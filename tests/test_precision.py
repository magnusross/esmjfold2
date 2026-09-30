"""BF16 storage, stable reductions, recurrence and design-gradient checks."""

import equinox as eqx
import esmjfold2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.models.esmc.configuration_esmc import ESMCConfig
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
from transformers.models.esmfold2.protein_utils import prepare_protein_features

from esmjfold2.primitives import LayerNorm, Linear, RMSNorm
from test_conversion import config  # noqa: F401 (shared pytest fixture)
from test_triangle_layout import make_block


def assert_similar(actual, expected, *, relative_error=0.04, cosine=0.995):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    assert np.isfinite(a).all()
    assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8) < relative_error
    if np.linalg.norm(a) * np.linalg.norm(b) > 1e-12:
        assert np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > cosine


def test_conversion_preserves_bf16(tmp_path):
    source = torch.tensor([1.0, 1.0078125, -3.25], dtype=torch.bfloat16)
    converted = esmjfold2.from_torch(source)
    assert converted.dtype == jnp.bfloat16
    np.testing.assert_array_equal(converted.astype(jnp.float32), source.float().numpy())
    esmjfold2.save_model(converted, tmp_path / "bf16")
    restored = esmjfold2.load_model(tmp_path / "bf16")
    assert restored.dtype == jnp.bfloat16
    np.testing.assert_array_equal(restored, converted)


@pytest.mark.parametrize("norm", [LayerNorm(), RMSNorm()])
def test_low_precision_norm_uses_float32_reductions(norm):
    # Squaring these in FP16 overflows. BF16 reductions also lose accuracy.
    for dtype in (jnp.float16, jnp.bfloat16):
        x = jnp.asarray([[512, 1024, -1024, 256]], dtype=dtype)
        actual = eqx.filter_jit(norm)(x)
        expected = norm(x.astype(jnp.float32)).astype(dtype)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)


def test_linear_compute_follows_weight_dtype():
    linear = Linear(jnp.ones((3, 4), dtype=jnp.bfloat16))
    x = jnp.ones((2, 4), dtype=jnp.float32)
    assert eqx.filter_jit(linear)(x).dtype == jnp.bfloat16
    assert jax.grad(lambda a: linear(a).astype(jnp.float32).sum())(x).dtype == jnp.float32


@pytest.mark.parametrize("flow", ["outgoing", "incoming"])
def test_bf16_triangle_input_gradient(flow):
    original = make_block(flow=flow)
    mixed = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, original)
    pair = jax.random.normal(jax.random.key(19), (1, 7, 7, 16))
    mask = (jnp.arange(7)[None, :, None] != jnp.arange(7)[None, None, :]).astype(jnp.float32)
    cotangent = jax.random.normal(jax.random.key(20), pair.shape)

    @eqx.filter_jit
    def run(block, x):
        return jax.value_and_grad(lambda p: (block(p, mask).astype(jnp.float32) * cotangent).sum())(x)

    reference = run(original, pair)
    result = run(mixed, pair.astype(jnp.bfloat16))
    for actual, expected in zip(result, reference):
        assert_similar(actual, expected)


def test_esmc_soft_input_gradient_and_hidden_dtype():
    torch.manual_seed(4)
    original = esmjfold2.from_torch(ESMCForMaskedLM(ESMCConfig(d_model=32, n_heads=4, n_layers=5)).eval())
    mixed = esmjfold2.mixed_precision(original)
    x = jax.random.normal(jax.random.key(11), (1, 9, 32))
    sequence_id = jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1, -1]])

    @eqx.filter_jit
    def run(m, x):
        def objective(x):
            last, _ = m.esmc.transformer(x, sequence_id, collect_hidden_states=False)
            logits = m.lm_head(last).astype(jnp.float32)
            return -jax.nn.log_softmax(logits)[0, 2, 5]
        return jax.value_and_grad(objective)(x)

    expected = run(original, x)
    actual = run(mixed, x)
    for a, b in zip(actual, expected):
        assert_similar(a, b)
    last, hidden = eqx.filter_jit(mixed.esmc.transformer)(x, sequence_id)
    assert last.dtype == hidden.dtype == jnp.bfloat16
    fp32_bytes = sum(v.nbytes for v in jax.tree.leaves(original) if eqx.is_array(v))
    bf16_bytes = sum(v.nbytes for v in jax.tree.leaves(mixed) if eqx.is_array(v))
    assert bf16_bytes < fp32_bytes * 0.55


@pytest.mark.parametrize("model_class", [ESMFold2Model, ESMFold2ExperimentalModel])
def test_mixed_model_recycling_design_gradient_and_prediction(config, model_class):  # noqa: F811
    torch.manual_seed(6)
    config.folding_trunk.n_layers = 5  # checkpoint-group boundary and tail
    original = esmjfold2.from_torch(model_class(config).eval())
    mixed = esmjfold2.mixed_precision(original)
    assert mixed.structure_head.diffusion_module.conditioning.z_proj.weight.dtype == jnp.float32
    assert mixed.distogram_head.weight.dtype == jnp.float32
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    pssm = jax.nn.one_hot(features.res_type, 33).astype(jnp.float32) * 0.9 + 0.1 / 33
    lm = jax.random.normal(jax.random.key(5), (1, 3, 3, 32))

    @eqx.filter_jit
    def run(m, p):
        def loss(p):
            f = eqx.tree_at(lambda f: f.res_type, features, p)
            ctx = m._prepare_embeddings(f)
            z = m._run_trunk(ctx, m.language_model(lm), jax.random.key(6), num_loops=1)
            return -jax.nn.log_softmax(m._compute_distogram(z))[0, 0, 2, 3]
        return jax.value_and_grad(loss)(p)

    expected, actual = run(original, pssm), run(mixed, pssm)
    for a, b in zip(actual, expected):
        assert_similar(a, b, relative_error=0.08, cosine=0.99)
    ctx = eqx.filter_jit(mixed._prepare_embeddings)(features)
    assert ctx.z_init.dtype == jnp.bfloat16
    output = eqx.filter_jit(mixed)(
        features, lm_hidden_states=lm, key=jax.random.key(7), num_loops=1, num_sampling_steps=2
    )
    assert output.sample_atom_coords.dtype == jnp.float32
    assert output.distogram_logits.dtype == jnp.float32
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
