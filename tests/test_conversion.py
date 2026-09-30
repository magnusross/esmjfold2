"""Biohub Transformers model loading, conversion, and CPU numerical parity."""

import equinox as eqx
import esmjfold2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers.models.esmc.configuration_esmc import ESMCConfig
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM, ESMCModel
from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
from transformers.models.esmfold2.protein_utils import prepare_protein_features


@pytest.fixture
def config():
    config = ESMFold2Config(
        type="release",
        d_single=32,
        d_pair=16,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk={"n_layers": 1, "n_heads": 2},
        inputs={"atom_encoder": {"d_atom": 64, "n_heads": 2, "n_blocks": 1}},
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                "c_atom": 64,
                "atom_num_heads": 2,
                "atom_num_blocks": 1,
                "c_token": 32,
                "c_z": 16,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "folding_trunk": {"n_layers": 1, "n_heads": 2},
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
        },
        parcae={"coda_n_layers": 1},
        lm_encoder={"n_layers": 1},
        msa_encoder={
            "enabled": True,
            "d_msa": 32,
            "d_hidden": 8,
            "n_layers": 1,
            "n_heads_msa": 2,
            "msa_head_width": 8,
        },
        msa_encoder_overwrite=False,
    )
    return config


@pytest.mark.parametrize("model_class", [ESMCModel, ESMCForMaskedLM])
def test_esmc_parity(model_class):
    torch.manual_seed(4)
    model = model_class(ESMCConfig(d_model=32, n_heads=4, n_layers=2)).eval()
    converted = esmjfold2.from_torch(model)
    ids = np.array([[0, 5, 6, 7, 2]], dtype=np.int32)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids), output_hidden_states=True)
    output, hidden = eqx.filter_jit(converted)(jnp.asarray(ids), collect_hidden_states=True)
    reference = expected.logits if model_class is ESMCForMaskedLM else expected.last_hidden_state
    np.testing.assert_allclose(output, reference.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hidden, expected.hidden_states.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("model_class", [ESMFold2Model, ESMFold2ExperimentalModel])
def test_checkpoint_conversion_and_prediction(config, model_class, tmp_path):
    """Exercise the real loader, all converters, JIT inference and serialization."""
    torch.manual_seed(0)
    config.type = "experimental" if model_class is ESMFold2ExperimentalModel else "release"
    lm_path = tmp_path / "esmc"
    ESMCModel(ESMCConfig(d_model=32, n_heads=4, n_layers=2)).save_pretrained(lm_path)
    config.esmc_id = str(lm_path)
    config.save_pretrained(tmp_path)
    save_file(model_class(config).eval().state_dict(), tmp_path / "model.safetensors")
    model = ESMFold2Model.from_pretrained(tmp_path, dtype=torch.float32)
    model._esmc = model._esmc.to(dtype=torch.float32)
    assert isinstance(model, model_class)
    if model_class is ESMFold2Model:
        assert esmjfold2.from_torch(model).msa_encoder_overwrite is False
    converted = esmjfold2.from_torch(model)
    esmc = esmjfold2.from_torch(model._esmc)
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    lm = esmjfold2.compute_lm_hidden_states(
        esmc,
        raw["input_ids"],
        raw["asym_id"],
        raw["residue_index"],
        raw["mol_type"],
        raw["token_attention_mask"],
    )
    output = eqx.filter_jit(converted)(
        features, lm_hidden_states=lm, key=jax.random.key(0), num_loops=1, num_sampling_steps=2
    )
    assert output.sample_atom_coords.shape[-2:] == (raw["ref_pos"].shape[1], 3)
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
    esmjfold2.save_model(converted, tmp_path / "converted")
    restored = esmjfold2.load_model(tmp_path / "converted")
    assert eqx.tree_equal(converted, restored)


def test_folding_components_parity(config):
    """Compare deterministic modules with identical inputs, avoiding RNG differences."""
    torch.manual_seed(7)
    model = ESMFold2Model(config).eval()
    model.set_kernel_backend(None)
    converted = esmjfold2.from_torch(model)
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    ctx = converted._prepare_embeddings(features)

    def tensor(x):
        value = torch.from_numpy(np.array(x))
        return value.long() if value.dtype == torch.int32 else value

    def compare(jax_module, torch_module, **kwargs):
        with torch.no_grad():
            expected = torch_module(**{k: tensor(v) for k, v in kwargs.items()})
        actual = eqx.filter_jit(jax_module)(**kwargs)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)

    compare(
        converted.rel_pos,
        model.rel_pos,
        **{
            k: getattr(features, k)
            for k in ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")
        },
    )
    z = jax.random.normal(jax.random.key(1), (1, 3, 3, config.d_pair))
    compare(converted.folding_trunk, model.folding_trunk, pair=z, pair_attention_mask=ctx.pair_mask)

    x = jax.random.normal(jax.random.key(2), features.ref_pos.shape)
    args = {
        "x_noisy": x,
        "t_hat": jnp.array([1.5]),
        "ref_pos": features.ref_pos,
        "ref_charge": features.ref_charge,
        "ref_mask": features.atom_attention_mask,
        "ref_space_uid": features.ref_space_uid,
        "tok_idx": ctx.atom_to_token,
        "s_inputs": ctx.x_inputs,
        "z_trunk": z,
        "relative_position_encoding": ctx.relative_position_encoding,
        "token_attention_mask": features.token_attention_mask,
    }
    torch_args = {k: tensor(v) for k, v in args.items()}
    torch_args.update(
        ref_element=tensor(ctx.ref_element_oh),
        ref_atom_name_chars=tensor(ctx.ref_atom_name_chars_oh),
        s_trunk=None,
        **{k: raw[k] for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")},
    )
    with torch.no_grad():
        expected = model.structure_head.diffusion_module(**torch_args)["x_denoised"]
    actual = eqx.filter_jit(converted.structure_head.diffusion_module)(
        **args,
        ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh,
        n_tokens=3,
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def test_grouped_folding_trunk_preserves_gradients(config):
    # Five blocks exercises both a checkpointed group and the trailing block.
    config.folding_trunk.n_layers = 5
    trunk = esmjfold2.from_torch(ESMFold2Model(config).eval()).folding_trunk
    pair = jax.random.normal(jax.random.key(30), (1, 7, 7, 16))
    cotangent = jax.random.normal(jax.random.key(31), pair.shape)
    mask = jnp.ones((1, 7, 7), dtype=jnp.float32)

    def flat_scan(stack, x):
        @jax.checkpoint
        def body(p, params):
            block = eqx.combine(stack.block_static, params)
            return block(p, pair_attention_mask=mask), None

        return jax.lax.scan(body, x, stack.block_params)[0]

    def evaluate(forward):
        return eqx.filter_jit(jax.value_and_grad(lambda x: jnp.sum(forward(trunk, x) * cotangent)))(
            pair
        )

    old_value, old_grad = evaluate(flat_scan)
    new_value, new_grad = evaluate(lambda stack, x: stack(x, pair_attention_mask=mask))
    np.testing.assert_allclose(new_value, old_value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-5, atol=1e-5)

    def parameter_gradients(forward):
        return eqx.filter_jit(
            eqx.filter_value_and_grad(lambda stack: jnp.sum(forward(stack, pair) * cotangent))
        )(trunk)[1]

    old_params = parameter_gradients(flat_scan)
    new_params = parameter_gradients(lambda stack, x: stack(x, pair_attention_mask=mask))
    for old, new in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params)):
        np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-5)
