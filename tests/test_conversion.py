"""Native Biohub model loading, conversion, and CPU numerical parity."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from esm.models.esmc import EsmcConfig, EsmcForMaskedLM, EsmcModel
from esm.models.esmfold2 import EsmFold2Config, EsmFold2ExperimentalModel, EsmFold2Model
from esm.models.esmfold2.protein_utils import prepare_protein_features
from safetensors.torch import save_file

import esmjfold2


@pytest.fixture
def config():
    return EsmFold2Config(
        hidden_size=32,
        pairwise_hidden_size=16,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk_num_hidden_layers=1,
        folding_trunk_num_attention_heads=2,
        atom_encoder={"hidden_size": 64, "num_attention_heads": 2, "num_hidden_layers": 1},
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                "atom_encoder": {"hidden_size": 64, "num_attention_heads": 2, "num_hidden_layers": 1},
                "token_hidden_size": 32,
                "c_z": 16,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
        },
        parcae_num_coda_layers=1,
        lm_encoder={"num_hidden_layers": 1},
        msa_encoder={"enabled": True, "hidden_size": 32, "outer_hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "head_width": 8},
    )


@pytest.mark.parametrize("model_class", [EsmcModel, EsmcForMaskedLM])
def test_esmc_parity(model_class):
    torch.manual_seed(4)
    model = model_class(EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2)).eval()
    converted = esmjfold2.from_torch(model)
    ids = np.array([[0, 5, 6, 7, 2]], dtype=np.int32)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids), output_hidden_states=True)
    output, hidden = eqx.filter_jit(converted)(jnp.asarray(ids), collect_hidden_states=True)
    reference = expected.logits if model_class is EsmcForMaskedLM else expected.last_hidden_state
    np.testing.assert_allclose(output, reference.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hidden, expected.hidden_states.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("model_class", [EsmFold2Model, EsmFold2ExperimentalModel])
def test_checkpoint_conversion_and_prediction(config, model_class, tmp_path):
    """Exercise the real loader, all converters, JIT inference and serialization."""
    torch.manual_seed(0)
    config.type = "experimental" if model_class is EsmFold2ExperimentalModel else "release"
    config.esmc_config = EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2)
    config.save_pretrained(tmp_path)
    save_file(model_class(config).eval().state_dict(), tmp_path / "model.safetensors")
    model = EsmFold2Model.from_pretrained(tmp_path, device="cpu", dtype=torch.float32, esmc_precision="fp32")
    assert isinstance(model, model_class)
    converted = esmjfold2.from_torch(model)
    esmc = esmjfold2.from_torch(model.esmc)
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    lm = esmjfold2.compute_lm_hidden_states(
        esmc, raw["input_ids"], raw["asym_id"], raw["residue_index"],
        raw["mol_type"], raw["token_attention_mask"],
    )
    output = eqx.filter_jit(converted)(features, lm_hidden_states=lm, key=jax.random.key(0), num_loops=1, num_sampling_steps=2)
    assert output.sample_atom_coords.shape[-2:] == (raw["ref_pos"].shape[1], 3)
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
    esmjfold2.save_model(converted, tmp_path / "converted")
    restored = esmjfold2.load_model(tmp_path / "converted")
    assert eqx.tree_equal(converted, restored)


def test_folding_components_parity(config):
    """Compare deterministic modules with identical inputs, avoiding RNG differences."""
    torch.manual_seed(7)
    model = EsmFold2Model(config).eval()
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

    compare(converted.rel_pos, model.rel_pos, **{
        k: getattr(features, k)
        for k in ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")
    })
    z = jax.random.normal(jax.random.key(1), (1, 3, 3, config.pairwise_hidden_size))
    compare(converted.folding_trunk, model.folding_trunk, pair=z, pair_attention_mask=ctx.pair_mask)

    x = jax.random.normal(jax.random.key(2), features.ref_pos.shape)
    args = {
        "x_noisy": x, "t_hat": jnp.array([1.5]), "ref_pos": features.ref_pos,
        "ref_charge": features.ref_charge, "ref_mask": features.atom_attention_mask,
        "ref_space_uid": features.ref_space_uid, "tok_idx": ctx.atom_to_token,
        "s_inputs": ctx.x_inputs, "z_trunk": z,
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
        **args, ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh, n_tokens=3,
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def test_grouped_folding_trunk_preserves_gradients(config):
    # Five blocks exercises both a checkpointed group and the trailing block.
    config.folding_trunk_num_hidden_layers = 5
    trunk = esmjfold2.from_torch(EsmFold2Model(config).eval()).folding_trunk
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
        return eqx.filter_jit(jax.value_and_grad(
            lambda x: jnp.sum(forward(trunk, x) * cotangent)
        ))(pair)

    old_value, old_grad = evaluate(flat_scan)
    new_value, new_grad = evaluate(
        lambda stack, x: stack(x, pair_attention_mask=mask)
    )
    np.testing.assert_allclose(new_value, old_value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-5, atol=1e-5)

    def parameter_gradients(forward):
        return eqx.filter_jit(eqx.filter_value_and_grad(
            lambda stack: jnp.sum(forward(stack, pair) * cotangent)
        ))(trunk)[1]

    old_params = parameter_gradients(flat_scan)
    new_params = parameter_gradients(
        lambda stack, x: stack(x, pair_attention_mask=mask)
    )
    for old, new in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params)):
        np.testing.assert_allclose(new, old, rtol=1e-5, atol=1e-5)
