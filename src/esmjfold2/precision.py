"""Mixed precision for ESMC and the folding path; geometry stays float32."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .esmc import ESMC, ESMCForMaskedLM
from .primitives import LayerNorm, RMSNorm


def _bf16(module):
    # Do not downcast normalization parameters. Norm implementations accumulate
    # in fp32 and return the input dtype, so they do not widen residual streams.
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x,
        module,
        is_leaf=lambda x: isinstance(x, (LayerNorm, RMSNorm)),
    )


def mixed_precision(model):
    """Return a BF16 ESMC or mixed-precision ESMFold2 model.

    Does not mutate the original. ESMFold2's embedding, language projection,
    pair trunk and confidence paths use BF16; the diffusion head, recurrence
    gates and distogram/loss boundary stay FP32. Conversion and loading APIs
    remain unchanged; apply this once after conversion and before JIT.
    """
    if isinstance(model, (ESMC, ESMCForMaskedLM)):
        return _bf16(model)

    from .model import ESMFold2
    from .experimental import ESMFold2Experimental

    if not isinstance(model, (ESMFold2, ESMFold2Experimental)):
        raise TypeError(f"Unsupported mixed-precision model: {type(model).__name__}")
    names = [
        "inputs_embedder", "z_init_1", "z_init_2", "rel_pos", "token_bonds",
        "language_model", "folding_trunk", "confidence_head",
    ]
    names += (
        ["lm_encoder", "parcae_coda"] if isinstance(model, ESMFold2)
        else ["pair_loop_proj"]
    )
    for name in names:
        module = getattr(model, name)
        if module is not None:
            # Confidence boundaries are physical distances, not weights.
            converted = _bf16(module)
            if name == "confidence_head":
                converted = eqx.tree_at(lambda m: m.boundaries, converted, module.boundaries)
            model = eqx.tree_at(lambda m: getattr(m, name), model, converted)
    return model
