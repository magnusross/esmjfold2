"""Profile the 300M model through Mosaic's unchanged loaders.

Requires Mosaic and binderopt source on PYTHONPATH as well as this repo's src.
Run each precision/phase in a fresh process on a free GPU. Example:
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python benchmarks/mixed_precision.py --precision bf16 --tokens 650 --phase design
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MOSAIC_CACHE_DIR", "/data/magnross/.cache/mosaic")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

from binderopt.features import build_esm_binder_features
from binderopt.models import DesignEnsemble, build_design_loss
from binderopt.objective import build_structure_objective, protein_only
from binderopt.targets import target_from_config
from mosaic.losses.esmc import load_esmc
from mosaic.models import esmfold2 as wrappers
from mosaic.optimizers import batched_eval


def sync(tree):
    jax.block_until_ready(tree)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    p.add_argument("--tokens", type=int, default=650)
    p.add_argument("--phase", choices=("design", "forward", "coordinate_grad"), default="design")
    p.add_argument("--targets", type=int, choices=(1, 2), default=1)
    p.add_argument("--sampling-steps", type=int, default=14)
    p.add_argument("--output", type=Path, help="Save values and gradients for cross-process comparison")
    a = p.parse_args()
    binder_length = 80
    dtype = torch.float32 if a.precision == "fp32" else None
    if dtype is not None:
        # Benchmark-only override of an existing private option; public factories
        # and the production loading API are unchanged.
        load = wrappers._load_pretrained
        wrappers._load_pretrained = lambda checkpoint, **kw: load(checkpoint, dtype=dtype, **kw)
    model = wrappers.ESMFold2ExperimentalFast300M(steps="1000k")
    root = Path(os.environ["MOSAIC_CACHE_DIR"]) / "huggingface"
    snapshots = root / "models--Biohub--ESMC-300M" / "snapshots"
    snapshot = snapshots / "a59b831785f907e96e6a246b1d142bfb76df31ee"
    shared = load_esmc(str(snapshot), dtype=dtype)
    model = eqx.tree_at(lambda m: m.esmc, model, shared)
    gc.collect()
    cfg_dir = Path(__file__).resolve().parents[2] / "binderopt" / "configs"
    prepared = []
    for target_id in ("EGFR_human", "EGFR_mouse")[:a.targets]:
        cfg = yaml.safe_load((cfg_dir / "target" / f"{target_id}.yaml").read_text())
        cfg["chains"][0]["sequence"] = cfg["chains"][0]["sequence"][:a.tokens - binder_length]
        prepared.append(build_esm_binder_features(model, binder_length, target_from_config(cfg), supports_msa=False))
    weights_bytes = sum(x.nbytes for x in jax.tree.leaves((model.esmf, shared)) if eqx.is_array(x))
    shapes = [(int(p.features.features.res_type.shape[1]), int(p.features.features.ref_pos.shape[1])) for p in prepared]
    print(json.dumps({"stage": "ready", "precision": a.precision, "phase": a.phase,
                      "shapes": shapes, "weights_gib": weights_bytes / 2**30,
                      "esmc_dtype": str(shared.esmc.embed.weight.dtype),
                      "pair_dtype": str(model.esmf.z_init_1.weight.dtype)}), flush=True)
    sequence = jax.nn.softmax(jax.random.normal(jax.random.key(10), (binder_length, 20)) * 0.2)
    keys = jax.random.split(jax.random.key(17), 1)
    if a.phase == "design":
        objective = yaml.safe_load((cfg_dir / "design.yaml").read_text())["design"]["objective"]
        structures = [protein_only(build_structure_objective(objective), item.masks) for item in prepared]
        stripped = eqx.tree_at(lambda m: m.esmc, model, None)
        ensemble = DesignEnsemble(shared, (stripped,), model)
        loss = build_design_loss(
            ensemble, structures, [p.features for p in prepared],
            ("EGFR_human", "EGFR_mouse")[:a.targets],
            {"recycling_steps": 1, "sampling_steps": a.sampling_steps, "num_diffusion_samples": 1,
             "msa_max_depth": 1024, "lm_dropout": 0.5}, objective,
        )
        def run():
            return batched_eval(loss, sequence[None], keys)
    else:
        pack = prepared[0].features

        @eqx.filter_jit
        def forward(m, ps):
            soft = jax.nn.one_hot(pack.features.res_type, 33)
            soft = soft.at[0, :binder_length].set(ps @ model.res_type_perm)
            features = eqx.tree_at(lambda f: f.res_type, pack.features, soft)
            return m(features, lm_hidden_states=pack.target_lm_hidden, key=keys[0],
                     num_loops=1, num_sampling_steps=a.sampling_steps)

        @eqx.filter_jit
        def coordinate_gradient(m, ps):
            def scalar(ps):
                xyz = forward(m, ps).sample_atom_coords
                # A coordinate-dependent objective exercises diffusion backward.
                return jnp.mean(jnp.square(xyz - xyz.mean(axis=1, keepdims=True)))
            return jax.value_and_grad(scalar)(ps)

        def run():
            if a.phase == "forward":
                return forward(model.esmf, sequence)
            return coordinate_gradient(model.esmf, sequence)
    sync((model, shared, prepared[0].features, sequence, keys))
    device = jax.devices()[0]
    before = device.memory_stats()
    t = time.monotonic()
    result = run()
    sync(result)
    first_seconds = time.monotonic() - t
    t = time.monotonic()
    result = run()
    sync(result)
    seconds = time.monotonic() - t
    stats = device.memory_stats()
    arrays = [np.asarray(x, dtype=np.float32) for x in jax.tree.leaves(result) if eqx.is_array(x)]
    print(json.dumps({"stage": "result", "precision": a.precision, "phase": a.phase,
                      "tokens": a.tokens, "targets": a.targets, "sampling_steps": a.sampling_steps,
                      "first_seconds": first_seconds, "warm_seconds": seconds,
                      "resident_before_gib": before["bytes_in_use"] / 2**30,
                      "peak_before_gib": before["peak_bytes_in_use"] / 2**30,
                      "peak_gib": stats["peak_bytes_in_use"] / 2**30,
                      "finite": all(np.isfinite(x).all() for x in arrays)}), flush=True)
    if a.output:
        if a.phase == "design":
            values, _, gradients = result
        elif a.phase == "coordinate_grad":
            values, gradients = result
        else:
            values, gradients = result.sample_atom_coords, jnp.array([])
        np.savez(a.output, values=np.asarray(values, dtype=np.float32), gradients=np.asarray(gradients, dtype=np.float32))


if __name__ == "__main__":
    main()
