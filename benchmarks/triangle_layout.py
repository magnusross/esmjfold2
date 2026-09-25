"""Compare the original and channel-major triangle blocks on one JAX device.

Run from the repository root with CUDA_VISIBLE_DEVICES set to one free GPU:
    PYTHONPATH=src python benchmarks/triangle_layout.py

This benchmarks one block, not an entire ESMFold2 prediction or design step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# The parity test retains the original implementation as the reference.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_triangle_layout import make_block, reference


def median_ms(fn, *args, repeats=8):
    fn(*args).block_until_ready()  # Compile and warm up before measuring.
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(*args).block_until_ready()
        samples.append((time.perf_counter() - started) * 1000)
    return float(np.median(samples))


def main():
    print(f"device={jax.devices()[0]} jax={jax.__version__}", flush=True)
    print("flow,tokens,channels,mode,original_ms,channel_major_ms,speedup", flush=True)
    for flow in ("outgoing", "incoming"):
        for tokens in (128, 256, 384):
            channels = 256  # ESMFold2's default pair and triangle latent width.
            block = make_block(channels=channels, latent=channels, flow=flow)
            pair = jax.random.normal(jax.random.key(7), (1, tokens, tokens, channels)) * 0.2
            mask = jnp.ones((1, tokens, tokens), dtype=bool)
            operations = (
                ("forward", eqx.filter_jit(lambda b, x, m: reference(b, x, m)),
                 eqx.filter_jit(lambda b, x, m: b(x, m)), (block, pair, mask)),
                ("input_gradient",
                 eqx.filter_jit(jax.grad(lambda x, b, m: jnp.sum(reference(b, x, m)))),
                 eqx.filter_jit(jax.grad(lambda x, b, m: jnp.sum(b(x, m)))),
                 (pair, block, mask)),
            )
            for mode, original, optimized, args in operations:
                old_ms = median_ms(original, *args)
                new_ms = median_ms(optimized, *args)
                print(
                    f"{flow},{tokens},{channels},{mode},"
                    f"{old_ms:.3f},{new_ms:.3f},{old_ms / new_ms:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
