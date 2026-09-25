# Binderopt design workflow comparison

This branch applies only the triangle layout change to `esmjfold2` commit `40a33070c4db646152dc374c650f0dedf3c506a6`, the revision pinned by Binderopt. It was tested against Binderopt commit `078d32c193f4620986b9c32f94acf36c8d9a9410` without editing Binderopt or its Python environment.

## Workload

- One NVIDIA L40S, GPU 0, JAX 0.10.2; no competing GPU jobs observed.
- Binderopt `target=4ZQK optimizer=pgd profile=l40s`, 80-residue binder, 150 optimization steps, batch size 1, two designs with base seed 1. This is the development-only 300M ESMFold2 experimental ensemble.
- Offline, previously staged model assets under `/data/magnross/.cache/mosaic`.
- The installed Binderopt environment was invoked directly. All 25 installed `esmjfold2` Python source files were checked byte-for-byte against the pinned commit. `PYTHONPATH` selected either its installed `esmjfold2` (baseline) or this branch's `src/` (changed). `PYTHONDONTWRITEBYTECODE=1` prevented writes to the source trees.
- Outputs and distinct JAX compilation caches were under `/tmp/binderopt-triangle-workflow`. The first process for each variant started with an empty JAX cache; the second reused its own variant's populated cache. Each process used a new `experiment_id` so no completed cell was overwritten.

Both variants used these Hydra overrides:

```text
target=4ZQK optimizer=pgd profile=l40s profile.cache.offline=true
num_designs=2 output_root=/tmp/binderopt-triangle-workflow/outputs
```

The environment included `CUDA_VISIBLE_DEVICES=0`, `MOSAIC_CACHE_DIR=/data/magnross/.cache/mosaic`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `JAX_COMPILATION_CACHE_DIR` pointing at the variant's temporary cache. Baseline used `PYTHONPATH=/data/magnross/code/binderopt/src`; changed used `PYTHONPATH=/tmp/esmjfold2-pinned-realtest/src:/data/magnross/code/binderopt/src`. The baseline experiment IDs were `triangle_baseline` and `triangle_baseline_repeat`; changed IDs were `triangle_optimized` and `triangle_optimized_repeat`.

## Recorded results

Seconds are read from each cell's `result.json` (`design_timing` and `designs[].design_time_s`). Speedup is baseline / changed, so larger than 1 is faster.

| Run | Metric | Baseline s | Changed s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Empty JAX caches | Complete cell | 454.856 | 387.330 | 1.17x |
| Empty JAX caches | Optimization | 413.162 | 346.684 | 1.19x |
| Empty JAX caches | First design | 233.820 | 199.461 | 1.17x |
| Empty JAX caches | Second design | 179.286 | 147.168 | 1.22x |
| Populated JAX caches | Complete cell | 394.924 | 330.630 | 1.19x |
| Populated JAX caches | Optimization | 367.384 | 303.147 | 1.21x |
| Populated JAX caches | First design | 188.207 | 155.590 | 1.21x |
| Populated JAX caches | Second design | 179.124 | 147.503 | 1.21x |

The two variants used the same recorded `design_key` for each design. Final sequences differed, but the two **unmodified baseline** processes also produced different final sequences from those same keys; their recorded objectives first differed at steps 27 and 77 for designs 0 and 1. Therefore sequence divergence in these process-level runs cannot by itself be assigned to the layout change. This test measures runtime, not equivalence of complete design outcomes. Focused triangle forward and input-gradient parity tests passed. At 256 channels and 16 tokens, the input-gradient relative L2 difference was 4.6e-6 outgoing and 8.7e-6 incoming. Model-weight gradients show a small reduction-order difference and are not used by Binderopt's design optimizer.

The result applies to this L40S development recipe and target length. Full reference-model design and ESMFold2 ranking were not measured.
