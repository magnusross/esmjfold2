# Combined layout and memory optimisations

Branch `comp-optimisations` merges the channel-major triangle layout,
four-block folding-trunk checkpoint groups, and 128-row pair-transition
chunking. The static diffusion-layer loop experiment is not included.

The full repository test suite passed on CPU: 26 tests. Ruff passed on the
merged source and tests.

The 300M 15-PGDH target (612 tokens / 5,216 atoms) was profiled on L40S GPU
2 using the same setup described in `pair_transition_chunking.md`: float32,
precomputed ESMC-300M features, one recycle, no MSA, 14 diffusion steps for
full forward, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Peaks include model
and feature buffers; times are second synchronized JIT calls.

| Probe | Peak JAX bytes in use | Warmed time |
| --- | ---: | ---: |
| Trunk input gradient | 19.06 GiB | 13.89 s |
| Full-fold forward | 6.36 GiB | 3.41 s |

The trunk-gradient probe differentiates the mean final pair state with
respect to the initial pair state. It does not include a full design loss.
The combined trunk gradient uses less memory than transition chunking alone
(24.04 GiB on the same L40S), with more recomputation time (11.25 s for
chunking alone). The full-forward peak is essentially unchanged from
chunking alone (6.34 GiB).

This test does not establish full-model or MSA-enabled memory use on GH200.
