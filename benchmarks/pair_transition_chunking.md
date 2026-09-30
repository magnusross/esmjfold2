# Chunked pair transition: 300M memory experiment

## Setup

The checkpoint is `biohub/ESMFold2-Experimental-Fast-base300M-step1000k`,
converted through this repository's native loader. ESMC-300M features were
precomputed. Runs used float32, no MSA, one recycle (`num_loops=1`), and 14
diffusion steps for full forward. The targets came from `binderopt`: MBP plus
an 80-residue binder is 450 tokens / 4,000 atoms; the 15-PGDH homodimer plus
the binder is 612 tokens / 5,216 atoms.

All table entries were measured on the same L40S (GPU 2), in separate
processes. `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` were set. Peak memory is the JAX
device's cumulative `peak_bytes_in_use` after two calls, including resident
weights and features. Time is the second synchronized JIT call. The trunk
backward probe differentiates the mean final pair state with respect to the
initial pair state; it does not run the entire binder-design loss.

| Probe | Unchunked | 128-row chunks |
| --- | ---: | ---: |
| 450-token trunk input-gradient peak | 16.56 GiB | 14.18 GiB |
| 450-token trunk input-gradient time | 5.62 s | 6.21 s |
| 612-token trunk input-gradient | OOM (25.0 GiB allocation request) | 24.04 GiB; 11.25 s |
| 612-token trunk forward peak | 9.42 GiB | 6.88 GiB |
| 612-token trunk forward time | 2.47 s | 2.64 s |
| 612-token full-fold forward peak | 8.24 GiB | 6.34 GiB |
| 612-token full-fold forward time | 3.23 s | 3.41 s |

At 450 tokens, trunk backward peak decreased 14.0% and warmed runtime
increased 10.5%. At 612 tokens, full-forward peak decreased 23.1%; its
warmed runtime increased 5.7%. The formerly failing 612-token trunk backward
completed. These measurements do not establish that a full ESMFold2 model or
the whole design pipeline fits 600 residues on GH200: only 300M weights were
tested locally, and the experimental checkpoint's confidence head is not
functional.

## Mechanism and choice of chunk size

The pair transition's normalization and SwiGLU are independent across pair
rows. The implementation maps over 128-row bands, padding the last band and
trimming it afterward. Each band is rematerialized during backward. This
bounds the expanded SwiGLU activation size while leaving shorter inputs on
the original direct path. It only changes the folding-trunk `Transition`,
not the MSA encoder's `PairTransition`.

The isolated 612-token transition input-gradient peak fell from 11.46 to
6.67 GiB; the one-block probe fell from 18.42 to 13.57 GiB. A 256-row
variant on GPU 2 used 25.83 GiB and 12.04 s for the 612-token trunk
backward, compared with 24.04 GiB and 11.25 s at 128 rows, so 128 was kept.
The focused CPU test compares forward output, input gradients, and parameter
gradients with the unchunked formula, both below and across the chunk
boundary. All 20 focused transition and existing triangle tests passed.

The installed `binderopt` environment pins an older `esmjfold2` converter,
so its Mosaic design loss cannot run directly against this branch without
updating that integration. This report uses native 300M model calls.

The previous grouped trunk-checkpoint experiment is preserved on
`perf/memory-grouped-trunk-checkpoint`. It reduced the 612-token trunk
backward peak to 24.6 GiB but had a larger 450-token runtime cost in its
shared-GPU run. Measurements across those branches used different GPU loads;
the table above is the direct same-GPU comparison for this experiment.
