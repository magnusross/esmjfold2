# Grouped folding-trunk checkpoint: memory experiment

## Setup

Measured on one NVIDIA L40S (46,068 MiB) with JAX 0.10.2 and the
`biohub/ESMFold2-Experimental-Fast-base300M-step1000k` checkpoint, converted
through this repository's native `esm` loader. The model used precomputed
ESMC-300M features, one recycle (`num_loops=1`), no MSA, float32 arrays,
and one prediction at a time. Inputs came from the `binderopt` target configs:
MBP plus an 80-residue binder is 450 tokens / 4,000 atoms; the 15-PGDH
homodimer plus the binder is 612 tokens / 5,216 atoms.

Each entry below is a separate process with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. A second user's small training job
remained on the GPU. Peaks are this process's `peak_bytes_in_use` from
`jax.devices()[0].memory_stats()`, including resident model and feature
buffers. Time is the second, synchronized JIT call; shared-GPU contention
can affect it. The trunk-gradient probe differentiates the mean final pair
state with respect to its initial pair state. It exercises the trunk
backward without running an optimizer or a full binder-design loss.

| Probe | Flat scan | Groups of four |
| --- | ---: | ---: |
| 450-token trunk input-gradient peak | 17.0 GiB | 14.2 GiB |
| 450-token trunk input-gradient time | 7.0 s | 9.0 s |
| 612-token trunk input-gradient | OOM (25.0 GiB allocation request) | 24.6 GiB; 16.9 s |
| 612-token folding stack input-gradient | OOM (23.2 GiB allocation request) | 22.8 GiB; 6.6 s |
| 612-token trunk forward peak | 9.7 GiB | 9.6 GiB |
| 612-token trunk forward time | 3.10 s | 3.17 s |

The 450-token backward peak decreased about 16%; its time increased about
29%. The 612-token trunk backward completed under the local allocation
budget with grouped checkpointing. These results do not establish that the
full ESMFold2 models will fit at 600 residues on GH200: only 300M weights
were available for local testing, and these experimental checkpoints have
non-functional confidence heads.

## Mechanism and alternatives tried

The original `FoldingTrunk` checkpoints each block but the `lax.scan`
backward still keeps block-boundary pair states. A 612 × 612 × 256 float32
pair state is about 366 MiB; a 24-block stack can therefore retain several
GiB of boundaries. The change nests the existing per-block scan in groups
of four and checkpoints each group, so its internal boundaries are
recomputed during backward. Trunks of at most four blocks retain the
original scan.

Checkpointing only the pair transition increased the 612-token single-block
gradient peak from 18.4 to 18.8 GiB, so that attempt was removed. Six-block
groups gave the same 612-token memory use (24.7 GiB) with no clear runtime
improvement, so four-block groups were kept. The targeted CPU regression
test compares forward values, input gradients, and parameter gradients
against the original flat scan across a group boundary.

The installed `binderopt` environment pins an older `esmjfold2` converter.
It cannot run its Mosaic design loss directly against this branch without
updating that integration. The table uses this branch's native 300M model
calls instead.
