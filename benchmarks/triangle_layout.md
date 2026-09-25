# Triangle layout microbenchmark

Run from the repository root with one free GPU:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmarks/triangle_layout.py
```

Measured on one NVIDIA L40S with JAX 0.10.2. Each number is the median of eight warmed, synchronized calls. The benchmark uses float32 input, batch 1, an all-valid pair mask, bias-free projections, and 256 pair and latent channels (the ESMFold2 default). It compares this branch against the original arithmetic retained in `tests/test_triangle_layout.py`.

| Flow | Tokens | Operation | Original ms | Channel-major ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Outgoing | 128 | Forward | 0.745 | 0.753 | 0.99x |
| Outgoing | 128 | Input gradient | 1.674 | 1.213 | 1.38x |
| Outgoing | 256 | Forward | 4.118 | 3.118 | 1.32x |
| Outgoing | 256 | Input gradient | 9.601 | 6.920 | 1.39x |
| Outgoing | 384 | Forward | 9.587 | 7.569 | 1.27x |
| Outgoing | 384 | Input gradient | 22.733 | 15.565 | 1.46x |
| Incoming | 128 | Forward | 0.573 | 0.559 | 1.03x |
| Incoming | 128 | Input gradient | 1.235 | 1.155 | 1.07x |
| Incoming | 256 | Forward | 3.093 | 3.047 | 1.02x |
| Incoming | 256 | Input gradient | 7.693 | 6.973 | 1.10x |
| Incoming | 384 | Forward | 7.596 | 7.583 | 1.00x |
| Incoming | 384 | Input gradient | 18.139 | 15.528 | 1.17x |

These are single-block measurements. They do not establish a speedup for a full ESMFold2 prediction or Binderopt design step. The changed contraction order may shift floating-point results; the focused tests compare forward values and input gradients with the original implementation.
