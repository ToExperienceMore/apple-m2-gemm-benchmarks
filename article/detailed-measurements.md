# Detailed measurements


### Throughput by matrix size

All entries below are TFLOPS, using the same selection rules as the 2048 table. Values that would round to zero are shown as <0.01; full precision is available in the recorded data.

| N | AMX pack A+B | AMX prepacked | BNNS | SGEMM | MPS | ANE constant B | ANE runtime A+B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.26 | 1.94 | 0.35 | 0.58 | <0.01 | <0.01 | <0.01 |
| 128 | 0.59 | 2.42 | 0.88 | 0.92 | 0.01 | 0.03 | 0.04 |
| 256 | 0.94 | 2.88 | 1.23 | 1.29 | 0.07 | 0.23 | 0.32 |
| 512 | 1.68 | 3.08 | 1.47 | 1.37 | 0.29 | 1.28 | 1.43 |
| 1024 | 2.27 | 3.14 | 1.54 | 1.38 | 1.29 | 4.27 | 3.69 |
| 2048 | 2.36 | 2.93 | 1.43 | 1.16 | 2.34 | 8.54 | 5.22 |
| 4096 | 1.65 | 1.78 | 0.68 | 1.06 | 2.40 | 7.02 | 5.41 |

### Direct AMX instruction throughput

The following numbers are TFLOPS. The benchmark uses Corsix's instruction generator and counting scheme, filtered to `matfp`, one thread, far-spaced Z accumulators, and three arithmetic modes. The rows use two accumulators for FP16→FP16, one for FP16→FP32, and four for FP32→FP32.

| AMX arithmetic | Run 1 | Run 2 | Run 3 | Mean | Best |
|---|---:|---:|---:|---:|---:|
| FP16 × FP16 → FP16 | 3.19 | 3.27 | 3.22 | 3.23 | 3.27 |
| FP16 × FP16 → FP32 | 1.48 | 1.61 | 1.63 | 1.57 | 1.63 |
| FP32 × FP32 → FP32 | 1.62 | 1.62 | 1.63 | 1.62 | 1.63 |

The FP16→FP16 best result reproduces the approximately 3.27 T/s figure in [Corsix's M2 measurements](https://github.com/corsix/amx/issues/6#issuecomment-1477016531). These are arithmetic loops without GEMM operand streaming; complete GEMM results are in the [article](m2-benchmark.md).

### GPU timestamps versus synchronous wall time

MPS also exposes command-buffer GPU timestamps. They exclude the host-side completion overhead included in the main table. Values that would round to zero are shown as <0.01.

| N | Synchronous wall ms | GPU timestamp ms | GPU timestamp TFLOPS |
|---:|---:|---:|---:|
| 64 | 0.31 | 0.12 | <0.01 |
| 128 | 0.34 | 0.14 | 0.03 |
| 256 | 0.51 | 0.23 | 0.14 |
| 512 | 0.94 | 0.71 | 0.38 |
| 1024 | 1.67 | 1.39 | 1.55 |
| 2048 | 7.33 | 7.06 | 2.43 |
| 4096 | 57.23 | 56.79 | 2.42 |

