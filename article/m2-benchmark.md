---
title: "Benchmarking Apple M2: AMX, GPU, and Neural Engine"
published: false
description: "GEMM throughput, compute references, execution-path verification, and reproducible code for Apple M2."
tags: performance, benchmarking, machinelearning, cpp
---

An M2 MacBook Air packs several compute engines into a single chip: CPU vector units, the AMX matrix coprocessor, a GPU, and the Apple Neural Engine (ANE). Exploring these engines can reveal more of the machine's computing potential.

The ANE is particularly interesting. Open-source projects such as [ANEForge](https://github.com/sbryngelson/ANEForge) have demonstrated complete ResNet-18 and ResNet-50 inference, alongside other neural-network workloads. They also make it possible to compile and execute programs directly on the ANE, giving developers more control over where computation runs.

How much of that potential translates into practical performance? This article starts with a compute reference for each engine: theoretical rates for CPU/NEON and GPU, measured instruction throughput for AMX, and Apple's nominal rate for ANE. GEMM benchmarks on AMX, GPU and ANE then show how much throughput each delivers on square matrices from 64 to 4096, and how packing and arithmetic precision affect the results.

Three findings stand out:

- **ANE constant-B reaches 8.54 TFLOPS at N=2048**, compared with 2.34 TFLOPS for GPU.
- **Prepacked AMX reaches 2.93 TFLOPS**, falling to 2.36 TFLOPS when A/B packing is included. Both use FP16 accumulation.
- **AMX with FP32 accumulation reaches 88% of its measured arithmetic reference.** Input precision alone is not enough to choose that reference.

**[Source code, data and reproduction scripts](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks)**

## 1. Compute references

A fused multiply-add counts as two operations:

```text
GEMM TFLOPS = 2 × M × N × K / elapsed_seconds / 10^12
% of reference = Measured TFLOPS / Reference T/s
```

| Path / mode | Reference | Type of reference |
|---|---:|---|
| CPU, one P-core, FP32 NEON | 0.112 T/s | Theoretical: 3.50 GHz × 4 vector FMAs/cycle × 4 FP32 lanes × 2 FLOPs/FMA |
| AMX, FP16 × FP16 → FP16 | 3.27 T/s | Best of three local single-thread instruction runs |
| AMX, FP16 × FP16 → FP32 | 1.63 T/s | Best of three local single-thread instruction runs |
| AMX, FP32 × FP32 → FP32 | 1.63 T/s | Best of three local single-thread instruction runs |
| 10-core GPU | 3.57 T/s | Architecture/clock-based reference estimate |
| ANE | 15.80 T/s | Apple's published nominal operation rate |

The GPU estimate is `10 × 256 × 1.395 GHz ≈ 3.57 T/s`. Apple's nominal **15.80 T/s** ANE figure does not specify accumulation precision. NEON provides a theoretical reference; the measured GEMM paths here are AMX, GPU and ANE.

## 2. GEMM: measured hardware performance

AMX is tested with a custom kernel and two Apple library interfaces: **BNNS** for FP16 inputs and **Accelerate SGEMM** for FP32. The GPU uses **Metal Performance Shaders (MPS)**; the ANE uses **ANEForge**, with B either constant in the compiled graph or supplied at runtime.

**M=N=K=2048, batch=1.** Mean synchronous wall time with reused buffers, after 10 warmups and 10 samples of 10 calls.

| Hardware / implementation, N=2048 | Arithmetic / I/O | Reference T/s | Mean ms | TFLOPS | % of reference |
|---|---|---:|---:|---:|---:|
| AMX GEMM, pack A+B each call | FP16 × FP16 → FP16 | 3.27 | 7.28 | 2.36 | 72.1% |
| AMX GEMM, A+B prepacked | FP16 × FP16 → FP16 | 3.27 | 5.87 | 2.93 | 89.5% |
| AMX / BNNS | FP16 × FP16 → FP32 | 1.63 | 12.01 | 1.43 | 88.0% |
| AMX / Accelerate SGEMM | FP32 × FP32 → FP32 | 1.63 | 14.80 | 1.16 | 71.3% |
| GPU / MPS | FP16 I/O | 3.57 | 7.33 | 2.34 | 65.7% |
| ANE / constant B | FP16 I/O | 15.80 | 2.01 | 8.54 | 54.1% |
| ANE / runtime A+B | FP16 I/O | 15.80 | 3.29 | 5.22 | 33.0% |

“FP16 I/O” specifies buffer precision; GPU/ANE accumulation precision is unverified. Reference rates are measured for AMX, estimated for GPU and vendor-nominal for ANE, so the percentages are not a uniform utilization metric.

### Performance across matrix sizes

![GEMM throughput across seven matrix sizes](https://raw.githubusercontent.com/ToExperienceMore/apple-m2-gemm-benchmarks/main/article/figures/gemm-size-sweep.png)

CPU/NEON is the single-P-core theoretical reference (0.112 TFLOPS, dashed). Measured curves use AMX with A/B packing, GPU via MPS, and ANE with constant B.

GPU throughput rises with matrix size. At larger sizes, ANE leads these implementations. Several paths lose throughput at N=4096; the cause remains to be investigated.

### Sample variability

![Mean latency and all ten recorded samples for N=2048](https://raw.githubusercontent.com/ToExperienceMore/apple-m2-gemm-benchmarks/main/article/figures/gemm-2048-latency.png)

AMX with packing and GPU are close: **7.28 ms** versus **7.33 ms**, with overlapping sample ranges.

## 3. How the hardware was tested

- **Hardware:** MacBook Air M2, 4 performance + 4 efficiency CPU cores, 10-core GPU, 16 GB memory.
- **Power:** GEMM sweeps on battery, AMX instruction references on AC.
- **Inputs:** identical seeded FP16 matrices, converted to FP32 for SGEMM. Buffers are reused without cache flushes; runtime A+B supplies both as inputs with fixed contents during timing.
- **Threads:** best mean per size from BNNS 1/4/8 threads and custom AMX 1/4 workers; SGEMM uses `VECLIB_MAXIMUM_THREADS=8`.

Allocation, compilation, initial copies and validation are outside timing. BNNS/SGEMM include library-internal packing. MPS includes commit and completion waits, with encoding before each sample. ANE uses synchronous native execution on resident buffers.

## 4. AMX: packing and accumulation precision

### Packing cost

The custom AMX kernel computes 32×64 output tiles. Packing arranges A in 32-row panels and B in 64-column panels for its outer-product loop. The first row includes this packing on every call; the prepacked row reuses both packed matrices. Both include GEMM loads and C stores.

### FP16 inputs, FP32 accumulation

The AMX path through BNNS reaches **1.43 TFLOPS** at N=2048. Dividing by the **3.27 T/s** FP16-accumulation reference gives **44%**, but FP16 inputs do not imply FP16 accumulation.

Sampling located the hotspot in `libBNNS.dylib`. A runtime breakpoint captured:

```text
instruction:      0x002012a8 → AMX_MATFP(x8)
x8:               0x00000c0000000000
(x8 >> 42) & 0xf:  3
```

[MATFP mode 3](https://github.com/corsix/amx/blob/main/matfp.md) means FP16 inputs with **FP32 accumulation**. The relevant measured reference is therefore **1.63 T/s**, and this AMX implementation achieves **88%** of it. A CPU library call can execute on AMX; the API name and input type alone do not identify the arithmetic.

SGEMM was verified as AMX `FMA32`; the custom kernel uses `MATFP` mode 2 for FP16 accumulation. The GPU path records command-buffer timestamps; ANE execution requires device mask `0x4`. The [sampling, disassembly and runtime records](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/evidence/README.md) document these paths.

## 5. Precision and error

At N=2048, the custom FP16-accumulating AMX kernel has **0.65% relative L2 error** against FP32, compared with roughly **0.02–0.03%** for the other paths. The faster AMX result therefore comes with lower accumulation precision.

All implementations pass their configured checks. The custom kernel is checked against sequential hardware FP16 FMA; the others use FP32 references, with additional FP64 spot checks for ANE. Exact tolerances and error records are in the [validation guide](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/README.md#numerical-validation).

## 6. Reproduce and extend

The [GitHub repository](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks) includes setup instructions, benchmark and plotting scripts, raw samples and [detailed measurement tables](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/article/detailed-measurements.md). See the [benchmark commands](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/README.md#full-benchmark) to run the same size sweep.

The next step is to investigate the gap between measured GEMM throughput and each engine’s compute reference: what limits performance, and how much headroom remains for optimization?

## References

- [Apple M2 announcement](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/) — hardware overview and nominal ANE operation rate.
- [Apple Accelerate](https://developer.apple.com/documentation/accelerate) — BNNS and BLAS interfaces.
- [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) — GPU GEMM interface.
- [Corsix AMX](https://github.com/corsix/amx), [MATFP](https://github.com/corsix/amx/blob/main/matfp.md), [FMA](https://github.com/corsix/amx/blob/main/fma.md), and [M2 measurements](https://github.com/corsix/amx/issues/6#issuecomment-1477016531) — instruction encoding, precision modes, and throughput reference.
- [ANEForge](https://github.com/sbryngelson/ANEForge) — direct ANE graph compilation and execution; see [dependency versions](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/vendor/VERSIONS.json).
- [Apple vs. Oranges](https://arxiv.org/html/2502.05317v1) and [Metal benchmarks](https://github.com/philipturner/metal-benchmarks) — architectural and GPU performance context.
