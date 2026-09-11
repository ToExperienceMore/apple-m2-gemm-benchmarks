An M2 MacBook Air packs several compute engines into a single chip: CPU vector units, the AMX matrix coprocessor, a GPU, and the Apple Neural Engine (ANE). Beyond conventional CPU vector computation, **AMX, GPU and ANE can all be accessed through existing libraries and community tools**.

I was especially interested in ANE after confirming that I could **run ResNet-18 inference end-to-end on the ANE**. ANEForge also provides an entry point to **build, compile and execute supported tensor graphs**, and the [project reports support for selected Llama/Qwen-class language models](https://github.com/sbryngelson/ANEForge#language-models).

These discoveries made me want to measure the scale of all three engines' computing capabilities. This benchmark compares **measured matrix-multiplication (GEMM) throughput** on AMX, GPU and ANE against a **reference compute throughput** for each engine, using square matrices from 64 × 64 to 4096 × 4096. Throughput is reported in TFLOPS—trillions of floating-point operations per second.

**Software paths to the hardware**

| Hardware | Software entry point | What it provides in this benchmark |
|---|---|---|
| GPU | [Metal and MPS](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) | Apple's official GPU compute stack; MPS supplies the matrix-multiplication implementation. |
| AMX | [Accelerate: BNNS and BLAS](https://developer.apple.com/documentation/accelerate) | Library matrix operations that were verified to execute on AMX in the N=2048 tests. |
| AMX | [Corsix's AMX instruction documentation](https://github.com/corsix/amx) | Reverse-engineered instructions used by the custom matrix kernel and instruction-throughput tests. |
| ANE | [ANEForge](https://github.com/sbryngelson/ANEForge) | Direct compilation and execution of supported tensor graphs, used here for matrix multiplication. |

[Core ML can also use ANE](https://developer.apple.com/documentation/coreml/mlcomputeunits/all) for model inference. ANEForge offers a more direct route for supported tensor computations outside that deployment flow; its operation and shape constraints differ from writing general-purpose GPU kernels.

For **2048 × 2048 matrices**, three findings stand out:

- **ANE reaches 8.54 TFLOPS, AMX 2.93 TFLOPS, and GPU 2.34 TFLOPS.**
- These results reach roughly **54%, 90% and 66%** of their respective reference rates.
- **Input precision alone does not determine the right reference.** The precision used to add up the products matters too.

## 1. Compute throughput references

The CPU/NEON and GPU figures below are theoretical estimates, the AMX figures are measured instruction throughput, and the ANE figure is Apple's published nominal rate. T/s means trillions of operations per second. In `FP16 × FP16 → FP32`, the arrow indicates the precision used to add up the products, called *accumulation*.

| Path / mode | Reference throughput | Source |
|---|---:|---|
| CPU, one P-core, FP32 NEON | 0.112 T/s | Theoretical: 3.50 GHz × 4 vector FMAs/cycle × 4 FP32 lanes × 2 FLOPs/FMA |
| AMX, FP16 × FP16 → FP16 | 3.27 T/s | Measured single-thread instruction throughput |
| AMX, FP16 × FP16 → FP32 | 1.63 T/s | Measured single-thread instruction throughput |
| AMX, FP32 × FP32 → FP32 | 1.63 T/s | Measured single-thread instruction throughput |
| 10-core GPU | 3.57 T/s | Architecture/clock-based reference estimate |
| ANE | 15.80 T/s | Apple's published nominal operation rate |

A **single P-core's FP32 NEON reference is 0.112 T/s**; AMX and GPU references are in the low single-digit T/s range, while ANE's nominal rate is 15.80 T/s. This gives the scale of the matrix-computing capability beyond one CPU core, without treating the different precision modes as equivalent.

The GPU estimate is `10 × 256 × 1.395 GHz ≈ 3.57 T/s`. Apple's ANE figure does not specify accumulation precision.

For matrix multiplication, each multiply-and-add counts as two operations:

```text
GEMM TFLOPS = 2 × M × N × K / elapsed_seconds / 10^12
% of reference = Measured TFLOPS / Reference T/s × 100
```

Here the matrices are square, so M=N=K is the number of rows and columns.

## 2. Matrix multiplication benchmark results

I tested available high-performance GEMM implementations through the software paths above, including both the custom AMX kernel and Apple library implementations.

**Matrix size: 2048 × 2048.**

| Hardware / implementation, N=2048 | Input / accumulation precision | Reference throughput (T/s) | Mean time (ms) ↓ | Throughput (TFLOPS) ↑ | Throughput / reference (%) |
|---|---|---:|---:|---:|---:|
| AMX GEMM, packing included | FP16 × FP16 → FP16 | 3.27 | 7.28 | 2.36 | 72.1% |
| AMX GEMM, prepacked | FP16 × FP16 → FP16 | 3.27 | 5.87 | 2.93 | 89.5% |
| AMX / BNNS | FP16 × FP16 → FP32 | 1.63 | 12.01 | 1.43 | 88.0% |
| AMX / Accelerate SGEMM | FP32 × FP32 → FP32 | 1.63 | 14.80 | 1.16 | 71.3% |
| GPU / MPS | FP16 I/O | 3.57 | 7.33 | 2.34 | 65.7% |
| ANE / constant B | FP16 I/O | 15.80 | 2.01 | 8.54 | 54.1% |
| ANE / runtime inputs | FP16 I/O | 15.80 | 3.29 | 5.22 | 33.0% |

ANE delivers the highest throughput here; AMX and GPU both reach the low single-digit TFLOPS range.

“FP16 I/O” describes the input and output values; GPU/ANE accumulation precision is unverified. The percentages use the reference types in Section 1, rather than a common measure of hardware utilization.

### Performance across matrix sizes

![GEMM throughput across seven matrix sizes](https://raw.githubusercontent.com/ToExperienceMore/apple-m2-gemm-benchmarks/main/article/figures/gemm-size-sweep.png)

The dashed line is the **single P-core FP32 NEON theoretical reference**; the other curves are measured GEMM throughput.

GPU throughput rises with matrix size, and ANE leads at larger sizes in these tests. Several implementations lose throughput at 4096 × 4096; the cause remains to be investigated.

## 3. How the benchmarks were run

The tests ran on a MacBook Air M2 with 4 performance and 4 efficiency CPU cores, a 10-core GPU and 16 GB memory. Full measurement conditions are in the [reproduction notes](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/README.md#measurement-conditions).

[SiliconScope](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/evidence/README.md#watching-engine-activity-with-siliconscope) provides a live view of CPU, GPU and ANE activity while the workload runs. AMX execution and precision were checked separately through sampling, disassembly and breakpoints; the [verification guide](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/evidence/README.md) contains the detailed records and reproduction steps.

## 4. Why accumulation precision matters

**FP16 inputs do not necessarily mean FP16 accumulation.** Each output value is a sum of products, and the running sum can use a different precision from the inputs.

BNNS reaches **1.43 TFLOPS** at N=2048. Comparing it with the **3.27 T/s** AMX reference for FP16 accumulation gives about **44%**. But runtime checks confirmed that BNNS uses **FP32 accumulation**. Its matching instruction reference is **1.63 T/s**, so it reaches **88%** of that rate. The input type alone would have led to the wrong conclusion about how much of the arithmetic capacity was being used.

The checks also confirmed FP32 AMX calculations for SGEMM and FP16 accumulation for the custom kernel. Instruction modes and debugger records are in the [execution evidence](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/evidence/README.md#detailed-execution-path-verification).

### The precision tradeoff

At N=2048, the custom AMX kernel with FP16 accumulation has **0.65% relative L2 error** against an FP32 reference, compared with roughly **0.02–0.03%** for the other implementations. Relative L2 error measures the overall output difference relative to the reference. The faster custom AMX result comes with a larger measured error.

All implementations pass their configured correctness checks. The [validation guide](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/README.md#numerical-validation) gives the reference calculations and tolerances.

## 5. What this benchmark shows

**AMX, GPU and ANE all have usable compute paths, and each delivered multi-TFLOPS GEMM throughput on this M2.** The measurements establish the scale of their usable matrix-computing capability and how it compares with the reference rates.

The remaining gaps to the reference rates are starting points for bottleneck investigation, not guaranteed recoverable speedups. The next step is to examine what limits each implementation, especially at larger matrix sizes. The [GitHub README](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/README.md#quick-start) provides build and benchmark commands; [detailed measurements](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/article/detailed-measurements.md) support further analysis.

## References

- [Apple M2 announcement](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/) — hardware overview and nominal ANE operation rate.
- [Apple Accelerate](https://developer.apple.com/documentation/accelerate) — BNNS and BLAS interfaces.
- [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) — GPU GEMM interface.
- [Corsix AMX](https://github.com/corsix/amx), [MATFP](https://github.com/corsix/amx/blob/main/matfp.md), [FMA](https://github.com/corsix/amx/blob/main/fma.md), and [M2 measurements](https://github.com/corsix/amx/issues/6#issuecomment-1477016531) — instruction encoding, precision modes, and throughput reference.
- [ANEForge](https://github.com/sbryngelson/ANEForge) — direct ANE graph compilation and execution; see [dependency versions](https://github.com/ToExperienceMore/apple-m2-gemm-benchmarks/blob/main/vendor/VERSIONS.json).
- [Apple vs. Oranges](https://arxiv.org/html/2502.05317v1) and [Metal benchmarks](https://github.com/philipturner/metal-benchmarks) — architectural and GPU performance context.
