"""ANE FP16 square GEMM: constant-weight and two-runtime-input paths."""

import argparse, json, time, subprocess, sys
from pathlib import Path
import numpy as np
import aneforge as af

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--output-dir", type=Path, default=ROOT / "matmul-results")
p.add_argument(
    "--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096]
)
p.add_argument("--loops", type=int, default=10)
p.add_argument("--samples", type=int, default=10)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--mode", choices=["both", "constant", "dynamic"], default="both")
p.add_argument("--worker", action="store_true")
p.add_argument("--timer", choices=["python", "native", "compare"], default="python")
a = p.parse_args()
OUT = a.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
if min(a.sizes + a.sizes[:0] + [a.loops, a.samples]) < 1 or a.warmup < 0:
    p.error("positive sizes/loops/samples required")
if not a.worker:
    rows = []
    for n in a.sizes:
        for mode in (["constant", "dynamic"] if a.mode == "both" else [a.mode]):
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--sizes",
                str(n),
                "--mode",
                mode,
                "--loops",
                str(a.loops),
                "--samples",
                str(a.samples),
                "--warmup",
                str(a.warmup),
                "--output-dir",
                str(OUT),
            ]
            cmd.extend(["--timer", a.timer])
            print(f"RUN N={n} mode={mode}", flush=True)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                (OUT / f"{mode}_{n}.log").write_text(r.stdout + r.stderr)
                if r.returncode:
                    raise RuntimeError(
                        f"exit {r.returncode}: " + (r.stderr + r.stdout)[-1200:]
                    )
                row = json.loads((OUT / f"{mode}_{n}.json").read_text())
                rows.append(row)
                print(
                    f"PASS {mode} {n}: {row['mean_ms']:.4f} ms, {row['tflops']:.3f} TFLOPS, full relative L2 {row['relative_l2_full']:.6g}",
                    flush=True,
                )
            except Exception as e:
                rows.append({"n": n, "mode": mode, "error": str(e)})
                print("FAIL", n, mode, str(e), flush=True)
            (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print("DONE", flush=True)
    sys.exit(int(any("error" in row for row in rows)))
n = a.sizes[0]
rng = np.random.default_rng(20260910 + n)
A = (rng.integers(-32, 33, size=(n, n)) / 64).astype(np.float16)
B = (rng.integers(-32, 33, size=(n, n)) / 64).astype(np.float16)
x = af.input((n, n))
if a.mode == "constant":
    y = x @ B
else:
    b = af.input((n, n))
    y = x @ b
start = time.perf_counter()
net = af.compile(y, opt=0, build_dir=str(OUT / f"compiled_{a.mode}_{n}"))
compile_s = time.perf_counter() - start
try:
    native = None
    if a.timer != "python":
        from matmul_native_timer import NativeTimer

        native = NativeTimer(net, OUT, a.loops)
    np.copyto(net.input_view(x._name), A)
    if a.mode == "dynamic":
        np.copyto(net.input_view(b._name), B)
    net.output_view().fill(np.nan)
    for _ in range(a.warmup):
        net.execute()
    python_samples = []
    native_samples = []
    output = net.output_view()
    for sample_index in range(a.samples):
        order = (
            (["python", "native"] if sample_index % 2 == 0 else ["native", "python"])
            if a.timer == "compare"
            else [a.timer]
        )
        for timer in order:
            output.fill(
                np.nan
            )  # Outside timing: prove this path overwrites the output.
            if timer == "native":
                native_samples.append(native.measure_ms())
            else:
                start = time.perf_counter_ns()
                for _ in range(a.loops):
                    net.execute()
                python_samples.append((time.perf_counter_ns() - start) / 1e6 / a.loops)
            assert np.isfinite(output).all(), f"{timer}: nonfinite output"
    samples = native_samples if native_samples else python_samples
    C = net.output_view().copy()
    assert np.isfinite(C).all(), "nonfinite output"
    coords = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)] + [
        (int(rng.integers(n)), int(rng.integers(n))) for _ in range(60)
    ]
    ref = np.array(
        [A[i, :].astype(np.float64) @ B[:, j].astype(np.float64) for i, j in coords]
    )
    got = np.array([C[i, j] for i, j in coords], dtype=np.float64)
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    assert rel < 0.01, f"wrong output rel={rel}"
    # Full CPU FP32 reference, outside the measured intervals. Inputs are exact
    # multiples of 1/64; retain the independent FP64 dot checks above as well.
    reference = A.astype(np.float32) @ B.astype(np.float32)
    delta = C.astype(np.float32) - reference
    full_rel = float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30))
    max_abs = float(np.max(np.abs(delta)))
    full_ok = bool(np.allclose(C, reference, rtol=1e-3, atol=1e-3))
    assert full_ok, f"full output mismatch: relative L2={full_rel}, max abs={max_abs}"
    ms = float(np.mean(samples))
    tf = 2 * n**3 / ms / 1e9
    row = {
        "n": n,
        "mode": a.mode,
        "dtype": "fp16",
        "loops": a.loops,
        "samples": a.samples,
        "warmup": a.warmup,
        "compile_s": compile_s,
        "mean_ms": ms,
        "median_ms": float(np.median(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "tflops": tf,
        "relative_l2_64_samples": rel,
        "max_abs_64_samples": float(np.max(np.abs(got - ref))),
        "samples_ms": samples,
        "timing": "resident buffers; no host IO in timed loop; synchronous dispatch per GEMM; CPU monotonic wall time, not hardware-only timing",
    }
    row.update(
        timer=a.timer,
        python_samples_ms=python_samples,
        native_samples_ms=native_samples,
        primary_timer="native" if native_samples else "python",
    )
    if native:
        row["device_mask"] = native.device_mask
    if python_samples:
        row["python_mean_ms"] = float(np.mean(python_samples))
    row.update(
        relative_l2_full=full_rel,
        max_abs_full=max_abs,
        full_allclose=full_ok,
        rtol=1e-3,
        atol=1e-3,
        reference="CPU NumPy FP32 full matrix + 64 FP64 dot checks",
    )
    (OUT / f"{a.mode}_{n}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
finally:
    net.release()
