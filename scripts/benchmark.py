"""Portable runner. Every invocation creates a new results directory."""

import argparse, datetime, json, os, platform, statistics, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--suite", choices=["gemm", "amx", "all"], default="gemm")
p.add_argument(
    "--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096]
)
p.add_argument("--output", type=Path)
a = p.parse_args()
if platform.system() != "Darwin" or platform.machine() != "arm64":
    p.error("Apple Silicon macOS required")
if any(n < 64 or n % 64 for n in a.sizes):
    p.error("sizes must be positive multiples of 64, at least 64")
if not (ROOT / "build/library_gemm").exists():
    p.error("Run scripts/build.py first")
OUT = (
    a.output or ROOT / "runs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
).resolve()
if OUT.exists():
    p.error("Output already exists; choose a new directory")
OUT.mkdir(parents=True)
env = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "vendor/ANEForge"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "ANEFORGE_DYLIB": str(ROOT / "build/libane_e5rt_dispatch.dylib"),
    "ANEFORGE_NO_AUTOBUILD": "1",
    "ANEFORGE_CACHE": str(OUT / "cache"),
    "ANEFORGE_CACHE_DIR": str(OUT / "program-cache"),
    "TMPDIR": str(OUT / "tmp"),
}
(OUT / "tmp").mkdir()
rows = []


def run(cmd, tag, custom_env=None):
    with (OUT / (tag + ".log")).open("w") as log:
        subprocess.run(
            list(map(str, cmd)),
            check=True,
            env=custom_env or env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )


def save(r):
    rows.append(r)
    (OUT / "all.json").write_text(json.dumps(rows, indent=2))
    print(
        f"{r['backend']} N={r.get('n','-')} {r.get('tflops',0):.4f} TFLOPS", flush=True
    )


def error(C, ref):
    return dict(
        relative_l2=float(np.linalg.norm(C - ref) / np.linalg.norm(ref)),
        max_abs=float(np.max(np.abs(C - ref))),
    )


(OUT / "environment.json").write_text(
    json.dumps(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "command": sys.argv,
            "compiler": subprocess.check_output(
                ["xcrun", "clang", "--version"], text=True
            ),
            "hardware": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ),
            "versions": json.loads((ROOT / "vendor/VERSIONS.json").read_text()),
        },
        indent=2,
    )
)
if a.suite in ["gemm", "all"]:
    for n in a.sizes:
        d = OUT / str(n)
        d.mkdir()
        rng = np.random.default_rng(20260910 + n)
        A = (rng.integers(-32, 33, size=(n, n)) / 64).astype(np.float16)
        B = (rng.integers(-32, 33, size=(n, n)) / 64).astype(np.float16)
        A.tofile(d / "A.bin")
        B.tofile(d / "B.bin")
        ref = A.astype(np.float32) @ B.astype(np.float32)
        for mode, threads in [
            ("cpu", 1),
            ("cpu", 4),
            ("cpu", 8),
            ("sgemm", 0),
            ("gpu", 0),
        ]:
            tag = f"{n}-{mode}-{threads}"
            run(
                [
                    ROOT / "build/library_gemm",
                    mode,
                    n,
                    10,
                    10,
                    10,
                    threads,
                    d,
                    d / (tag + ".bin"),
                    d / (tag + ".json"),
                ],
                tag,
                {**env, "VECLIB_MAXIMUM_THREADS": "8"} if mode == "sgemm" else env,
            )
            r = json.loads((d / (tag + ".json")).read_text())
            C = (
                np.fromfile(d / (tag + ".bin"), np.float16)
                .reshape(n, n)
                .astype(np.float32)
            )
            tol = 0.001 if mode == "sgemm" else 0.01
            if not np.allclose(C, ref, rtol=tol, atol=tol):
                raise RuntimeError("Correctness failed: " + tag)
            ms = statistics.mean(r["samples_ms"])
            r.update(
                n=n,
                backend=mode,
                mean_ms=ms,
                tflops=2 * n**3 / ms / 1e9,
                rtol=tol,
                atol=tol,
                allclose=True,
                **error(C, ref),
            )
            save(r)
        for mode in ["constant", "dynamic"]:
            tag = f"{n}-ane-{mode}"
            run(
                [
                    sys.executable,
                    ROOT / "src/ane/matmul_ane.py",
                    "--worker",
                    "--sizes",
                    n,
                    "--mode",
                    mode,
                    "--timer",
                    "native",
                    "--warmup",
                    10,
                    "--loops",
                    10,
                    "--samples",
                    10,
                    "--output-dir",
                    d,
                ],
                tag,
            )
            r = json.loads((d / f"{mode}_{n}.json").read_text())
            r["backend"] = "ane-" + mode
            save(r)
        for threads in [1, 4]:
            tag = f"{n}-amx-fp16-{threads}"
            run([ROOT / "build/gemm_fp16", n, threads, d, d / tag], tag)
            r = json.loads((d / (tag + ".json")).read_text())
            C = (
                np.fromfile(d / (tag + ".bin"), np.float16)
                .reshape(n, n)
                .astype(np.float32)
            )
            if not np.isfinite(C).all():
                raise RuntimeError("Nonfinite FP16 GEMM output")
            r.update(
                backend="amx-fp16",
                mean_ms=statistics.mean(r["samples_ms"]),
                prepacked_ms=statistics.mean(r["prepacked_samples_ms"]),
                **error(C, ref),
            )
            r["tflops"] = 2 * n**3 / r["mean_ms"] / 1e9
            r["prepacked_tflops"] = 2 * n**3 / r["prepacked_ms"] / 1e9
            save(r)
if a.suite in ["amx", "all"]:
    for i in range(1, 4):
        tag = f"amx-instructions-{i}"
        run([ROOT / "build/amx_instructions"], tag)
        (OUT / (tag + ".csv")).write_bytes((OUT / (tag + ".log")).read_bytes())
(OUT / "VALID.json").write_text(
    json.dumps(
        dict(
            completed=datetime.datetime.now().astimezone().isoformat(),
            configurations=len(rows),
            suite=a.suite,
        ),
        indent=2,
    )
)
print("Saved:", OUT)
