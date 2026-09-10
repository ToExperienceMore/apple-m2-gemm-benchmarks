"""Build the benchmark executables inside this bundle."""

import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


run(
    [
        "xcrun",
        "clang++",
        "-O3",
        "-std=c++17",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
        ROOT / "src/library_gemm.mm",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "Accelerate",
        "-o",
        BUILD / "library_gemm",
    ]
)
run(
    [
        "xcrun",
        "clang++",
        "-O3",
        "-std=c++17",
        "-I",
        ROOT / "vendor/corsix-amx",
        ROOT / "src/amx/gemm_fp16.cpp",
        "-o",
        BUILD / "gemm_fp16",
    ]
)
run([sys.executable, ROOT / "vendor/corsix-amx/perf_kernels.py"], cwd=BUILD)
run(
    [
        "xcrun",
        "clang",
        "-O2",
        "-g",
        "-I",
        ROOT / "vendor/corsix-amx",
        ROOT / "src/amx/perf_filtered.c",
        BUILD / "perf_kernels.c",
        "-o",
        BUILD / "amx_instructions",
    ]
)
lib = ROOT / "vendor/ANEForge/aneforge/_lib"
run(
    [
        "xcrun",
        "clang++",
        "-O2",
        "-Wall",
        "-Wextra",
        "-dynamiclib",
        "-fPIC",
        "-fobjc-arc",
        "-framework",
        "Foundation",
        "-I",
        lib,
        lib / "ane_e5rt_dispatch.mm",
        "-o",
        BUILD / "libane_e5rt_dispatch.dylib",
    ]
)
print("Built all four benchmark paths in", BUILD)
