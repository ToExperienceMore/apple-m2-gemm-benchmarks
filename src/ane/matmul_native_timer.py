"""Native repeat timing using the existing loaded ANEForge dispatch function."""

import ctypes
import subprocess
from pathlib import Path
from aneforge import _runtime


class NativeTimer:
    def __init__(self, net, output_dir, loops):
        source = Path(__file__).with_suffix(".c")
        library = output_dir / "libmatmul_native_timer.dylib"
        # Worker output is separate by benchmark run; build before any timing.
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(
                [
                    "clang",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-dynamiclib",
                    str(source),
                    "-o",
                    str(library),
                ],
                check=True,
            )
        self.library = ctypes.CDLL(str(library))
        self.run = self.library.matmul_time_native
        self.run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.run.restype = ctypes.c_int
        self.execute = ctypes.cast(
            _runtime._lib.ane_e5rt_program_execute, ctypes.c_void_p
        )
        self.program = ctypes.c_void_p(net._prog._handle)
        self.loops = ctypes.c_uint64(loops)
        self.elapsed = ctypes.c_double()
        self.elapsed_ptr = ctypes.pointer(self.elapsed)
        self.device_mask = net._prog._device_mask
        if self.device_mask != _runtime.ANE_MASK:
            raise RuntimeError(f"Expected ANE-only mask, got {self.device_mask}")

    def measure_ms(self):
        rc = self.run(self.execute, self.program, self.loops, self.elapsed_ptr)
        if rc:
            raise RuntimeError(f"Native timing failed: {rc}")
        return self.elapsed.value / 1e6 / self.loops.value
