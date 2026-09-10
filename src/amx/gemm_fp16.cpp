#include "aarch64.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <vector>
using H = _Float16;
static double now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static H *alloc(size_t n) {
  void *p;
  if (posix_memalign(&p, 128, n * 2))
    abort();
  return (H *)p;
}
static void pack(const H *A, const H *B, H *ap, H *bp, int n) {
  for (int i = 0; i < n; i += 32)
    for (int kb = 0; kb < n; kb += 64)
      for (int r = 0; r < 32; r++)
        for (int k = kb; k < kb + 64; k++)
          ap[(size_t)i * n + k * 32 + r] = A[(size_t)(i + r) * n + k];
  for (int j = 0; j < n; j += 64)
    for (int k = 0; k < n; k++)
      memcpy(bp + (size_t)j * n + k * 64, B + (size_t)k * n + j, 128);
}
// MATFP lane mode 2: FP16 X,Y,Z on M2. Two independent 32x32 Z banks.
__attribute__((noinline)) static void tile(const H *a, const H *b, H *c,
                                           int n) {
  constexpr uint64_t mode = 2ull << 42;
  AMX_MATFP(mode | (3ull << 32));
  AMX_MATFP(mode | (1ull << 20) | (3ull << 32));
  for (int k = 0; k < n; k++) {
    AMX_LDX((uint64_t)(b + k * 64) | (1ull << 62));
    AMX_LDY((uint64_t)(a + k * 32));
    AMX_MATFP(mode);
    AMX_MATFP(mode | (1ull << 20) | (64ull << 10));
  }
  for (int r = 0; r < 32; r++) {
    AMX_STZ((uint64_t)(c + (size_t)r * n) | ((uint64_t)(r * 2) << 56));
    AMX_STZ((uint64_t)(c + (size_t)r * n + 32) | ((uint64_t)(r * 2 + 1) << 56));
  }
}
struct Job {
  H *a, *b, *c;
  int n, workers;
};
static void work(void *ptr, size_t id) {
  auto &q = *(Job *)ptr;
  AMX_SET();
  int cols = q.n / 64, total = q.n / 32 * cols;
  for (int t = id; t < total; t += q.workers) {
    int i = (t / cols) * 32, j = (t % cols) * 64;
    tile(q.a + (size_t)i * q.n, q.b + (size_t)j * q.n,
         q.c + (size_t)i * q.n + j, q.n);
  }
  AMX_CLR();
}
static void compute(Job &q) {
  if (q.workers == 1)
    work(&q, 0);
  else
    dispatch_apply_f(q.workers,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &q,
                     work);
}
static H hfma(H a, H b, H c) {
  asm("fmadd %h0, %h1, %h2, %h3" : "=w"(c) : "w"(a), "w"(b), "w"(c));
  return c;
}
int main(int argc, char **v) {
  if (argc != 5)
    return 2;
  int n = atoi(v[1]), threads = atoi(v[2]);
  if (n < 64 || n % 64 || threads < 1)
    return 2;
  size_t count = (size_t)n * n;
  H *A = alloc(count), *B = alloc(count), *C = alloc(count), *ap = alloc(count),
    *bp = alloc(count);
  std::string root = v[3], out = v[4];
  if (!std::ifstream(root + "/A.bin", std::ios::binary)
           .read((char *)A, count * 2) ||
      !std::ifstream(root + "/B.bin", std::ios::binary)
           .read((char *)B, count * 2)) {
    std::cerr << "Input file missing or truncated\n";
    return 2;
  }
  Job q{ap, bp, C, n, threads};
  auto run = [&]() {
    pack(A, B, ap, bp, n);
    compute(q);
  };
  run();
  // Exact comparison with sequential hardware FP16 fused multiply-add.
  int checks = n <= 128 ? n * n : 128;
  for (int t = 0; t < checks; t++) {
    size_t ix = n <= 128 ? t : ((size_t)t * 104729 + 17) % count;
    int i = ix / n, j = ix % n;
    H s = 0;
    for (int k = 0; k < n; k++)
      s = hfma(A[(size_t)i * n + k], B[(size_t)k * n + j], s);
    if (C[ix] != s) {
      std::cerr << "FAIL " << ix << " " << (float)C[ix] << " vs " << (float)s
                << "\n";
      return 3;
    }
  }
  for (int i = 0; i < 10; i++)
    run();
  std::vector<double> full, packed;
  for (int s = 0; s < 10; s++) {
    double t = now();
    for (int l = 0; l < 10; l++)
      run();
    full.push_back((now() - t) / 10);
  }
  for (int s = 0; s < 10; s++) {
    double t = now();
    for (int l = 0; l < 10; l++)
      compute(q);
    packed.push_back((now() - t) / 10);
  }
  std::ofstream f(out + ".json");
  f << "{\"n\":" << n << ",\"threads\":" << threads
    << ",\"fp16_fma_exact_checks\":" << checks << ",\"samples_ms\":[";
  for (int i = 0; i < 10; i++)
    f << (i ? "," : "") << full[i];
  f << "],\"prepacked_samples_ms\":[";
  for (int i = 0; i < 10; i++)
    f << (i ? "," : "") << packed[i];
  f << "]}";
  std::ofstream(out + ".bin", std::ios::binary).write((char *)C, count * 2);
  std::cout << "PASS " << n << " threads=" << threads << "\n";
  free(A);
  free(B);
  free(C);
  free(ap);
  free(bp);
}
