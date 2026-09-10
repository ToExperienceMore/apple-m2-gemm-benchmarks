// CPU monotonic timing around a native dispatch loop, NOT device timestamps.
// Caller owns all buffers, program state, and the result scalar.
#include <stdint.h>
#include <time.h>

typedef int (*execute_fn)(void *);

int matmul_time_native(execute_fn execute, void *program, uint64_t loops,
                       double *elapsed_ns) {
  if (!execute || !program || !loops || !elapsed_ns)
    return -1;
  struct timespec start, end;
  if (clock_gettime(CLOCK_MONOTONIC, &start))
    return -2;
  for (uint64_t i = 0; i < loops; ++i) {
    int rc = execute(program);
    if (rc)
      return rc;
  }
  if (clock_gettime(CLOCK_MONOTONIC, &end))
    return -2;
  *elapsed_ns = (double)(end.tv_sec - start.tv_sec) * 1e9 +
                (double)(end.tv_nsec - start.tv_nsec);
  return 0;
}
