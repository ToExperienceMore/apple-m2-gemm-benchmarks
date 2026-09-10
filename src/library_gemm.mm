#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <vector>
static double now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static void readfile(const char *path, void *p, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f.read((char *)p, bytes))
    throw std::runtime_error("input read failed");
}
static BNNSNDArrayDescriptor desc(void *p, int n) {
  BNNSNDArrayDescriptor d = {};
  d.layout = BNNSDataLayoutRowMajorMatrix;
  d.size[0] = n;
  d.size[1] = n;
  d.stride[0] = 1;
  d.stride[1] = n;
  d.data = p;
  d.data_type = BNNSDataTypeFloat16;
  return d;
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 10)
        return 2;
      std::string mode = argv[1];
      int n = atoi(argv[2]), warmup = atoi(argv[3]), loops = atoi(argv[4]),
          samples = atoi(argv[5]), threads = atoi(argv[6]);
      size_t bytes = (size_t)n * n * 2;
      std::string root = argv[7];
      std::vector<double> times, deviceTimes;
      std::vector<uint16_t> A(n * (size_t)n), B(A.size()), C(A.size(), 0x7e00);
      readfile((root + "/A.bin").c_str(), A.data(), bytes);
      readfile((root + "/B.bin").c_str(), B.data(), bytes);
      size_t workspaceBytes = 0;
      if (mode == "sgemm") {
        std::vector<float> af(A.size()), bf(A.size()), cf(A.size());
        for (size_t i = 0; i < A.size(); i++) {
          af[i] = ((__fp16 *)A.data())[i];
          bf[i] = ((__fp16 *)B.data())[i];
        }
        auto run = [&]() {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1,
                      af.data(), n, bf.data(), n, 0, cf.data(), n);
        };
        for (int i = 0; i < warmup; i++)
          run();
        for (int s = 0; s < samples; s++) {
          double t = now();
          for (int i = 0; i < loops; i++)
            run();
          times.push_back((now() - t) / loops);
        }
        for (size_t i = 0; i < C.size(); i++)
          ((__fp16 *)C.data())[i] = cf[i];
      } else if (mode == "cpu") {
        auto ad = desc(A.data(), n), bd = desc(B.data(), n),
             cd = desc(C.data(), n);
        BNNSFilterParameters params = {};
        params.n_threads = threads;
        ssize_t need =
            BNNSMatMulWorkspaceSize(false, false, 1, &ad, &bd, &cd, &params);
        if (need < 0)
          throw std::runtime_error("workspace query failed");
        workspaceBytes = need;
        void *workspace = nullptr;
        if (posix_memalign(&workspace, 64, std::max<size_t>(64, need)))
          throw std::bad_alloc();
        auto run = [&]() {
          if (BNNSMatMul(false, false, 1, &ad, &bd, &cd, workspace, &params))
            throw std::runtime_error("BNNS failure");
        };
        for (int i = 0; i < warmup; i++)
          run();
        for (int s = 0; s < samples; s++) {
          double t = now();
          for (int i = 0; i < loops; i++)
            run();
          times.push_back((now() - t) / loops);
        }
        free(workspace);
      } else if (mode == "gpu") {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto md =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:n
                                                 rowBytes:n * 2
                                                 dataType:MPSDataTypeFloat16];
        id<MTLBuffer> ab =
            [dev newBufferWithBytes:A.data()
                             length:bytes
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb =
            [dev newBufferWithBytes:B.data()
                             length:bytes
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> cb =
            [dev newBufferWithBytes:C.data()
                             length:bytes
                            options:MTLResourceStorageModeShared];
        auto am = [[MPSMatrix alloc] initWithBuffer:ab descriptor:md],
             bm = [[MPSMatrix alloc] initWithBuffer:bb descriptor:md],
             cm = [[MPSMatrix alloc] initWithBuffer:cb descriptor:md];
        auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                        transposeLeft:false
                                                       transposeRight:false
                                                           resultRows:n
                                                        resultColumns:n
                                                      interiorColumns:n
                                                                alpha:1
                                                                 beta:0];
        auto encode = [&]() {
          id<MTLCommandBuffer> c = [q commandBuffer];
          [kernel encodeToCommandBuffer:c
                             leftMatrix:am
                            rightMatrix:bm
                           resultMatrix:cm];
          return c;
        };
        auto run = [&](id<MTLCommandBuffer> c) {
          [c commit];
          [c waitUntilCompleted];
          if (c.status == MTLCommandBufferStatusError)
            throw std::runtime_error(c.error.description.UTF8String);
        };
        for (int i = 0; i < warmup; i++)
          run(encode());
        for (int s = 0; s < samples; s++) {
          @autoreleasepool {
            NSMutableArray<id<MTLCommandBuffer>> *commands =
                [NSMutableArray arrayWithCapacity:loops];
            for (int i = 0; i < loops; i++)
              [commands addObject:encode()];
            double t = now();
            for (id<MTLCommandBuffer> c in commands)
              run(c);
            times.push_back((now() - t) / loops);
            double dt = 0;
            for (id<MTLCommandBuffer> c in commands)
              dt += (c.GPUEndTime - c.GPUStartTime) * 1000;
            deviceTimes.push_back(dt / loops);
          }
        }
        memcpy(C.data(), cb.contents, bytes);
      } else
        throw std::runtime_error("unknown mode");
      std::ofstream out(argv[8], std::ios::binary);
      out.write((char *)C.data(), bytes);
      std::ofstream result(argv[9]);
      result << "{\"samples_ms\":[";
      for (size_t i = 0; i < times.size(); i++)
        result << (i ? "," : "") << times[i];
      result << "],\"gpu_samples_ms\":[";
      for (size_t i = 0; i < deviceTimes.size(); i++)
        result << (i ? "," : "") << deviceTimes[i];
      result << "],\"threads\":" << threads
             << ",\"workspace_bytes\":" << workspaceBytes << "}";
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      return 1;
    }
  }
}
