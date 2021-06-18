#include <groute/event_pool.h>
#include <utils/cuda_utils.h>
#include <utils/stopwatch.h>

#include <iostream>

__global__ static void copyp2p(void* dest, const void* src, size_t size) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  auto unit_size = sizeof(int4);
  auto num_elems = size / unit_size;
  auto rest_bytes = size % unit_size;

  auto* dest_4 = (int4*) dest;
  auto* src_4 = (const int4*) src;

#pragma unroll
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    dest_4[i] = src_4[i];
  }

  auto* dest_c = (char*) dest + size - rest_bytes;
  auto* src_c = (const char*) src + size - rest_bytes;

#pragma unroll
  for (size_t i = globalId; i < rest_bytes; i += gridSize) {
    dest_c[i] = src_c[i];
  }
}

int main(int argc, char** argv) {
  size_t size = 32 * 1024 * 1024;
  char* src_ptr;
  std::vector<char*> dst_ptrs;
  int src_dev = 0;
  std::vector<int> dst_devs;

  for (int i = 1; i < argc; i++) {
    auto dst = std::string(argv[i]);
    dst_devs.push_back(std::stoi(dst));
  }

  std::vector<int> physical_devs{dst_devs};
  physical_devs.push_back(src_dev);

  for (int physical_dev_i : physical_devs) {
    GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev_i));
    for (int physical_dev_j : physical_devs)
      if (physical_dev_i != physical_dev_j)
        cudaDeviceEnablePeerAccess(physical_dev_j, 0);
  }

  GROUTE_CUDA_CHECK(cudaSetDevice(src_dev));
  GROUTE_CUDA_CHECK(cudaMalloc(&src_ptr, size));

  for (int dst_dev : dst_devs) {
    GROUTE_CUDA_CHECK(cudaSetDevice(dst_dev));
    char* ptr;
    GROUTE_CUDA_CHECK(cudaMalloc(&ptr, size));
    dst_ptrs.push_back(ptr);
  }

  GROUTE_CUDA_CHECK(cudaSetDevice(src_dev));

  std::vector<groute::Stream> streams;

  for (int dst_dev : dst_devs) {
    GROUTE_CUDA_CHECK(cudaSetDevice(dst_dev));
    streams.emplace_back(dst_dev);
  }

  {
    double total_time = 0;
    size_t total_size = 0;
    Stopwatch sw;

    for (int iter = 0; iter < 100; iter++) {
      sw.start();

      for (int i = 0; i < dst_devs.size(); i++) {
        GROUTE_CUDA_CHECK(cudaSetDevice(dst_devs[i]));
        GROUTE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptrs[i], dst_devs[i], src_ptr,
                                              src_dev, size,
                                              streams[i].cuda_stream));
      }
      for (auto& stream : streams) {
        stream.Sync();
      }
      sw.stop();
      total_time += sw.ms();
      total_size += size * dst_devs.size();
    }
    std::cout << "Copy with cudaMemcpyAsync:" << std::endl;
    std::cout << "Total time: " << total_time << " ms"
              << " Total size: " << (float) total_size / 1024 / 1024 << " MB"
              << " Bandwidth: " << (float) size / 1024 / 1024 / (sw.ms() / 1000)
              << " MB/s" << std::endl;
  }

  std::cout << std::endl;

  {
    double total_time = 0;
    size_t total_size = 0;
    Stopwatch sw;

    for (int iter = 0; iter < 100; iter++) {
      sw.start();

      for (int i = 0; i < dst_devs.size(); i++) {
        GROUTE_CUDA_CHECK(cudaSetDevice(dst_devs[i]));
        int blockSize = 0;
        int numBlocks = 0;

        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
        copyp2p<<<numBlocks, blockSize, 0, streams[i].cuda_stream>>>(
            dst_ptrs[i], src_ptr, size);
      }
      for (auto& stream : streams) {
        stream.Sync();
      }
      sw.stop();
      total_time += sw.ms();
      total_size += size * dst_devs.size();
    }
    std::cout << "Copy with kernel:" << std::endl;
    std::cout << "Total time: " << total_time << " ms"
              << " Total size: " << (float) total_size / 1024 / 1024 << " MB"
              << " Bandwidth: " << (float) size / 1024 / 1024 / (sw.ms() / 1000)
              << " MB/s" << std::endl;
  }
}
