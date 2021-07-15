
#ifndef RING_BUFFER_H
#define RING_BUFFER_H
#include <groute/common.h>
#include <groute/internal/cuda_utils.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace groute {

template <typename T>
class RingBuffer {
  int m_dev;
  size_t m_capacity;
  T* m_data;
  thrust::device_vector<T> m_dev_data;
  thrust::host_vector<T> m_host_data;
  std::mutex m_mutex;

  thrust::host_vector<
      size_t, thrust::system::cuda::experimental::pinned_allocator<size_t>>
      m_start, m_end, m_pending;
  Stream m_stream;

 public:
  RingBuffer(int dev, size_t capacity)
      : m_dev(dev), m_capacity(capacity), m_stream(dev) {
    GROUTE_CUDA_CHECK(cudaSetDevice(dev));

    if (dev != Device::Host) {
      m_dev_data.resize(capacity);
      m_data = thrust::raw_pointer_cast(m_dev_data.data());
    } else {
      m_host_data.resize(capacity);
      m_data = thrust::raw_pointer_cast(m_host_data.data());
    }
    m_start.push_back(0);
    m_end.push_back(0);
    m_pending.push_back(0);
  }

  std::vector<Buffer<T>> GetWritableBuffer(size_t size) {
    std::unique_lock<std::mutex> lock(m_mutex);
    std::vector<Buffer<T>> segs;
    size_t pending = m_pending[0];
    size_t used_space = m_pending[0] - m_start[0];
    size_t rest_space = m_capacity - used_space;
    size_t new_pending = m_pending[0] + size;

    if (size > rest_space) {
      printf(
          "Dev: %d buffer overflow, start %lu end %lu pending %lu requiring "
          "size %lu\n",
          m_dev, m_start, m_end, m_pending, size);
      exit(16);
    }

    pending %= m_capacity;
    new_pending %= m_capacity;

    if (new_pending > pending) {
      segs.template emplace_back(m_data + pending, size);
    } else if (new_pending < pending) {
      segs.template emplace_back(m_data + pending, size - new_pending);
      if (new_pending > 0) {
        segs.template emplace_back(m_data, new_pending);
      }
    }
    m_pending[0] += size;

    return segs;
  }

  // N.B. Only one of the writers can invoke this method
  void CommitPending(const Event& event) {
    event.Wait(m_stream.cuda_stream);
    GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_end.data(), m_pending.data(),
                                      sizeof(size_t), cudaMemcpyHostToHost,
                                      m_stream.cuda_stream));
  }

  size_t GetPendingSize() {
    std::unique_lock<std::mutex> lock(m_mutex);

    return m_pending[0] - m_end[0];
  }

  size_t GetReadableSize() {
    std::unique_lock<std::mutex> lock(m_mutex);

    return m_end[0] - m_start[0];
  }

  size_t GetOccupiedSize() {
    std::unique_lock<std::mutex> lock(m_mutex);

    return m_pending[0] - m_start[0];
  }

  std::vector<Buffer<T>> GetReadableBuffer() {
    std::unique_lock<std::mutex> lock(m_mutex);
    std::vector<Buffer<T>> segs;

    size_t start = m_start[0];
    size_t end = m_end[0];
    size_t size = end - start;

    start %= m_capacity;
    end %= m_capacity;

    if (end > start) {
      segs.template emplace_back(m_data + start, size - end);
    } else if (end < start) {
      segs.template emplace_back(m_data + start, size - end);
      if (end > 0) {
        segs.template emplace_back(m_data, end);
      }
    }

    m_start = m_end;
    return segs;
  }

  void PrintOffsets() {
    std::unique_lock<std::mutex> lock(m_mutex);

    printf("Dev: %d, start %lu end %lu pending %lu requiring\n", m_dev,
           m_start[0] % m_capacity, m_end[0] % m_capacity, m_pending[0] % m_capacity);
  }
};

}  // namespace groute

#endif  // RING_BUFFER_H
