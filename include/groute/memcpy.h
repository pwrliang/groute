// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __GROUTE_MEMCPY_H
#define __GROUTE_MEMCPY_H

#include <cuda_runtime.h>
#include <groute/common.h>
#include <groute/event_pool.h>
#include <groute/internal/cuda_utils.h>
#include <groute/internal/pinned_allocation.h>
#include <groute/internal/worker.h>
#include <utils/stopwatch.h>

#include <cassert>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#define BATCHED_D2D_CAPACITY 160
#define BATCHED_D2D_PADDING 16
namespace groute {

typedef std::function<void(size_t, const Event&)> MemcpyCallback;

__global__ static void copyp2p(void* dest, const void* src, size_t size) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  using type = char;
  auto unit_size = sizeof(type);
  auto num_elems = size / unit_size;
  auto rest_bytes = size % unit_size;

  auto* dest_4 = (type*) dest;
  auto* src_4 = (const type*) src;

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

class MemcpyWork : public groute::internal::IWork {
 public:
  int src_dev_id;
  int dst_dev_id;

  size_t copy_bytes;
  size_t dst_size;

  const int fragment_size;

  void* src_buffer;
  void* dst_buffer;

  Event src_ready_event;
  Event dst_ready_event;

  cudaStream_t copy_stream;
  cudaEvent_t sync_event;

  MemcpyCallback completion_callback;

  std::atomic<int>& active_count;

  bool using_memcpy{};

  std::map<int, std::set<int>> topology{{0, {1, 2, 3, 6}}, {1, {0, 2, 3, 7}},
                                        {2, {0, 1, 3, 4}}, {3, {0, 1, 2, 5}},
                                        {4, {2, 5, 6, 7}}, {5, {3, 4, 6, 7}},
                                        {6, {0, 4, 5, 7}}, {7, {1, 4, 5, 6}}};

 private:
  EventPool& m_event_pool;

  void CheckParams() const {
    // Verifying since parameters are expected to be provided by multiple
    // resources

    assert(!Device::IsNull(src_dev_id));
    assert(!Device::IsNull(dst_dev_id));

    assert(src_buffer != nullptr);
    assert(dst_buffer != nullptr);

    assert(copy_stream != nullptr);
    assert(sync_event != nullptr);

    assert(copy_bytes <= dst_size);
  }

  void CopyAsync(void* dst_buffer, const void* src_buffer, size_t count) {
    if (!Device::IsHost(src_dev_id) && !Device::IsHost(dst_dev_id)) {
      // dev to dev
      int access;
      GROUTE_CUDA_CHECK(
          cudaDeviceCanAccessPeer(&access, src_dev_id, dst_dev_id));

      int curr;
      GROUTE_CUDA_CHECK(cudaGetDevice(&curr));

      auto& dsts = topology.at(src_dev_id);
      Stopwatch sw;

      sw.start();
      int prev_count = active_count.fetch_add(1);

      if (access && dsts.find(dst_dev_id) != dsts.end() && false) {
        dim3 bd(256, 1, 1);
        dim3 gd(round_up(count, bd.x), 1, 1);

        gd.x = std::min(gd.x, 768u);

        int blockSize = 0;
        int numBlocks = 0;

        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
        copyp2p<<<numBlocks, blockSize, 0, copy_stream>>>(dst_buffer,
                                                          src_buffer, count);
      } else {
        GROUTE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_buffer, dst_dev_id,
                                              src_buffer, src_dev_id, count,
                                              copy_stream));
      }
      std::cout << "active num: " << prev_count << " " << src_dev_id << "->"
                << dst_dev_id << " Stream: " << copy_stream << std::endl;
      //      if (prev_count > 0) {
      //        std::cout << prev_count << std::endl;
      //      }
      //      GROUTE_CUDA_CHECK(cudaStreamSynchronize(copy_stream));
      //      sw.stop();
      //
      //      int size_in_mb = count / 1024 / 1024;
      //      int bw = size_in_mb / (sw.ms() / 1000);
      //      if (size_in_mb > 2 && bw > 0 && bw < 10000) {
      //        std::cout << src_dev_id << "->" << dst_dev_id << " Bandwidth: "
      //        << bw
      //                  << std::endl;
      //      }
    } else if (Device::IsHost(src_dev_id)) {
      // host to dev
      GROUTE_CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src_buffer, count,
                                        cudaMemcpyHostToDevice, copy_stream));
    } else if (Device::IsHost(dst_dev_id)) {
      // dev to host
      GROUTE_CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src_buffer, count,
                                        cudaMemcpyDeviceToHost, copy_stream));
    } else {
      // host to host
      assert(false);  // TODO: std::memcpy(dst_buffer, src_buffer, count);
    }
  }

  void Complete(size_t bytes, const Event& ev) const {
    if (completion_callback)
      completion_callback(bytes, ev);
    active_count.fetch_sub(1);
  }

 public:
  MemcpyWork(EventPool& event_pool, std::atomic<int>& atomic,
             int fragment_size = -1)
      : m_event_pool(event_pool),
        src_dev_id(Device::Null),
        dst_dev_id(Device::Null),
        fragment_size(fragment_size),
        copy_bytes(0),
        dst_size(0),
        src_buffer(nullptr),
        dst_buffer(nullptr),
        copy_stream(nullptr),
        sync_event(nullptr),
        completion_callback(nullptr),
        active_count(atomic) {
#ifndef NDEBUG
    if (fragment_size < -1 || fragment_size == 0)
      throw std::invalid_argument("invalid value for fragment_size");
#endif
  }

  void operator()(groute::internal::Barrier* barrier) override {
#ifndef NDEBUG
    CheckParams();
#endif

    src_ready_event.Wait(copy_stream);
    dst_ready_event.Wait(copy_stream);

    if (fragment_size < 0)  // No fragmentation
    {
      CopyAsync(dst_buffer, src_buffer, copy_bytes);
    } else {
      // Fragmented Copy
      auto fragment = fragment_size;
      size_t pos = 0;

      while (pos < copy_bytes) {
        void* receive = ((void*) ((char*) dst_buffer + pos));
        void* send = ((void*) ((char*) src_buffer + pos));

        CopyAsync(receive, send,
                  (size_t) ((pos + fragment) > copy_bytes ? (copy_bytes - pos)
                                                          : fragment));

        pos += fragment;

        if (pos >= copy_bytes)
          break;  // Avoid syncing on last segment

        // We must sync the host thread in order to achieve real fragmentation
        //
        GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, copy_stream));
        GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
      }
    }

    Complete(copy_bytes, m_event_pool.Record(copy_stream));
  }
};

struct IMemcpyInvoker {
  virtual ~IMemcpyInvoker() {}
  virtual void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) = 0;
};

class MemcpyInvoker : public IMemcpyInvoker {
 protected:
  const int m_dev_id;  // the real dev id
  std::vector<cudaStream_t> m_copy_streams;
  std::vector<cudaEvent_t> m_sync_events;

 public:
  MemcpyInvoker(int dev_id) : m_dev_id(dev_id) {
    GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));
    int n_gpus;
    GROUTE_CUDA_CHECK(cudaGetDeviceCount(&n_gpus));

    for (int i = 0; i < n_gpus; i++) {
      cudaStream_t stream;
      GROUTE_CUDA_CHECK(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      m_copy_streams.push_back(stream);
      cudaEvent_t event;
      GROUTE_CUDA_CHECK(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      m_sync_events.push_back(event);
    }
  }

  virtual ~MemcpyInvoker() {
    for (auto& stream : m_copy_streams) {
      GROUTE_CUDA_CHECK(cudaStreamDestroy(stream));
    }
    for (auto& event : m_sync_events) {
      GROUTE_CUDA_CHECK(cudaEventDestroy(event));
    }
  }

  void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) override {
    assert(memcpy_work->fragment_size ==
           -1);  // this invoker does not support fragmentation

    int current_dev;
    GROUTE_CUDA_CHECK(cudaGetDevice(&current_dev));

    if (current_dev != m_dev_id)
      GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));

    int dst_dev_id = memcpy_work->dst_dev_id;

    memcpy_work->copy_stream = m_copy_streams[dst_dev_id];
    memcpy_work->sync_event = m_sync_events[dst_dev_id];

    (*memcpy_work)(nullptr);  // invoke

    if (current_dev != m_dev_id)  // set back to the correct device
      GROUTE_CUDA_CHECK(cudaSetDevice(current_dev));
  }
};

class MemcpyWorker : public groute::internal::Worker<MemcpyWork>,
                     public IMemcpyInvoker {
 private:
  const int m_dev_id;  // the real dev id
  cudaStream_t m_copy_stream;
  cudaEvent_t m_sync_event;

 protected:
  /// Called by the worker thread on start
  void OnStart() override {
    GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));
    GROUTE_CUDA_CHECK(
        cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking));
    GROUTE_CUDA_CHECK(
        cudaEventCreateWithFlags(&m_sync_event, cudaEventDisableTiming));
  }

  void OnBeginWork(std::shared_ptr<MemcpyWork> work) override {
    work->copy_stream = m_copy_stream;
    work->sync_event = m_sync_event;
  }

 public:
  explicit MemcpyWorker(int dev_id)
      : groute::internal::Worker<MemcpyWork>(nullptr), m_dev_id(dev_id) {
    this->Run();
  }

  virtual ~MemcpyWorker() {
    GROUTE_CUDA_CHECK(cudaStreamDestroy(m_copy_stream));
    GROUTE_CUDA_CHECK(cudaEventDestroy(m_sync_event));
  }

  void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) override {
    this->Enqueue(memcpy_work);
  }
};
}  // namespace groute

#endif  // __GROUTE_MEMCPY_H
