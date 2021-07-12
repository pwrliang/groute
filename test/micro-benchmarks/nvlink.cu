#include <groute/communication.h>
#include <groute/event_pool.h>
#include <groute/link.h>
#include <groute/policy.h>
#include <groute/router.h>
#include <groute/worklist.h>
#include <thrust/device_vector.h>
#include <utils/cuda_utils.h>
#include <utils/stopwatch.h>

#include <cub/grid/grid_barrier.cuh>
#include <sstream>

DEFINE_int32(data_size, 1024, "Start with a specific number of GPUs");
DEFINE_int32(max_number, 10, "Start with a specific number of GPUs");
DEFINE_int32(chunk_size, 1024, "Start with a specific number of GPUs");
DEFINE_string(dst_dev, "0", "");
DEFINE_int32(nrings, 1, "number of rings");

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

#include <condition_variable>
#include <deque>
#include <mutex>

template <typename T>
class BlockingQueue {
 private:
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::deque<T> d_queue;

 public:
  void push(T const& value) {
    {
      std::unique_lock<std::mutex> lock(this->d_mutex);
      d_queue.push_front(value);
    }
    this->d_condition.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    this->d_condition.wait(lock, [=] { return !this->d_queue.empty(); });
    T rc(std::move(this->d_queue.back()));
    this->d_queue.pop_back();
    return rc;
  }
};

template <typename T>
void FilterAndSplitSegment(
    groute::Stream& stream, groute::Segment<T>& segment, int limit,
    std::vector<std::shared_ptr<groute::Worklist<T>>>& wl) {
  size_t seg_size = segment.GetSegmentSize();
  T* data = segment.GetSegmentPtr();
  size_t num_split = wl.size();

  for (int idx = 0; idx < num_split; idx++) {
    groute::dev::Worklist<T> d_dev_wl = wl[idx]->DeviceObject();

    LaunchKernel(stream, seg_size, [=] __device__() {
      auto tid = TID_1D;
      auto nthreads = TOTAL_THREADS_1D;

      for (int i = 0 + tid; i < seg_size; i += nthreads) {
        auto key = data[i].first;
        auto val = data[i].second;

        if (val < limit) {
          if (idx == key % num_split) {
            d_dev_wl.append_warp(thrust::make_pair(key, val + 1));
          }
        }
      }
    });
  }
}

void Rings() {
  int root = 0;
  int ngpus;
  int data_size = FLAGS_data_size;
  int chunk_size = FLAGS_chunk_size;
  int num_buffers = 5;
  int max_number = FLAGS_max_number;
  using data_type = thrust::pair<int, int>;

  GROUTE_CUDA_CHECK(cudaGetDeviceCount(&ngpus));

  groute::Context context(ngpus);
  auto policy = groute::router::Policy::CreateMultiRingsPolicy(ngpus);
  groute::router::Router<data_type> router(context, policy);
  std::vector<groute::Link<data_type>> out_links;
  std::vector<groute::Link<data_type>> in_links;
  std::vector<std::shared_ptr<BlockingQueue<groute::Segment<data_type>>>>
      segs_for_send;
  std::vector<std::vector<std::shared_ptr<groute::Worklist<data_type>>>> tmp_wl;

  tmp_wl.resize(ngpus);

  int nring = policy->GetRouteNum();
  std::cout << "Ring num: " << nring << std::endl;

  // Init
  for (int dev = 0; dev < ngpus; dev++) {
    GROUTE_CUDA_CHECK(cudaSetDevice(dev));
    in_links.emplace_back(router, dev, chunk_size, num_buffers);
    out_links.emplace_back(dev, router);
    segs_for_send.push_back(
        std::make_shared<BlockingQueue<groute::Segment<data_type>>>());
    for (int ring = 0; ring < nring; ring++) {
      tmp_wl[dev].push_back(
          std::make_shared<groute::Worklist<data_type>>(data_size));
    }
  }

  thrust::host_vector<data_type> host_buffer(data_size);

  for (int i = 0; i < host_buffer.size(); i++) {
    host_buffer[i] = thrust::make_pair(i, 0);
  }

  thrust::device_vector<data_type> dev_buffer(host_buffer.begin(),
                                              host_buffer.end());

  groute::Segment<data_type> seg(thrust::raw_pointer_cast(dev_buffer.data()),
                                 data_size);
  seg.metadata = 0;
  out_links[root].Send(seg, groute::Event());

  bool running = true;

  std::vector<std::thread> threads;
  std::atomic<int32_t> remain_elems(data_size);
  Stopwatch sw;

  sw.start();

  for (int dev = 0; dev < ngpus; dev++) {
    threads.emplace_back(
        [&](int curr_dev) {
          GROUTE_CUDA_CHECK(cudaSetDevice(curr_dev));
          groute::Stream stream(curr_dev);
          auto& in_link = in_links[curr_dev];
          auto& worklists = tmp_wl[curr_dev];

          while (running) {
            auto fut = in_link.Receive();
            auto seg = fut.get();
            // seg will be empty when router's shutdown method is invoked
            if (seg.Empty())
              break;

            // waiting for receive are done
            seg.Wait(stream.cuda_stream);

            for (auto& wl : worklists) {
              wl->ResetAsync(stream.cuda_stream);
            }

            int finished_size = seg.GetSegmentSize();

            FilterAndSplitSegment(stream, seg, max_number, worklists);

            groute::Event split_ev =
                context.RecordEvent(curr_dev, stream.cuda_stream);

            in_link.ReleaseBuffer(seg, split_ev);

            std::stringstream ss;
            ss << "Dev: " << curr_dev << " Seg size: ";
            for (int ring = 0; ring < worklists.size(); ring++) {
              auto split_seg = worklists[ring]->ToSeg(stream);
              ss << split_seg.GetSegmentSize() << " ";
              if (!split_seg.Empty()) {
                finished_size -= split_seg.GetSegmentSize();
                split_seg.metadata = 0;
                segs_for_send[curr_dev]->push(split_seg);
              }
            }
            ss << std::endl;

            std::cout << ss.str();

            int n_remain =
                remain_elems.fetch_sub(finished_size) - finished_size;

            if (n_remain <= 0) {
              running = false;
              std::cout << "Done!" << std::endl;
              // notify all senders to exit
              for (int i_dev = 0; i_dev < ngpus; i_dev++) {
                segs_for_send[i_dev]->push(groute::Segment<data_type>());
              }
              router.Shutdown();
            }
          }
        },
        dev);

    threads.emplace_back(
        [&](int curr_dev) {
          GROUTE_CUDA_CHECK(cudaSetDevice(curr_dev));
          groute::Stream stream(curr_dev);
          auto& out_link = out_links[curr_dev];

          while (running) {
            auto seg = segs_for_send[curr_dev]->pop();
            if (seg.Empty()) {
              std::cout << "Sender " << curr_dev << " exit." << std::endl;
              break;
            }
            auto sent_ev = out_link.Send(seg, groute::Event()).get();
            sent_ev.Sync();
          }
        },
        dev);
  }

  for (auto& th : threads) {
    th.join();
  }
  sw.stop();

  std::cout << "Running time: " << sw.ms() << std::endl;
}

void TestSR(std::vector<int>& dsts) {
  using data_type = int32_t;
  size_t size = 1024 * 1024 * 1024;
  size_t chunk_size = 8 * 1024 * 1024;
  size_t num_buffer = 10;
  float size_in_mb = size * sizeof(data_type) / 1024.0f / 1024;
  int src = 0;

  int ngpus;
  GROUTE_CUDA_CHECK(cudaGetDeviceCount(&ngpus));

  groute::Context context(ngpus);
  auto policy = groute::router::Policy::CreateScatterPolicy(src, dsts);
  groute::router::Router<data_type> router(context, policy);
  auto sender = router.GetSender(src);
  //  auto receiver = router.GetReceiver(dst);

  context.SetDevice(src);
  thrust::device_vector<data_type> send_data(size);
  std::map<int, std::unique_ptr<thrust::device_vector<data_type>>>
      recv_data_vec;
  std::map<int, std::unique_ptr<groute::router::IPipelinedReceiver<data_type>>>
      receivers;
  for (auto dst : dsts) {
    context.SetDevice(dst);
    recv_data_vec[dst] =
        std::make_unique<thrust::device_vector<data_type>>(size);
    receivers[dst] =
        router.CreatePipelinedReceiver(dst, chunk_size, num_buffer);
  }
  std::vector<std::thread> threads;
  Stopwatch sw;

  for (auto dst : dsts) {
    GROUTE_CUDA_CHECK(cudaSetDevice(dst));
    groute::Stream stream(dst);
    sw.start();
    GROUTE_CUDA_CHECK(cudaMemcpyAsync(
        thrust::raw_pointer_cast(recv_data_vec[dst]->data()),
        thrust::raw_pointer_cast(send_data.data()), sizeof(data_type) * size,
        cudaMemcpyDeviceToDevice, stream.cuda_stream));
    stream.Sync();
    sw.stop();
    std::cout << src << "->" << dst << " Size: " << size_in_mb << " MB"
              << " Copy time: " << sw.ms()
              << " Bandwidth: " << size_in_mb / (sw.ms() / 1000) << " MB/s"
              << std::endl;
  }

  sw.start();

  threads.emplace_back([&]() {
    context.SetDevice(src);
    groute::Segment<data_type> seg(thrust::raw_pointer_cast(send_data.data()),
                                   send_data.size());
    auto send_ready = sender->Send(seg, groute::Event()).get();
    send_ready.Sync();
    std::cout << "Shutdown router" << std::endl;
    router.Shutdown();
  });

  std::atomic<int> total_recv_size(0);

  for (auto dst : dsts) {
    threads.emplace_back([&, dst]() {
      context.SetDevice(dst);
      auto& pipeline_receiver = receivers[dst];
      size_t received = 0;

      while (true) {
        auto pending = pipeline_receiver->Receive().get();
        if (pending.Empty())
          break;
        pending.Sync();
        received += pending.GetSegmentSize();
        groute::Segment<data_type> seg(pending.GetSegmentPtr(),
                                       pending.GetSegmentSize());
        pipeline_receiver->ReleaseBuffer(seg, groute::Event());
      }
      total_recv_size += received;
      std::cout << "Dev " << dst << " done, received: "
                << received * sizeof(data_type) / 1024.0f / 1024 << " MB"
                << std::endl;
      //    auto pending = receiver
      //                       ->Receive(groute::Buffer<data_type>(
      //                                     thrust::raw_pointer_cast(recv_data.data()),
      //                                     recv_data.size()),
      //                                 groute::Event())
      //                       .get();
      //    pending.Sync();
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  sw.stop();

  std::cout << "Size: " << size_in_mb << " MB"
            << " Copy time: " << sw.ms() << " Bandwidth: "
            << total_recv_size / 1024.0 / 1024 / (sw.ms() / 1000) << " MB/s"
            << std::endl;
}

struct RankData {
  uint32_t node;
  float rank;

  __host__ __device__ __forceinline__ RankData(uint32_t node, float rank)
      : node(node), rank(rank) {}
  __host__ __device__ __forceinline__ RankData()
      : node(UINT_MAX), rank(-1.0f) {}
  __device__ __host__ __forceinline__ int operator%(int rhs) {
    return node % rhs;
  }
};

template <typename T>
void SplitSegment(groute::Stream& stream, groute::Segment<T>& input_seg,
                  size_t num_split,
                  std::vector<std::shared_ptr<groute::Worklist<T>>>& wl) {
  auto seg_size = input_seg.GetSegmentSize();
  auto* data = input_seg.GetSegmentPtr();

  for (int idx = 0; idx < num_split; idx++) {
    wl[idx]->ResetAsync(stream.cuda_stream);

    groute::dev::Worklist<T> d_dev_wl = wl[idx]->DeviceObject();

    LaunchKernel(stream, seg_size, [=] __device__() {
      auto tid = TID_1D;
      auto nthreads = TOTAL_THREADS_1D;

      for (int i = tid; i < seg_size; i += nthreads) {
        if (idx == data[i] % num_split) {
          d_dev_wl.append_warp(data[i]);
        }
      }
    });
  }
}

template <typename T>
__global__ void SplitKernel(cub::GridBarrier barrier,
                            groute::Segment<T> input_seg, int num_split,
                            groute::dev::Worklist<T>* output_wl) {
  auto tid = TID_1D;
  auto nthreads = TOTAL_THREADS_1D;
  auto seg_size = input_seg.GetSegmentSize();
  auto* data = input_seg.GetSegmentPtr();

  for (int idx = 0; idx < num_split; idx++) {
    if (tid == 0) {
      output_wl[idx].reset();
      barrier.Sync();
    }
    for (int i = tid; i < seg_size; i += nthreads) {
      if (idx == data[i] % num_split) {
        output_wl[idx].append_warp(data[i]);
      }
    }
    barrier.Sync();
  }
}

template <typename T>
void SplitSegmentFused(groute::Stream& stream,
                       cub::GridBarrierLifetime& barrier,
                       groute::Segment<T>& input_seg, size_t num_split,
                       thrust::device_vector<groute::dev::Worklist<T>>& wl) {
  auto seg_size = input_seg.GetSegmentSize();
  auto* data = input_seg.GetSegmentPtr();

  int fused_work_residency = 0;
  int BlockSize = 256;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fused_work_residency,
                                                SplitKernel<T>, BlockSize, 0);
  barrier.Setup(fused_work_residency);
  SplitKernel<<<fused_work_residency, BlockSize, 0, stream.cuda_stream>>>(
      barrier, input_seg, num_split, thrust::raw_pointer_cast(wl.data()));
}

void TestSplit() {
  groute::Stream stream;
  thrust::device_vector<RankData> ranks(10 * 1000 * 1000);
  int max_split = 4;
  std::vector<std::shared_ptr<groute::Worklist<RankData>>> wls;
  thrust::device_vector<groute::dev::Worklist<RankData>> d_wls;

  std::cout << "Data size: " << ranks.size() * sizeof(RankData) / 1024 / 1024
            << " MB" << std::endl;

  {
    thrust::host_vector<RankData> h_ranks(ranks.size());

    for (int i = 0; i < h_ranks.size(); i++) {
      h_ranks[i].node = i;
    }
    ranks = h_ranks;
  }

  for (int i = 0; i < max_split; i++) {
    auto wl = std::make_shared<groute::Worklist<RankData>>(ranks.size());
    wl->ResetAsync(stream.cuda_stream);
    wls.push_back(wl);
    d_wls.push_back(wl->DeviceObject());
  }

  groute::Segment<RankData> input_seg(thrust::raw_pointer_cast(ranks.data()),
                                      ranks.size());
  Stopwatch sw;

  for (int i = 1; i <= max_split; i++) {
    double total_time = 0;
    for (int _ = 0; _ < 100; _++) {
      sw.start();
      SplitSegment(stream, input_seg, i, wls);
      stream.Sync();
      sw.stop();
      total_time += sw.ms();
    }

    std::cout << "split num: " << i << " time: " << total_time / 100
              << std::endl;
  }

  cub::GridBarrierLifetime barrier;

  for (int i = 1; i <= max_split; i++) {
    double total_time = 0;
    for (int _ = 0; _ < 100; _++) {
      sw.start();
      SplitSegmentFused(stream, barrier, input_seg, i, d_wls);
      stream.Sync();
      sw.stop();
      total_time += sw.ms();
    }

    std::cout << "split num: " << i << " time: " << total_time / 100
              << std::endl;
  }
}

int main(int argc, char** argv) {
//  gflags::ParseCommandLineFlags(&argc, &argv, true);

//  TestSplit();
//  return 0;
  //  Rings();

  size_t size = 512 * 1024;
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

  std::vector<groute::Stream> streams(dst_devs.size());

  {
    double total_time = 0;
    size_t total_size = 0;
    Stopwatch sw;

    for (int iter = 0; iter < 100; iter++) {
      sw.start();
      for (int i = 0; i < dst_devs.size(); i++) {
        GROUTE_CUDA_CHECK(cudaMemcpyAsync(dst_ptrs[i], src_ptr, size,
                                          cudaMemcpyDeviceToDevice,
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
              << " Total size: " << (float) total_size / 1024 / 1024 << " MB "
              << " Bandwidth: " << (float) size / 1024 / 1024 / (sw.ms() / 1000)
              << " MB/s" << std::endl;
  }

  std::cout << std::endl;

//  TestSR(dst_devs);
}
