#include <groute/context.h>
#include <groute/multi_channel_distributed_worklist.h>
#include <groute/policy.h>
#include <groute/router.h>
#include <gtest/gtest.h>
#include <utils/app_skeleton.h>
__global__ void CountKernel(int* buffer, int count, int offset, int* bins) {
  int tid = TID_1D;
  if (tid < count) {
    atomicAdd(bins + buffer[tid] - offset, 1);
  }
}
namespace histogram {

struct SplitOps {
  __device__ __forceinline__ groute::SplitFlags on_receive(int work) {
    return ((work / m_seg_size) == m_seg_index) ? groute::SF_Take
                                                : groute::SF_Pass;
  }

  __device__ __forceinline__ groute::SplitFlags on_send(int work) {
    return ((work / m_seg_size) == m_seg_index) ? groute::SF_Take
                                                : groute::SF_Pass;
  }

  __device__ __forceinline__ int pack(int work) { return work; }

  __device__ __forceinline__ int unpack(int work) { return work; }

  __device__ __host__ SplitOps(int split_seg_index, int split_seg_size)
      : m_seg_index(split_seg_index), m_seg_size(split_seg_size) {}

 private:
  int m_seg_index;
  int m_seg_size;
};
}  // namespace histogram

void TestMultiChannelDistributedWorklistTest(size_t histo_size,
                                             size_t work_size) {
  int ngpus;
  GROUTE_CUDA_CHECK(cudaGetDeviceCount(&ngpus));
  size_t max_work_size = work_size;  // round_up(work_size, (size_t)ngpus);
  size_t num_exch_buffs = 4 * ngpus;
  size_t max_exch_size = work_size;  // round_up(max_work_size, num_exch_buffs);
  size_t histo_seg_size = histo_size / ngpus;
  histo_size = histo_seg_size * ngpus;

  groute::Context context(ngpus);

  groute::router::Router<int> input_router(
      context, groute::router::Policy::CreateScatterPolicy(groute::Device::Host,
                                                           range(ngpus)));
  groute::router::ISender<int>* input_sender =
      input_router.GetSender(groute::Device::Host);
  std::vector<std::unique_ptr<groute::router::IPipelinedReceiver<int>>>
      input_receivers;

  for (size_t i = 0; i < ngpus; ++i) {
    auto receiver = input_router.CreatePipelinedReceiver(i, max_work_size, 1);
    input_receivers.push_back(std::move(receiver));
  }

  std::cout << "receivers created" << std::endl;

  srand(static_cast<unsigned>(22522));
  std::vector<int> host_worklist;

  for (int ii = 0, count = work_size; ii < count; ++ii) {
    host_worklist.push_back((rand() * round_up(histo_size, RAND_MAX)) %
                            histo_size);
  }

  input_sender->Send(
      groute::Segment<int>(&host_worklist[0], host_worklist.size()),
      groute::Event());
  input_sender->Shutdown();

  std::cout << "initial data has been sent" << std::endl;

  std::vector<std::shared_ptr<groute::router::IRouter<int>>> exchange_routers;

  exchange_routers.push_back(std::make_shared<groute::router::Router<int>>(
      context, groute::router::Policy::CreateRingPolicy(ngpus)));
  //  exchange_routers.push_back(std::make_shared<groute::router::Router<int>>(
  //      context, groute::router::Policy::CreateRingPolicy(ngpus)));

    exchange_routers.push_back(std::make_shared<groute::router::Router<int>>(
        context, groute::router::SimplePolicy::CreateRingPolicy(
                     {0, 3, 2, 1, 7, 4, 5, 6})));
  //  exchange_routers.push_back(std::make_shared<groute::router::Router<int>>(
  //      context, groute::router::SimplePolicy::CreateRingPolicy(
  //                   {0, 6, 5, 4, 7, 1, 2, 3})));

  groute::MultiChannelDistributedWorklist<int, int> distributed_worklist(
      context, exchange_routers, ngpus);
  std::vector<
      std::unique_ptr<groute::IMultiChannelDistributedWorklistPeer<int, int>>>
      worklist_peers;

  std::vector<int*> dev_segs(ngpus);

  for (int i = 0; i < ngpus; ++i) {
    context.SetDevice(i);

    GROUTE_CUDA_CHECK(cudaMalloc(&dev_segs[i], histo_seg_size * sizeof(int)));
    GROUTE_CUDA_CHECK(cudaMemset(dev_segs[i], 0, histo_seg_size * sizeof(int)));

    worklist_peers.push_back(distributed_worklist.CreatePeer(
        i, histogram::SplitOps(i, histo_seg_size), max_work_size, max_exch_size,
        num_exch_buffs));
  }

  std::cout << "peers created" << std::endl;

  std::vector<std::thread> workers;
  groute::internal::Barrier barrier(ngpus);

  for (size_t i = 0; i < ngpus; ++i) {
    std::thread worker([&, i]() {
      context.SetDevice(i);
      groute::Stream stream = context.CreateStream(i);
      auto& worklist_peer = worklist_peers[i];
      auto input_fut = input_receivers[i]->Receive();
      auto input_seg = input_fut.get();

      std::cout << "Input size: " << input_seg.GetSegmentSize() << std::endl;
      // Add counter
      distributed_worklist.ReportWork(input_seg.GetSegmentSize());

      barrier.Sync();

      auto& input_worklist = worklist_peer->GetLocalInputWorklist();

      input_seg.Wait(stream.cuda_stream);
      worklist_peer->PerformSplitSend(input_seg, stream);

      while (true) {
        auto input_segs = worklist_peer->GetLocalWork(stream);
        size_t total_segs_size = 0;

        if (input_segs.empty())
          break;

        for (auto seg : input_segs) {
          dim3 count_block_dims(32, 1, 1);
          dim3 count_grid_dims(
              round_up(seg.GetSegmentSize(), count_block_dims.x), 1, 1);

          CountKernel<<<count_grid_dims, count_block_dims, 0,
                        stream.cuda_stream>>>(seg.GetSegmentPtr(),
                                              seg.GetSegmentSize(),
                                              i * histo_seg_size, dev_segs[i]);

          input_worklist.PopItemsAsync(seg.GetSegmentSize(),
                                       stream.cuda_stream);
          std::cout << "Dev: " << i << " Pop: " << seg.GetSegmentSize()
                    << std::endl;
          total_segs_size += seg.GetSegmentSize();
        }

        // report work because the tasks are consumed
        distributed_worklist.ReportWork(-(int) total_segs_size);
      }

      stream.Sync();
    });

    workers.push_back(std::move(worker));
  }

  for (size_t i = 0; i < ngpus; ++i) {
    // Join workers
    workers[i].join();
  }

  std::vector<int> regression_segs(histo_seg_size * ngpus, 0);
  std::vector<int> host_segs(histo_seg_size * ngpus);

  for (auto it : host_worklist) {
    ++regression_segs[it];
  }

  for (size_t i = 0; i < ngpus; ++i) {
    context.SetDevice(i);
    GROUTE_CUDA_CHECK(cudaMemcpy(&host_segs[i * histo_seg_size], dev_segs[i],
                                 histo_seg_size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
  }

  int over_errors = 0, miss_errors = 0;
  std::vector<int> over_error_indices, miss_error_indices;

  for (int i = 0; i < histo_size; ++i) {
    int hv = host_segs[i];
    int rv = regression_segs[i];

    if (hv > rv) {
      ++over_errors;
      over_error_indices.push_back(i);
    }

    else if (hv < rv) {
      ++miss_errors;
      miss_error_indices.push_back(i);
    }
  }

  ASSERT_EQ(0, over_errors + miss_errors);

  for (size_t i = 0; i < ngpus; ++i) {
    GROUTE_CUDA_CHECK(cudaFree(dev_segs[i]));
  }
}

TEST(Worklist, Ring_2) {
  TestMultiChannelDistributedWorklistTest(1024, 409600);
}
