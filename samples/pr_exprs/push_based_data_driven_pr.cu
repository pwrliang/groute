//
// Created by liang on 2/16/18.
//
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <cub/grid/grid_barrier.cuh>
#include <cub/cub.cuh>
#include <gflags/gflags.h>
#include <utils/cuda_utils.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/cta_work.h>
#include "pr_common.h"

DECLARE_double(wl_alloc_factor_local);
DECLARE_double(threshold);
DECLARE_string(model);
DECLARE_uint32(max_iteration);
//#define OUTLINING

namespace pushbased_dd_pr {
    template<typename WorkSource,
            typename WorkTarget,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            WorkTarget work_target,
            TGraph graph,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup =
                round_up(work_size, blockDim.x) * blockDim.x; // We need all threads in active blocks to enter the loop

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> np_local = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);
                current_ranks[node] = 1.0 - ALPHA;  // Initial rank

                np_local.start = graph.begin_edge(node);
                np_local.size = graph.end_edge(node) - np_local.start;

                if (np_local.size > 0) // Skip zero-degree nodes
                {
                    rank_t update = ((1.0 - ALPHA) * ALPHA) / np_local.size; // Initial update
                    np_local.meta_data = update;
                }
                work_target.append_warp(node);
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    np_local,
                    [&graph, &residual](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                    }
            );
        }
    }

    template<typename TValue>
    __inline__ __device__ TValue warpReduce(TValue localSum) {
        localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

        return localSum;
    }

    template<template<typename> class TRankDatum>
    __device__ void PageRankCheck__Single__(TRankDatum<rank_t> current_ranks,
                                            rank_t *block_sum_buffer, rank_t *rtn_sum) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        const int SMEMDIM = blockDim.x / warpSize;
        __shared__ rank_t smem[32];

        uint32_t work_size = current_ranks.size;
        rank_t local_sum = 0;

        for (uint32_t node = 0 + tid; node < work_size; node += nthreads) {
            rank_t dist = current_ranks[node];
            if (dist != IDENTITY_ELEMENT)
                local_sum += dist;
        }

        local_sum = warpReduce(local_sum);

        // let every first thread in warps write out the warp-level sum to shared-memory
        if (laneIdx == 0) {
            smem[warpIdx] = local_sum;
        }

        /* "__syncthreads" ensures every first thread in warps in the same block to wait for write-job done
         * if we remove this sync, the threads of first warp will read the old values when last time written
         * The side-effect is : use old values to evaluate the termination which just causes later termination
         * I think this is ok
         */
//        __syncthreads();

        // let first warp get partial sum from shared-memory
        if (warpIdx == 0) {
            local_sum = (threadIdx.x < SMEMDIM) ? smem[threadIdx.x] : 0;
            // let first warp do the reduce again
            local_sum = warpReduce(local_sum);

            // let first thread in the first warp write out the block-level sum
            if (laneIdx == 0) {
                block_sum_buffer[blockIdx.x] = local_sum;
            }
        }

        // let the "master" thread to sum up the block-level sum
        if (tid == 0) {
            uint32_t sum = 0;
            for (int bid = 0; bid < gridDim.x; bid++) {
                sum += block_sum_buffer[bid];
            }
            *rtn_sum = sum;
        }
    }

    template<
            typename WorkSource, typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankAsyncKernelCTA__Single__(
            WorkSource work_source,
            WorkTarget work_target,
            TGraph graph,
            RankDatum<rank_t> current_ranks, //Pagerank values
            ResidualDatum<rank_t> residual //delta
    ) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.len();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, IDENTITY_ELEMENT};

            if (i < work_size) {
                index_t node = work_source.read(i);

                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                //filter out unnecessary computations
                if (res > 0) {
                    current_ranks[node] += res;

                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }

            //use other warps/blocks to help high-degree nodes
            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&work_target, &graph, &residual](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        if (prev <= EPSILON && prev + update > EPSILON) {
                            work_target.append_warp(dest);
                        }
                    }
            );
        }
    }

    template<
            typename WorkSource,
            typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankAsyncKernel__Single__(
            WorkSource work_source,
            WorkTarget work_target,
            TGraph graph,
            RankDatum<rank_t> current_ranks, //Pagerank values
            ResidualDatum<rank_t> residual //delta
    ) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.len();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {

            index_t node = work_source.read(i);
            rank_t old_delta = atomicExch(residual.get_item_ptr(node), 0);

            //filter out unnecessary computations
            if (old_delta > 0) {
                current_ranks[node] += old_delta;

                index_t begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                rank_t new_delta = ALPHA * old_delta / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; edge++) {
                    index_t dest = graph.edge_dest(edge);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), new_delta);

                    if (prev <= EPSILON && prev + new_delta > EPSILON) {
                        work_target.append_warp(dest);
                    }
                }
            }

        }
    }

    template<
            typename TGraph,
            typename Worklist,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankController__Single__(
            Worklist work_source,
            Worklist work_target,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            int max_iteration,
            int *running_flag,
            rank_t threshold,
            rank_t *block_sum_buffer,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;
        rank_t pr_sum;

        Worklist *in_wl = &work_source, *out_wl = &work_target;

        while (*running_flag) {
            PageRankAsyncKernelCTA__Single__(*in_wl,
                                             *out_wl,
                                             graph,
                                             current_ranks,
                                             residual);
            //For Sync or Async model, we both should add a barrier to wait for computation done
            grid_barrier.Sync();//Wait for other threads

            PageRankCheck__Single__(current_ranks, block_sum_buffer, &pr_sum);

            iteration++;

            if (tid == 0) {
                printf("%s Iter:%u In:%u Out:%u Current PR sum:%f Normed PR sum:%f\n", "Async",
                       iteration,
                       in_wl->len(),
                       out_wl->len(),
                       pr_sum,
                       pr_sum / graph.nnodes);
                //master thread reset the worklist
                in_wl->reset();
                // thread 0 notify other threads to exit.
                if (iteration >= max_iteration || out_wl->len() == 0) {
                    *running_flag = 0;
                }
            }

            utils::swap(in_wl, out_wl);

            grid_barrier.Sync();//make sure other threads can see "running_flag"
        }
    };

    struct Algo {
        static const char *NameLower() { return "pr"; }

        static const char *Name() { return "PR"; }


        template<
                typename TGraphAllocator, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static const std::vector<rank_t> &Gather(
                TGraphAllocator &graph_allocator, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            graph_allocator.GatherDatum(current_ranks);
            return current_ranks.GetHostData();
        }

        template<
                typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static std::vector<rank_t> Host(
                groute::graphs::host::CSRGraph &graph, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            return PageRankHost(graph);
        }

        static int Output(const char *file, const std::vector<rank_t> &ranks) {
            return PageRankOutput(file, ranks);
        }

        static int CheckErrors(std::vector<rank_t> &ranks, std::vector<rank_t> &regression) {
            return PageRankCheckErrors(ranks, regression);
        }
    };
}

bool PushBasedDataDrivenPR() {
    printf("PushBasedTopologyDrivenPR - %s model", FLAGS_model.data());
    typedef groute::Worklist<index_t> Worklist;
    const int device = 0;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> last_round_residual;

    groute::graphs::traversal::Context<pushbased_dd_pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(device);

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_round_residual);

    context.SyncDevice(device); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(device);
    int peak_clk = 1;
    GROUTE_CUDA_CHECK(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device));
    printf("Clock rate %d khz\n", peak_clk);

    groute::graphs::dev::CSRGraph dev_graph = dev_graph_allocator.DeviceObject();

    //prepare worklist
    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor_local;
    Worklist wl1(max_work_size), wl2(max_work_size);

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    pushbased_dd_pr::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                           (groute::dev::WorkSourceRange<index_t>(
                                                                                   dev_graph.owned_start_node(),
                                                                                   dev_graph.owned_nnodes()),
                                                                                   wl1.DeviceObject(),
                                                                                   dev_graph,
                                                                                   current_ranks.DeviceObject(),
                                                                                   residual.DeviceObject());

    stream.Sync();

    printf("max iterated rounds: %u\n", FLAGS_max_iteration);

    //for terminate check
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    utils::SharedValue<rank_t> d_out;


    Stopwatch sw(false);

#ifdef OUTLINING
    grid_dims = {FLAGS_grid_size, 1, 1};
    block_dims = {FLAGS_block_size, 1, 1};

    utils::SharedArray<rank_t> block_sum_buffer(grid_dims.x);
    utils::SharedValue<int> running_flag;
    utils::SharedValue<float> max_time, min_time;

    GROUTE_CUDA_CHECK(cudaMemset(block_sum_buffer.dev_ptr, 0, sizeof(rank_t) * block_sum_buffer.buffer_size));
    running_flag.set_val_H2D(1);
    max_time.set_val_H2D(0);
    min_time.set_val_H2D(INT_MAX);

    printf("grid_size:%d block_size:%d\n", grid_dims.x, block_dims.x);

    cub::GridBarrierLifetime gridBarrierLifetime;
    gridBarrierLifetime.Setup(grid_dims.x);

    printf("Outlining enabled\n");

    utils::SharedArray<uint32_t> rounds_at_end(grid_dims.x * block_dims.x);

    sw.start();

    pushbased_dd_pr::PageRankController__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
            wl1.DeviceObject(),
                    wl2.DeviceObject(),
                    dev_graph,
                    gridBarrierLifetime,
                    FLAGS_max_iteration,
                    running_flag.dev_ptr,
                    FLAGS_threshold,
                    block_sum_buffer.dev_ptr,
                    current_ranks.DeviceObject(),
                    residual.DeviceObject(),
                    last_round_residual.DeviceObject());

#else
    KernelSizing(grid_dims, block_dims, dev_graph.owned_nnodes());

    printf("Outlining disabled\n");
    printf("grid size %d block size %d\n", grid_dims.x, block_dims.x);

    int iteration = 0;
    bool running = true;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                           dev_graph.owned_nnodes());

    GROUTE_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    Worklist *in_wl = &wl1, *out_wl = &wl2;

    sw.start();

    while (running) {
        pushbased_dd_pr::PageRankAsyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                in_wl->DeviceObject(),
                        out_wl->DeviceObject(),
                        dev_graph,
                        current_ranks.DeviceObject(),
                        residual.DeviceObject());

        stream.Sync();


        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                               dev_graph.owned_nnodes());
        rank_t pr_sum = d_out.get_val_D2H();

        iteration++;

        printf("Iter:%u In:%u Out:%u Current PR sum:%f Normed PR sum:%f\n",
               iteration,
               in_wl->GetLength(stream),
               out_wl->GetLength(stream),
               pr_sum,
               pr_sum / dev_graph.owned_nnodes());
        if (out_wl->GetLength(stream) == 0) {
            running = false;
        }

        if (iteration >= FLAGS_max_iteration) {
            printf("maximum reached\n");
            break;
        }

        in_wl->ResetAsync(stream.cuda_stream);
        std::swap(in_wl, out_wl);

    }

    GROUTE_CUDA_CHECK(cudaFree(d_temp_storage));


#endif

    stream.Sync();

    sw.stop();


//    rounds_at_end.D2H();
//    std::vector<uint32_t> rounds_vec = rounds_at_end.host_vec;
//    uint32_t max_iter = 0, min_iter = UINT32_MAX;
//    for (uint32_t ind = 0; ind < rounds_vec.size(); ind++) {
//        max_iter = max(max_iter, rounds_vec[ind]);
//        min_iter = min(min_iter, rounds_vec[ind]);
//    }
//
//    printf("max iter %u min iter %u\n", max_iter, min_iter);

//    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
//                           dev_graph.owned_nnodes());
//
//    GROUTE_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
//
//    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
//                           dev_graph.owned_nnodes());
//    rank_t pr_sum = d_out.get_val_D2H();
//
//    printf("PR sum:%f\n", pr_sum / context.host_graph.nnodes);
    printf("%s model - PR done:%f ms.\n", FLAGS_model.data(), sw.ms());
//    if (pr_sum / context.host_graph.nnodes < FLAGS_threshold) {
//        printf("needs more iterations\n");
//        return false;
//    }

    // Gather
    auto gathered_output = pushbased_dd_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);
    if (FLAGS_output.length() != 0)
        pushbased_dd_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = pushbased_dd_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return pushbased_dd_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    }
    return true;
}
