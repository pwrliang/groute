//
// Created by liang on 3/12/18.
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
#include <groute/distributed_worklist.h>
#include <groute/cta_work.h>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <cub/grid/grid_barrier.cuh>
#include <cub/cub.cuh>
#include <gflags/gflags.h>
#include <boost/format.hpp>
#include <utils/cuda_utils.h>
#include <groute/graphs/traversal_algo.h>
#include "pr_common.h"

DECLARE_uint32(max_iteration);
DECLARE_double(threshold);
//#define OUTLINING
//#define ATOMIC

namespace pullbased_pr {
    const rank_t IDENTITY_ELEMENT = 0.0f;

    template<typename WorkSource,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> accumulated_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
            index_t node = work_source.get_work(ii);

            current_ranks[node] = IDENTITY_ELEMENT;
            accumulated_residual[node] = IDENTITY_ELEMENT;
            residual[node] = (1 - ALPHA);
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
    __device__ void PageRankCheck__Single__(cub::GridBarrier grid_barrier, TRankDatum<rank_t> current_ranks,
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

        if (laneIdx == 0)
            smem[warpIdx] = local_sum;
        __syncthreads();

        local_sum = (threadIdx.x < SMEMDIM) ? smem[threadIdx.x] : 0;

        if (warpIdx == 0)
            local_sum = warpReduce(local_sum);

        if (threadIdx.x == 0) {
            block_sum_buffer[blockIdx.x] = local_sum;
        }
        __threadfence();
        //here should use a barrier to make sure all threads written before master thread read the global memory
        grid_barrier.Sync();
        if (tid == 0) {
            double sum = 0;
            for (int bid = 0; bid < gridDim.x; bid++) {
                sum += block_sum_buffer[bid];
            }
            *rtn_sum = (rank_t) sum;
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankSyncKernelPull__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> residual_to_pull,
            ResidualDatum<rank_t> accumulated_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node);

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t src = graph.edge_src(edge);
                rank_t new_delta = residual_to_pull[src];
#ifdef ATOMIC
                atomicAdd(accumulated_residual.get_item_ptr(node), new_delta);
#else
                accumulated_residual[node] += new_delta; // accumulated new delta
#endif
            }

            // accumulated delta(produced at last round) to current ranks
#ifdef ATOMIC
            atomicAdd(current_ranks.get_item_ptr(node), residual[node]);
#else
            current_ranks[node] += residual[node];
#endif
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankSyncKernelPullCTA__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> residual_to_pull,
            ResidualDatum<rank_t> accumulated_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<index_t> local_work = {0, 0, 0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);

                index_t begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        in_degree = end_edge - begin_edge;

                if (in_degree > 0) {
                    local_work.start = begin_edge;
                    local_work.size = in_degree;
                    local_work.meta_data = node;
                }
                // accumulated delta(produced at last round) to current ranks
#ifdef ATOMIC
                atomicAdd(current_ranks.get_item_ptr(node), residual[node]);
#else
                current_ranks[node] += residual[node];
#endif
            }

            groute::dev::CTAWorkScheduler<index_t>::template schedule(
                    local_work,
                    [&graph, &residual_to_pull, &accumulated_residual]
                            (index_t edge, index_t node) {
                        index_t src = graph.edge_src(edge);
                        rank_t new_delta = residual_to_pull[src];
                        atomicAdd(accumulated_residual.get_item_ptr(node), new_delta);
                    }
            );
        }
    }

    template<
            typename TGraph,
            template<typename> class ResidualDatum,
            typename WorkSource>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankSyncKernelPush__Single__(
            WorkSource work_source,
            TGraph graph,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> residual_to_pull) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            index_t out_degree = graph.out_degree(node);
            if (out_degree > 0) {
                residual_to_pull[node] = ALPHA * residual[node] / out_degree;
            } else {
                residual_to_pull[node] = 0;
            }
        }
    };

    template<typename WorkSource, template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankSyncResidual(WorkSource work_source,
                              ResidualDatum<rank_t> residual,
                              ResidualDatum<rank_t> accumulated_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
#ifdef ATOMIC
            residual[node] = atomicExch(accumulated_residual.get_item_ptr(node), IDENTITY_ELEMENT);
#else
            residual[node] = accumulated_residual[node];
            accumulated_residual[node] = IDENTITY_ELEMENT;
#endif
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankController__Single__(
            WorkSource work_source,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            int max_iteration,
            int *running_flag,
            rank_t threshold,
            rank_t *block_sum_buffer,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> residual_to_pull,
            ResidualDatum<rank_t> accumulated_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;
        rank_t pr_sum;

        while (*running_flag) {
            // computing new_delta and push them into residual_to_pull

            /* PUSH STAGE */
            PageRankSyncKernelPush__Single__(work_source, graph, residual, residual_to_pull);

            grid_barrier.Sync();

            /* PULL STAGE */
            PageRankSyncKernelPullCTA__Single__(work_source,
                                                graph,
                                                current_ranks,
                                                residual,
                                                residual_to_pull,
                                                accumulated_residual);

            grid_barrier.Sync(); // ensure all residual are accumulated to current_ranks

            /* CHECK STAGE */
            PageRankCheck__Single__(grid_barrier, current_ranks, block_sum_buffer, &pr_sum);

            if (tid == 0) {
                printf("Iter:%u Current PR sum:%f Normed PR sum:%f\n", iteration, pr_sum,
                       pr_sum / work_source.get_size());
                // thread 0 notify other threads to exit.
                if (iteration >= max_iteration || pr_sum / work_source.get_size() >= threshold) {
                    *running_flag = 0;
                }
            }

            __threadfence();
            iteration++;
            grid_barrier.Sync();//make sure other threads can see "running_flag"

            /* DOUBLE BUFFER SYNC STAGE */
            PageRankSyncResidual(work_source, residual, accumulated_residual);

            grid_barrier.Sync();
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

bool PullBasedTopologyDrivenPR() {
    printf("PullBasedTopologyDrivenPR PR - %s model", "SYNC");
    const int device = 0;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> residual_to_pull;
    groute::graphs::single::NodeOutputDatum<rank_t> accumulated_residual;

    groute::graphs::traversal::Context<pullbased_pr::Algo> context(1);
    printf("Converting CSR to CSC ...\n");
    groute::graphs::host::CSCGraph host_graph(context.host_graph); // convert CSR format to CSC format

    groute::graphs::single::CSCGraphAllocator dev_graph_allocator(host_graph);

    context.SetDevice(device);

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, residual_to_pull, accumulated_residual);

    context.SyncDevice(device); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(device);
    int peak_clk = 1;
    GROUTE_CUDA_CHECK(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device));
    printf("Clock rate %d khz\n", peak_clk);

    groute::graphs::dev::CSCGraph dev_graph = dev_graph_allocator.DeviceObject();

    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    pullbased_pr::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                        (groute::dev::WorkSourceRange<index_t>(
                                                                                dev_graph.owned_start_node(),
                                                                                dev_graph.owned_nnodes()),
                                                                                dev_graph,
                                                                                current_ranks.DeviceObject(),
                                                                                residual.DeviceObject(),
                                                                                accumulated_residual.DeviceObject());

    stream.Sync();

    printf("max iterated rounds: %u\n", FLAGS_max_iteration);


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

    cub::GridBarrierLifetime gridBarrierLifetime;
    gridBarrierLifetime.Setup(grid_dims.x);

    printf("Outlining enabled\n");
    printf("grid size %d  %d\n", grid_dims.x, block_dims.x);

    sw.start();

    pullbased_pr::PageRankController__Single__
            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
            groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                  dev_graph.owned_nnodes()),
                    dev_graph,
                    gridBarrierLifetime,
                    FLAGS_max_iteration,
                    running_flag.dev_ptr,
                    FLAGS_threshold,
                    block_sum_buffer.dev_ptr,
                    current_ranks.DeviceObject(),
                    residual.DeviceObject(),
                    residual_to_pull.DeviceObject(),
                    accumulated_residual.DeviceObject());
#else
    printf("Outlining disabled\n");
    KernelSizing(grid_dims, block_dims, dev_graph.owned_nnodes());

    int iteration = 0;
    bool running = true;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    utils::SharedValue<double> d_out;


    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                           dev_graph.owned_nnodes());

    GROUTE_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    printf("grid size %d block size %d\n", grid_dims.x, block_dims.x);

    auto work_source = groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                             dev_graph.owned_nnodes());

    sw.start();

    while (running) {
        pullbased_pr::PageRankSyncKernelPush__Single__
                << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                               (work_source,
                                                       dev_graph,
                                                       residual.DeviceObject(),
                                                       residual_to_pull.DeviceObject());

        stream.Sync();

        pullbased_pr::PageRankSyncKernelPullCTA__Single__
                << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                work_source,
                        dev_graph,
                        current_ranks.DeviceObject(),
                        residual.DeviceObject(),
                        residual_to_pull.DeviceObject(),
                        accumulated_residual.DeviceObject());
        stream.Sync();

        pullbased_pr::PageRankSyncResidual << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                work_source,
                        residual.DeviceObject(),
                        accumulated_residual.DeviceObject());
        stream.Sync();

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                               dev_graph.owned_nnodes());
        double pr_sum = d_out.get_val_D2H();

        iteration++;

        printf("Iter:%u Current PR sum:%f Normed PR sum:%f\n", iteration, pr_sum,
               pr_sum / dev_graph.owned_nnodes());

        if (pr_sum / dev_graph.owned_nnodes() >= FLAGS_threshold) {
            running = false;
        } else if (iteration >= FLAGS_max_iteration) {
            printf("maximum reached\n");
            break;
        }
    }

    GROUTE_CUDA_CHECK(cudaFree(d_temp_storage));


#endif

    stream.Sync();

    sw.stop();

    printf("%s model - PR done:%f ms.\n", "SYNC", sw.ms());

    // Gather
    auto gathered_output = pullbased_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);
    if (FLAGS_output.length() != 0)
        pullbased_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = pullbased_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return pullbased_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    }
    return true;
}
